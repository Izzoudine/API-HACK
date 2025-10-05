import aiohttp
import asyncio
import numpy as np
import pandas as pd
from prophet import Prophet
from datetime import datetime
import joblib
import os
import io
import urllib.parse as urlp
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from scipy.stats import beta

# -------------------
# Activity-specific thresholds
# -------------------

ACTIVITY_THRESHOLDS = {
    "skiing": {
        "hot>5C": 5, "cold<0C": 0, "wet>8mm": 8,
        "weights": {"hot": 0.4, "cold": 0.3, "wet": 0.3}
    },
    "picnic": {
        "hot>30C": 30, "cold<8C": 8, "windy>20kmh": 20, "wet>10mm": 10,
        "weights": {"hot": 0.3, "cold": 0.2, "windy": 0.25, "wet": 0.25}
    },
    "hiking": {
        "hot>35C": 35, "cold<5C": 5, "windy>25kmh": 25, "wet>15mm": 15,
        "weights": {"hot": 0.25, "cold": 0.25, "windy": 0.25, "wet": 0.25}
    },
    "cycling": {
        "hot>32C": 32, "cold<10C": 10, "windy>30kmh": 30, "wet>12mm": 12,
        "weights": {"hot": 0.3, "cold": 0.2, "windy": 0.3, "wet": 0.2}
    },
    "running": {
        "hot>28C": 28, "cold<5C": 5, "windy>25kmh": 25, "wet>10mm": 10,
        "weights": {"hot": 0.3, "cold": 0.3, "windy": 0.2, "wet": 0.2}
    },
    "beach": {
        "hot>25C": 25, "cold<18C": 18, "windy>15kmh": 15, "wet>5mm": 5,
        "weights": {"hot": 0.4, "cold": 0.2, "windy": 0.2, "wet": 0.2}
    },
    "fishing": {
        "hot>35C": 35, "cold<5C": 5, "windy>20kmh": 20, "wet>10mm": 10,
        "weights": {"hot": 0.2, "cold": 0.3, "windy": 0.3, "wet": 0.2}
    },
    "camping": {
        "hot>30C": 30, "cold<10C": 10, "windy>20kmh": 20, "wet>15mm": 15,
        "weights": {"hot": 0.25, "cold": 0.25, "windy": 0.25, "wet": 0.25}
    },
    # 🌊 Water activities
    "surfing": {
        "hot>35C": 35, "cold<18C": 18, "windy>40kmh": 40, "wet>5mm": 5,
        "weights": {"hot": 0.35, "cold": 0.25, "windy": 0.2, "wet": 0.2}
    },
    "kayaking": {
        "hot>38C": 38, "cold<15C": 15, "windy>35kmh": 35, "wet>5mm": 5,
        "weights": {"hot": 0.3, "cold": 0.2, "windy": 0.3, "wet": 0.2}
    },
    # ⚽ Sports
    "soccer": {
        "hot>33C": 33, "cold<8C": 8, "wet>15mm": 15, "windy>35kmh": 35,
        "weights": {"hot": 0.3, "cold": 0.2, "wet": 0.3, "windy": 0.2}
    },
    "tennis": {
        "hot>32C": 32, "cold<10C": 10, "wet>5mm": 5, "windy>30kmh": 30,
        "weights": {"hot": 0.3, "cold": 0.2, "wet": 0.3, "windy": 0.2}
    },
    "golf": {
        "hot>33C": 33, "cold<10C": 10, "wet>10mm": 10, "windy>25kmh": 25,
        "weights": {"hot": 0.25, "cold": 0.25, "wet": 0.25, "windy": 0.25}
    },
    # 🧗 Adventure
    "climbing": {
        "hot>32C": 32, "cold<5C": 5, "windy>35kmh": 35, "wet>10mm": 10,
        "weights": {"hot": 0.3, "cold": 0.2, "windy": 0.3, "wet": 0.2}
    },
    "snowboarding": {
        "hot>3C": 3, "cold<-15C": -15, "wet>10mm": 10,
        "weights": {"hot": 0.4, "cold": 0.4, "wet": 0.2}
    },
   
    # 🌍 Default fallback for any unknown activity
    "other": {
        "hot>35C": 35, "cold<5C": 5, "wet>15mm": 15, "windy>25kmh": 25,
        "weights": {"hot": 0.25, "cold": 0.25, "wet": 0.25, "windy": 0.25}
    }
}

# -------------------
# NASA POWER API
# -------------------
async def fetch_power_series(lat, lon, start_year, end_year, cache_dir="cache"):
    cache_file_temp = f"{cache_dir}/power_temp_{lat}_{lon}_{start_year}_{end_year}.pkl"
    cache_file_wind = f"{cache_dir}/power_wind_{lat}_{lon}_{start_year}_{end_year}.pkl"
    cache_file_prec = f"{cache_dir}/power_prec_{lat}_{lon}_{start_year}_{end_year}.pkl"
    os.makedirs(cache_dir, exist_ok=True)
    
    if (os.path.exists(cache_file_temp) and 
        os.path.exists(cache_file_wind) and 
        os.path.exists(cache_file_prec)):
        return (pd.read_pickle(cache_file_temp), 
                pd.read_pickle(cache_file_wind), 
                pd.read_pickle(cache_file_prec))
    
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters=T2M,WS2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&"
           f"start={start_date}&end={end_date}&format=JSON")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=20) as r:
                r.raise_for_status()
                data = await r.json()
                dates = pd.to_datetime(list(data["properties"]["parameter"]["T2M"].keys()), format="%Y%m%d")
                temp = list(data["properties"]["parameter"]["T2M"].values())
                wind = list(data["properties"]["parameter"]["WS2M"].values())
                prec = list(data["properties"]["parameter"]["PRECTOTCORR"].values())
                df_temp = pd.DataFrame({"date": dates, "T2M": temp}).dropna()
                df_wind = pd.DataFrame({"date": dates, "WS2M": wind}).dropna()
                df_prec = pd.DataFrame({"date": dates, "precipitation": prec}).dropna()
                df_temp.to_pickle(cache_file_temp)
                df_wind.to_pickle(cache_file_wind)
                df_prec.to_pickle(cache_file_prec)
                return df_temp, df_wind, df_prec
        except Exception as e:
            print(f"Error fetching POWER data: {e} (URL: {url})")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# -------------------
# Prophet Forecast
# -------------------
def forecast_prophet(df, var, horizon_days, sample_freq="D"):
    if df.empty or var not in df.columns:
        print(f"Skipping forecast for {var}: Empty data or missing column.")
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    
    df = df[["date", var]].set_index("date").resample(sample_freq).mean().reset_index()
    df_p = df.rename(columns={"date": "ds", var: "y"}).dropna()
    
    model_file = f"cache/prophet_model_{var}.pkl"
    if os.path.exists(model_file):
        m = joblib.load(model_file)
    else:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, n_changepoints=5)
        m.fit(df_p)
        joblib.dump(m, model_file)
    
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst

# -------------------
# Monte Carlo & Probabilities
# -------------------
def monte_carlo_probs(forecast, activities, n_samples=1000):
    results = {}
    rng = np.random.default_rng()

    df_temp = forecast["temp"].rename(
        columns={"yhat": "yhat_temp", "yhat_lower": "yhat_lower_temp", "yhat_upper": "yhat_upper_temp", "ds": "date"}
    )
    df_wind = forecast["wind"].rename(
        columns={"yhat": "yhat_wind", "yhat_lower": "yhat_lower_wind", "yhat_upper": "yhat_upper_wind", "ds": "date"}
    )
    df_prec = forecast["prec"].rename(
        columns={"yhat": "yhat_prec", "yhat_lower": "yhat_lower_prec", "yhat_upper": "yhat_upper_prec", "ds": "date"}
    )

    df = df_temp.merge(df_wind, on="date").merge(df_prec, on="date")
    
    for _, row in df.iterrows():
        date = row["date"]
        temp_mean, temp_std = row["yhat_temp"], (row["yhat_upper_temp"] - row["yhat_lower_temp"]) / 4
        wind_mean, wind_std = row["yhat_wind"], (row["yhat_upper_wind"] - row["yhat_lower_wind"]) / 4
        prec_mean, prec_std = row["yhat_prec"], (row["yhat_upper_prec"] - row["yhat_lower_prec"]) / 4

        temp_samples = rng.normal(temp_mean, temp_std, n_samples)
        wind_samples = rng.normal(wind_mean, wind_std, n_samples)
        prec_samples = rng.normal(prec_mean, prec_std, n_samples)

        day_result = {}
        for activity in activities:
            if activity not in ACTIVITY_THRESHOLDS:
                continue
            thresh = ACTIVITY_THRESHOLDS[activity]
            weights = thresh.get("weights", {})

            hot_key = next((k for k in thresh if k.startswith("hot>")), None)
            cold_key = next((k for k in thresh if k.startswith("cold<")), None)
            windy_key = next((k for k in thresh if k.startswith("windy>")), None)
            wet_key = next((k for k in thresh if k.startswith("wet>")), None)

            probs = {}
            if hot_key and "hot" in weights:
                probs[f"very_hot_for_{activity}"] = np.mean(temp_samples > thresh[hot_key]) * 100
            if cold_key and "cold" in weights:
                probs[f"very_cold_for_{activity}"] = np.mean(temp_samples < thresh[cold_key]) * 100
            if windy_key and "windy" in weights:
                probs[f"very_windy_for_{activity}"] = np.mean(wind_samples > thresh[windy_key]) * 100
            if wet_key and "wet" in weights:
                probs[f"very_wet_for_{activity}"] = np.mean(prec_samples > thresh[wet_key]) * 100

            discomfort = np.zeros(n_samples)
            if hot_key and "hot" in weights:
                discomfort += (temp_samples > thresh[hot_key]) * weights["hot"]
            if cold_key and "cold" in weights:
                discomfort += (temp_samples < thresh[cold_key]) * weights["cold"]
            if windy_key and "windy" in weights:
                discomfort += (wind_samples > thresh[windy_key]) * weights["windy"]
            if wet_key and "wet" in weights:
                discomfort += (prec_samples > thresh[wet_key]) * weights["wet"]

            max_score = sum(weights.values()) if weights else 1
            probs[f"very_uncomfortable_for_{activity}"] = np.mean(discomfort / max_score) * 100

            day_result[activity] = probs

        results[date] = day_result

    return results

# -------------------
# Visualization
# -------------------
# -------------------
# Visualization
# -------------------

def visualize_forecast_probs(probs, activity, variable):
    dates, values, stds = [], [], []
    for date, day_result in probs.items():
        if activity in day_result and variable in day_result[activity]:
            dates.append(date)
            val = day_result[activity][variable]
            values.append(val)
            stds.append(max(1, min(val, 100 - val) * 0.1))

    if not dates:
        return None

    df = pd.DataFrame({"date": dates, "prob": values, "std": stds})
    all_samples = []

    for val, std in zip(df["prob"], df["std"]):
        mean = np.clip(val / 100, 1e-3, 1 - 1e-3)
        var = (std / 100) ** 2
        var = max(var, 1e-6)
        alpha = ((1 - mean) / var - 1 / mean) * mean**2
        beta_param = alpha * (1 / mean - 1)
        samples = beta.rvs(alpha, beta_param, size=5000)
        all_samples.extend(samples * 100)

    plt.figure(figsize=(8, 5))
    sns.histplot(all_samples, bins=30, kde=True, color='skyblue')
    plt.axvline(np.mean(values), color='red', linestyle='--', label=f"Mean: {np.mean(values):.1f}%")
    plt.title(f"Probability Distribution: {variable.replace('_',' ')} ({activity})")
    plt.xlabel("Probability (%)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # Sauvegarde dans un buffer mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return img_b64
# -------------------
# Main Forecast Function
# -------------------
async def forecast_extremes(lat, lon, future_date, activities, start_year=2024, end_year=2024):
    end_date = datetime(end_year, 12, 31)
    target_date = pd.to_datetime(future_date)
    horizon_days = (target_date - end_date).days + 1

    df_temp, df_wind, df_prec = await fetch_power_series(lat, lon, start_year, end_year)
    if df_temp.empty or df_wind.empty or df_prec.empty:
        return {"error": "Failed to fetch data"}

    fcst = {
        "temp": forecast_prophet(df_temp, "T2M", horizon_days),
        "wind": forecast_prophet(df_wind, "WS2M", horizon_days),
        "prec": forecast_prophet(df_prec, "precipitation", horizon_days),
    }

    probs = monte_carlo_probs(fcst, activities)
    target_date = pd.to_datetime(future_date).date()
    target_results = {k: v for k, v in probs.items() if k.date() == target_date}

    output = {"results": target_results, "plots": {}}

    # Génération des images base64
    for date, day_result in target_results.items():
        for activity in day_result:
            for variable in day_result[activity]:
                img_b64 = visualize_forecast_probs({date: day_result}, activity, variable)
                if img_b64:
                    output["plots"][f"{activity}_{variable}"] = img_b64

    return output

# -------------------
# Example Usage
# -------------------
if __name__ == "__main__":
    lat, lon = 40.7128, -74.006  # New York City
    future_date = "2026-12-31"
    activities = ["skiing"]
    
    async def main():
        results = await forecast_extremes(lat, lon, future_date, activities, start_year=2010, end_year=2024)
        return results
    
    asyncio.run(main())
