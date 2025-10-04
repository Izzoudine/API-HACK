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


# Activity-specific thresholds (unchanged)
ACTIVITY_THRESHOLDS = {
    "hiking": {
        "hot>25C": 30,
        "wet>5mm": 5,
        "weights": {"hot": 0.4, "wet": 0.6}
    },
    "cycling": {
        "windy>8": 8,
        "wet>3mm": 3,
        "weights": {"windy": 0.5, "wet": 0.5}
    },
    "picnicking": {
        "hot>28C": 28,
        "windy>6": 6,
        "wet>2mm": 2,
        "weights": {"hot": 0.3, "windy": 0.3, "wet": 0.4}
    }
}

# -------------------
# Data Fetching with Caching (Giovanni - Optional)
# -------------------
async def fetch_giovanni_series(lat, lon, start_year, end_year, variable="T2M", cache_dir="cache"):
    cache_file = f"{cache_dir}/giovanni_{variable}_{lat}_{lon}_{start_year}_{end_year}.pkl"
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    
    dataset_id = "M2T1NXSLV.5.12.4"
    service = "TimeSeries"
    bbox = f"{lon-0.05},{lat-0.05},{lon+0.05},{lat+0.05}"
    # Generate session ID via https://giovanni.gsfc.nasa.gov/giovanni/
    session_id = "YOUR_SESSION_ID"  # Replace with actual session ID
    df_list = []
    
    # Chunk by month to avoid server limits
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01T00:00:00Z"
            end_date = f"{year}-{month:02d}-28T23:59:59Z"  # Simplified; adjust for month end
            url = (f"https://giovanni.gsfc.nasa.gov/giovanni/daac-bin/service_manager?"
                   f"session={session_id}&service={service}&starttime={start_date}&"
                   f"endtime={end_date}&bbox={bbox}&"
                   f"data={urlp.quote(dataset_id + '/' + variable, safe='')}&format=CSV")
            
            async with aiohttp.ClientSession() as session:
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        async with session.get(url, timeout=20) as r:
                            r.raise_for_status()
                            csv_text = await r.text()
                            lines = csv_text.split("\n")
                            skiprows = next((i for i, line in enumerate(lines) if "," in line and not line.startswith("#")), 1)
                            df = pd.read_csv(io.StringIO(csv_text), skiprows=skiprows, header=None)
                            df.columns = ["date", variable]
                            df["date"] = pd.to_datetime(df["date"])
                            df[variable] = pd.to_numeric(df[variable], errors="coerce")
                            df = df[["date", variable]].dropna()
                            df_list.append(df)
                            break
                    except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
                        print(f"Attempt {attempt + 1} failed for Giovanni {variable} ({year}-{month:02d}): {e} (URL: {url})")
                        if attempt == 2:
                            print("Max retries reached; skipping.")
                            return pd.DataFrame()
                    except Exception as e:
                        print(f"Unexpected error for Giovanni {variable} ({year}-{month:02d}): {e} (URL: {url})")
                        return pd.DataFrame()
    
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        df.to_pickle(cache_file)
        return df
    return pd.DataFrame()

# -------------------
# Data Rods for Precipitation (Optional)
# -------------------
async def fetch_data_rods(lat, lon, start_year, end_year, variable="precipitationCal", cache_dir="cache"):
    cache_file = f"{cache_dir}/datarods_{variable}_{lat}_{lon}_{start_year}_{end_year}.pkl"
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    
    df_list = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            base_url = "https://hydro1.gesdisc.eosdis.nasa.gov/wdis/api/s4r?"
            params = {
                "variable": f"GPM_3IMERGDF/{variable}",
                "lat": str(lat),
                "lon": str(lon),
                "start": f"{year}-{month:02d}-01",
                "end": f"{year}-{month:02d}-28"  # Simplified
            }
            url = base_url + urlp.urlencode(params)
            
            async with aiohttp.ClientSession() as session:
                for attempt in range(3):
                    try:
                        async with session.get(url, timeout=20) as r:
                            r.raise_for_status()
                            ts_text = await r.text()
                            lines = ts_text.split("\n")
                            data_start = next((i for i, line in enumerate(lines) if "\t" in line and not line.startswith("#")), None)
                            if data_start is None:
                                raise ValueError(f"No data lines found for {year}-{month:02d}")
                            df = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), sep="\t", names=["time", "precipitation"])
                            df["date"] = pd.to_datetime(df["time"])
                            df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
                            df = df[["date", "precipitation"]].dropna()
                            df_list.append(df)
                            break
                    except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
                        print(f"Attempt {attempt + 1} failed for Data Rods ({year}-{month:02d}): {e} (URL: {url})")
                        if attempt == 2:
                            print("Max retries reached; skipping.")
                            continue
                    except Exception as e:
                        print(f"Unexpected error for Data Rods ({year}-{month:02d}): {e} (URL: {url})")
                        continue
    
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        df.to_pickle(cache_file)
        return df
    return pd.DataFrame()

# -------------------
# NASA POWER API (Default)
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
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
            print(f"Error fetching POWER data: {e} (URL: {url})")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# -------------------
# Compute Wind Speed from U/V Components (for Giovanni)
# -------------------
def compute_wind_speed(df_u, df_v):
    if df_u.empty or df_v.empty or 'date' not in df_u.columns or 'date' not in df_v.columns:
        print("Skipping wind computation: U2M or V2M data empty or missing 'date' column.")
        return pd.DataFrame(columns=["date", "WS2M"])
    
    df_merged = pd.merge(df_u, df_v, on="date", how="inner")
    if df_merged.empty:
        print("No overlapping dates for U2M and V2M; returning empty wind DF.")
        return pd.DataFrame(columns=["date", "WS2M"])
    df_merged["WS2M"] = np.sqrt(df_merged["U2M"]**2 + df_merged["V2M"]**2)
    return df_merged[["date", "WS2M"]]

# -------------------
# Prophet Forecast
# -------------------
def forecast_prophet(df, var, horizon_days, sample_freq="ME"):
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
    
    future = m.make_future_dataframe(periods=horizon_days)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst

# -------------------
# Monte Carlo & Probabilities
# -------------------
def monte_carlo_probs(fcst, target_date, activities, n_samples=1000):
    target_date = pd.to_datetime(target_date)
    row_temp = fcst["temp"].loc[fcst["temp"]["ds"] == target_date]
    row_wind = fcst["wind"].loc[fcst["wind"]["ds"] == target_date]
    row_prec = fcst["prec"].loc[fcst["prec"]["ds"] == target_date]
    
    if row_temp.empty or row_wind.empty or row_prec.empty:
        print("Skipping probabilities: Forecast data empty for target date.")
        return {}
    
    mu_t, lo_t, hi_t = row_temp["yhat"].values[0], row_temp["yhat_lower"].values[0], row_temp["yhat_upper"].values[0]
    mu_w, lo_w, hi_w = row_wind["yhat"].values[0], row_wind["yhat_lower"].values[0], row_wind["yhat_upper"].values[0]
    mu_p, lo_p, hi_p = row_prec["yhat"].values[0], row_prec["yhat_lower"].values[0], row_prec["yhat_upper"].values[0]
    
    sigma_t = (hi_t - lo_t) / (2 * 1.96)
    sigma_w = (hi_w - lo_w) / (2 * 1.96)
    sigma_p = (hi_p - lo_p) / (2 * 1.96)
    
    temp_samples = np.random.normal(mu_t, sigma_t, n_samples)
    wind_samples = np.random.normal(mu_w, sigma_w, n_samples)
    prec_samples = np.random.normal(mu_p, sigma_p, n_samples)
    
    results = {}
    for activity in activities:
        if activity not in ACTIVITY_THRESHOLDS:
            print(f"Warning: Activity '{activity}' not recognized. Skipping.")
            continue
        thresh = ACTIVITY_THRESHOLDS[activity]
        weights = thresh["weights"]
        probs = {}
        
        hot_key = next((k for k in thresh.keys() if k.startswith("hot>")), None)
        windy_key = next((k for k in thresh.keys() if k.startswith("windy>")), None)
        wet_key = next((k for k in thresh.keys() if k.startswith("wet>")), None)
        
        if hot_key and "hot" in weights:
            probs[f"too_hot_for_{activity}"] = np.mean(temp_samples > thresh[hot_key]) * 100
        if windy_key and "windy" in weights:
            probs[f"too_windy_for_{activity}"] = np.mean(wind_samples > thresh[windy_key]) * 100
        if wet_key and "wet" in weights:
            probs[f"too_wet_for_{activity}"] = np.mean(prec_samples > thresh[wet_key]) * 100
        
        discomfort_score = np.zeros(n_samples)
        if hot_key and "hot" in weights:
            discomfort_score += (temp_samples > thresh[hot_key]) * weights["hot"]
        if windy_key and "windy" in weights:
            discomfort_score += (wind_samples > thresh[windy_key]) * weights["windy"]
        if wet_key and "wet" in weights:
            discomfort_score += (prec_samples > thresh[wet_key]) * weights["wet"]
        probs[f"unsuitable_for_{activity}"] = np.mean(np.clip(discomfort_score, 0, 1.0)) * 100
        
        results[activity] = {k: float(v) for k, v in probs.items()}
    
    return results

# -------------------
# Main Forecast Function
# -------------------
async def forecast_extremes(lat, lon, future_date, activities, start_year=2024, end_year=2024):
    end_date = datetime(end_year, 12, 31)
    target_date = pd.to_datetime(future_date)
    horizon_days = (target_date - end_date).days + 1
    
    async def fetch_data():
        # Default to POWER API
        print("Attempting NASA POWER API (default)...")
        df_temp, df_wind, df_prec = await fetch_power_series(lat, lon, start_year, end_year)
        
        # Try Giovanni and Data Rods only if POWER fails
        if df_temp.empty or df_wind.empty or df_prec.empty:
            print("POWER API failed; trying Giovanni and Data Rods...")
            tasks = [
                fetch_giovanni_series(lat, lon, start_year, end_year, "T2M"),
                fetch_giovanni_series(lat, lon, start_year, end_year, "U2M"),
                fetch_giovanni_series(lat, lon, start_year, end_year, "V2M"),
                fetch_data_rods(lat, lon, start_year, end_year)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            df_temp, df_u, df_v, df_prec = results
            df_wind = compute_wind_speed(df_u, df_v)
        
        return df_temp, df_wind, df_prec
    
    df_temp, df_wind, df_prec = await fetch_data()
    
    if df_temp.empty or df_wind.empty or df_prec.empty:
        print("All data sources failed. Check API endpoints or authentication.")
        return {}
    
    fcst_temp = forecast_prophet(df_temp, "T2M", horizon_days)
    fcst_wind = forecast_prophet(df_wind, "WS2M", horizon_days)
    fcst_prec = forecast_prophet(df_prec, "precipitation", horizon_days)
    
    fcst = {"temp": fcst_temp, "wind": fcst_wind, "prec": fcst_prec}
    
    probs = monte_carlo_probs(fcst, future_date, activities)
    
    # Visualization: Chart for forecasts
    if not fcst["temp"].empty:
        labels = fcst["temp"]["ds"].dt.strftime('%Y-%m-%d').tolist()
        temp_data = fcst["temp"]["yhat"].tolist()
        wind_data = fcst["wind"]["yhat"].tolist() if not fcst["wind"].empty else [0] * len(labels)
        prec_data = fcst["prec"]["yhat"].tolist() if not fcst["prec"].empty else [0] * len(labels)
        
        print("\n```chartjs")
        print('''{
          "type": "line",
          "data": {
            "labels": ''' + str(labels) + ''',
            "datasets": [
              {
                "label": "Temperature (°C)",
                "data": ''' + str(temp_data) + ''',
                "borderColor": "#FF5733",
                "fill": false
              },
              {
                "label": "Wind Speed (m/s)",
                "data": ''' + str(wind_data) + ''',
                "borderColor": "#33FF57",
                "fill": false
              },
              {
                "label": "Precipitation (mm)",
                "data": ''' + str(prec_data) + ''',
                "borderColor": "#3357FF",
                "fill": false
              }
            ]
          },
          "options": {
            "scales": {
              "x": { "title": { "display": true, "text": "Date" } },
              "y": { "title": { "display": true, "text": "Value" } }
            }
          }
        }''')
        print("```")
    
    visualize_worldview(lat, lon, future_date)
    
    print(f"\nForecasted probabilities for {future_date} at lat={lat}, lon={lon}:")
    for activity, conditions in probs.items():
        print(f"\n{activity.capitalize()}:")
        for condition, prob in conditions.items():
            print(f"  {condition.replace('_', ' ')}: {prob:.1f}%")
    
    return probs

def visualize_worldview(lat, lon, date, variable="precipitation"):
    worldview_url = (f"https://worldview.earthdata.nasa.gov/?v={lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}"
                    f"&t={date}&l=GPM_3IMERG_Daily")
    print(f"Visualize data in Worldview: {worldview_url}")

# -------------------
# Example Usage
# -------------------
if __name__ == "__main__":
    lat, lon = 40.7128, -74.006  # New York City
    future_date = "2026-08-17"
    activities = ["hiking", "cycling", "picnicking"]
    
    async def main():
        results = await forecast_extremes(lat, lon, future_date, activities, start_year=2024, end_year=2024)
        return results
    
    asyncio.run(main())