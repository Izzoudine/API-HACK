from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from api.forecast import forecast_extremes

app = FastAPI(
    title="Extreme Climate Forecast API",
    description="Predict extreme weather probabilities for a given activity, date, and location using NASA POWER + Prophet.",
    version="1.0"
)

# Autoriser les requêtes frontend (ex: Vue.js, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/forecast")
async def get_forecast(
    lat: float = Query(..., description="Latitude (e.g., 48.8566)"),
    lon: float = Query(..., description="Longitude (e.g., 2.3522)"),
    date: str = Query(..., description="Date future au format YYYY-MM-DD"),
    activities: str = Query(..., description="Activités séparées par des virgules (e.g., skiing,picnic)")
):
    """
    Exemple d'appel :
    http://127.0.0.1:8000/forecast?lat=48.8566&lon=2.3522&date=2025-12-31&activities=skiing,picnic
    """
    try:
        activities_list = [a.strip() for a in activities.split(",")]
        results = await forecast_extremes(lat, lon, date, activities_list, start_year=2010, end_year=2024)

        # Conversion des clés datetime -> string pour JSON
        if "results" in results:
            results["results"] = {str(k): v for k, v in results["results"].items()}

        return {
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "date": date,
            "activities": activities_list,
            "data": results
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
