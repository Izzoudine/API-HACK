from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
from api.forecast import forecast_zone_extremes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Extreme Climate Forecast API",
    description="Predict extreme weather probabilities for multiple locations, dates, and activities using NASA POWER + Prophet.",
    version="1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for coordinates
class Coordinate(BaseModel):
    lat: float
    lon: float

# Pydantic model for request body
class ForecastRequest(BaseModel):
    coordinates: List[Coordinate]
    date: str
    activities: List[str]

@app.post("/forecast/bulk")
async def get_forecast_bulk(request: ForecastRequest = Body(...)):
    """
    Forecast weather for multiple lat/lon pairs.
    Example request body:
    {
        "coordinates": [{"lat": 48.8566, "lon": 2.3522}, {"lat": 40.7128, "lon": -74.0060}],
        "date": "2025-12-31",
        "activities": ["skiing", "picnic"]
    }
    """
    try:
        # Convert coordinates to list of [lat, lon] pairs
        points = [[coord.lat, coord.lon] for coord in request.coordinates]
        logger.info(f"Processing forecast for {len(points)} points, date={request.date}")
        forecast_result = await forecast_zone_extremes(
            points, request.date, request.activities, start_year=2010, end_year=2024
        )
        # Convert datetime keys to strings for JSON
        if "results" in forecast_result:
            forecast_result["results"] = {str(k): v for k, v in forecast_result["results"].items()}
        return {
            "status": "success",
            "coordinates": request.coordinates,
            "date": request.date,
            "activities": request.activities,
            "data": forecast_result
        }
    except Exception as e:
        logger.error(f"Error processing forecast: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/forecast")
async def get_forecast(
    lat: float = Query(..., description="Latitude (e.g., 48.8566)"),
    lon: float = Query(..., description="Longitude (e.g., 2.3522)"),
    date: str = Query(..., description="Date future au format YYYY-MM-DD"),
    activities: str = Query(..., description="Activités séparées par des virgules (e.g., skiing,picnic)")
):
    """
    Exemple d'appel:
    http://127.0.0.1:8000/forecast?lat=48.8566&lon=2.3522&date=2025-12-31&activities=skiing,picnic
    """
    try:
        activities_list = [a.strip() for a in activities.split(",")]
        # Pass single point as a list
        points = [[lat, lon]]
        forecast_result = await forecast_zone_extremes(points, date, activities_list, start_year=2010, end_year=2024)
        if "results" in forecast_result:
            forecast_result["results"] = {str(k): v for k, v in forecast_result["results"].items()}
        return {
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "date": date,
            "activities": activities_list,
            "data": forecast_result
        }
    except Exception as e:
        logger.error(f"Error processing forecast: {str(e)}")
        return {"status": "error", "message": str(e)}