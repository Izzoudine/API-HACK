# api/forecast.py
from fastapi import FastAPI
from pydantic import BaseModel
from main import forecast_extremes  # importe ta fonction existante
import asyncio

app = FastAPI()

class ForecastRequest(BaseModel):
    lat: float
    lon: float
    date: str
    activities: list

@app.get("/")
async def home():
    return {"message": "API NASA Forecast prête 🚀"}

@app.post("/")
async def forecast(req: ForecastRequest):
    try:
        # exécuter la fonction async
        results = await forecast_extremes(req.lat, req.lon, req.date, req.activities)
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
