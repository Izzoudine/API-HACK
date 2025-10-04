from fastapi import FastAPI
from pydantic import BaseModel
from main import forecast_extremes

app = FastAPI()

class ForecastRequest(BaseModel):
    lat: float
    lon: float
    date: str
    activities: list

@app.post("/")
async def forecast(req: ForecastRequest):
    results = await forecast_extremes(req.lat, req.lon, req.date, req.activities)
    return {"status": "ok", "results": results}
