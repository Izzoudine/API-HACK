# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
from api.forecast import forecast_extremes  # your function from earlier

app = FastAPI()

@app.post("/forecast")
async def forecast(request: Request):
    try:
        data = await request.json()
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        future_date = data.get("date")
        activities = data.get("activities", [])

        result = await forecast_extremes(lat, lon, future_date, activities)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
