# api/forecast.py
from vercel import Response
import json
from main import forecast_extremes
import asyncio

def handler(request):
    if request.method != "POST":
        return Response("Method Not Allowed", status=405)
    try:
        data = request.json
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        future_date = data.get("date")
        activities = data.get("activities", [])

        results = asyncio.run(forecast_extremes(lat, lon, future_date, activities))
        return Response(json.dumps({"status":"ok", "results":results}),
                        headers={"Content-Type": "application/json"})
    except Exception as e:
        return Response(json.dumps({"status":"error","message":str(e)}),
                        headers={"Content-Type": "application/json"},
                        status=500)
