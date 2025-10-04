from flask import Flask, request, jsonify
import asyncio
from datetime import datetime
# importer tes fonctions ici
from main import forecast_extremes  # mets le bon nom de fichier Python

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "API NASA Forecast prête 🚀"}

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.json
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        future_date = data.get("date")  # format "YYYY-MM-DD"
        activities = data.get("activities", [])

        # exécuter ta fonction async forecast_extremes
        results = asyncio.run(forecast_extremes(lat, lon, future_date, activities))

        return jsonify({"status": "ok", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
