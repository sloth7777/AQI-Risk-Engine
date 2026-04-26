from flask import Flask, request, jsonify, send_from_directory,render_template
import os
from flask_cors import CORS
import pickle, numpy as np, pandas as pd, os
from datetime import datetime
app = Flask(__name__)
CORS(app)
 


# ── Load model saved from your notebook ─────────────────────
try:
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    MODEL_OK = True
    print(" final_model.pkl loaded")
except FileNotFoundError:
    model    = None
    MODEL_OK = False
    print(" final_model.pkl not found — run the notebook first")
 
# ── Load city-month lookup for lag/rolling fallback ──────────
try:
    city_lookup = pd.read_csv("city_month_lookup.csv")
    LOOKUP_OK   = True
    print(" city_month_lookup.csv loaded")
except FileNotFoundError:
    city_lookup = None
    LOOKUP_OK   = False
    print("  city_month_lookup.csv not found — lags will mirror current AQI")
 
# ── Feature order MUST match training in notebook ────────────
FEATURES = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3",
    "AQI_lag_1", "AQI_lag_3", "AQI_lag_7",
    "AQI_roll_3", "AQI_roll_7",
    "winter_flag", "monsoon_flag"
]
 
# ── EPA sub-index (same logic as notebook + frontend) ────────
def compute_sub_aqi(val, bp):
    for c_lo, c_hi, i_lo, i_hi in bp:
        if val <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (val - c_lo) + i_lo)
    return 500
 
def calc_current_aqi(pm25, pm10, no2, co, so2, o3):
    BP = {
        "PM2.5": [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500,401,500)],
        "PM10":  [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)],
        "NO2":   [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)],
        "CO":    [(0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,40.4,301,400),(40.5,50.4,401,500)],
        "SO2":   [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300),(605,804,301,400),(805,1004,401,500)],
        "O3":    [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)],
    }
    vals = [pm25, pm10, no2, co, so2, o3]
    keys = list(BP.keys())
    sub  = {k: compute_sub_aqi(vals[i], BP[k]) for i, k in enumerate(keys)}
    dominant = max(sub, key=sub.get)
    return max(sub.values()), dominant, sub
 
def get_category(aqi):
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy (Sensitive Groups)"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"
 
# ── Persona advice — straight from your notebook ─────────────
def persona_advice(aqi, persona):
    if persona in ("asthma", "respiratory"):
        return ("High risk. Avoid going outside and use inhaler support if prescribed."
                if aqi > 100 else "Low risk, but avoid peak traffic hours.")
    elif persona == "child":
        return ("Children should avoid outdoor sports and playground activities."
                if aqi > 150 else "Short outdoor activity is safe.")
    elif persona in ("elderly", "senior"):
        return ("Stay indoors and avoid long walks."
                if aqi > 100 else "Safe for light outdoor movement.")
    else:
        return ("Wear a mask and reduce outdoor exposure."
                if aqi > 200 else "Normal activity allowed with awareness.")
 
def hospital_alert(aqi):
    if aqi > 200: return "High asthma emergency risk — increase pulmonology readiness."
    if aqi > 100: return "Prepare for increased respiratory OPD cases."
    return "Normal respiratory patient load expected."
 
 
# ── Routes ───────────────────────────────────────────────────
 
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_OK, "lookup_loaded": LOOKUP_OK})
 
 
@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_OK:
        return jsonify({"error": "Model not loaded. Run the notebook to generate final_model.pkl"}), 503
 
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
 
    try:
        pm25       = float(data.get("pm25", 0))
        pm10       = float(data.get("pm10", 0))
        no2        = float(data.get("no2",  0))
        co         = float(data.get("co",   0))
        so2        = float(data.get("so2",  0))
        o3         = float(data.get("o3",   0))
        city       = str(data.get("city", "")).strip()
        persona    = str(data.get("persona", "general"))
        is_winter  = int(data.get("is_winter",  0))
        is_monsoon = int(data.get("is_monsoon", 0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
 
    month = datetime.now().month
 
    # Current AQI via EPA formula
    current_aqi, dominant, sub_aqis = calc_current_aqi(pm25, pm10, no2, co, so2, o3)
 
    # Lag/rolling features — use city-month lookup if available
    lag_1 = lag_3 = lag_7 = roll_3 = roll_7 = float(current_aqi)
    if LOOKUP_OK and city:
        row = city_lookup[
            (city_lookup["City"].str.lower() == city.lower()) &
            (city_lookup["month"] == month)
        ]
        if not row.empty:
            h = float(row["AQI"].values[0])
            lag_1, lag_3, lag_7 = h*0.95, h*0.92, h*0.88
            roll_3, roll_7      = h*0.93, h*0.90
 
    # Feature vector
    X = np.array([[pm25, pm10, no2, co, so2, o3,
                   lag_1, lag_3, lag_7,
                   roll_3, roll_7,
                   is_winter, is_monsoon]])
 
    predicted_aqi = float(model.predict(X)[0])
    predicted_aqi = round(max(0, min(500, predicted_aqi)), 1)
 
    trend = ("improving" if predicted_aqi < current_aqi - 10 else
             "worsening" if predicted_aqi > current_aqi + 10 else "stable")
 
    return jsonify({
        "current_aqi":        current_aqi,
        "current_category":   get_category(current_aqi),
        "dominant_pollutant": dominant,
        "sub_aqis":           sub_aqis,
        "predicted_next_day": predicted_aqi,
        "predicted_category": get_category(predicted_aqi),
        "trend":              trend,
        "persona_advice":     persona_advice(predicted_aqi, persona),
        "hospital_alert":     hospital_alert(predicted_aqi),
        "timestamp":          datetime.now().isoformat()
    })
 
 
if __name__ == "__main__":
    print("\n AQI API running at http://localhost:5000")
    print("   POST /predict  — get prediction")
    print("   GET  /health   — model status\n")
    app.run(debug=True, port=5000)
 