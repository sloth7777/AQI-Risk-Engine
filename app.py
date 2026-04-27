from flask import Flask, request, jsonify, send_from_directory
from flask_bcrypt import Bcrypt

from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime

# NEW
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
bcrypt = Bcrypt(app)
# ── CONFIG ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "aqi_users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "super-secret-key"  # change in production

db = SQLAlchemy(app)
jwt = JWTManager(app)

model_path = os.path.join(BASE_DIR, "fiinal_model.pkl")
lookup_path = os.path.join(BASE_DIR, "city_month_lookup.csv")

# ── MODELS ────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    city = db.Column(db.String(50))
    predicted_aqi = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# CREATE DB
with app.app_context():
    db.create_all()

# ── LOAD MODEL ────────────────────────────────────────
try:
    model = joblib.load(model_path)
    MODEL_OK = True
    print("✅ Model loaded")
except Exception as e:
    model = None
    MODEL_OK = False
    print("❌ Model failed:", e)

# ── LOAD LOOKUP ───────────────────────────────────────
try:
    city_lookup = pd.read_csv(lookup_path)
    LOOKUP_OK = True
except:
    city_lookup = None
    LOOKUP_OK = False

# ── AQI LOGIC (UNCHANGED) ─────────────────────────────
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
    sub = {k: compute_sub_aqi(vals[i], BP[k]) for i, k in enumerate(keys)}

    dominant = max(sub, key=sub.get)
    return max(sub.values()), dominant, sub

def get_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy (Sensitive Groups)"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

# ── AUTH ROUTES ───────────────────────────────────────
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input format"}), 400

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing fields"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    user = User(email=email, password=hashed_password)

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User created"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()

    # auto-create user (your desired behavior)
    if not user:
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()

    # verify password
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Invalid credentials"}), 401

    # 🔥 THIS WAS MISSING (CRITICAL)
    token = create_access_token(identity=user.id)

    return jsonify({"token": token}), 200
# ── MAIN ROUTES ───────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    if not MODEL_OK:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    user_id = get_jwt_identity()

    pm25 = float(data.get("pm25") or 0)
    pm10 = float(data.get("pm10") or 0)
    no2  = float(data.get("no2")  or 0)
    co   = float(data.get("co")   or 0)
    so2  = float(data.get("so2")  or 0)
    o3   = float(data.get("o3")   or 0)
    city = str(data.get("city") or "")

    current_aqi, _, _ = calc_current_aqi(pm25, pm10, no2, co, so2, o3)

    X = np.array([[pm25, pm10, no2, co, so2, o3,
                   current_aqi, current_aqi, current_aqi,
                   current_aqi, current_aqi,
                   0, 0]])

    predicted_aqi = float(model.predict(X)[0])
    predicted_aqi = round(max(0, min(500, predicted_aqi)), 1)

    # SAVE
    pred = Prediction(
        user_id=user_id,
        city=city,
        predicted_aqi=predicted_aqi
    )
    db.session.add(pred)
    db.session.commit()

    return jsonify({
        "current_aqi": current_aqi,
        "predicted_aqi": predicted_aqi,
        "category": get_category(predicted_aqi),
        "timestamp": datetime.now().isoformat()
    })

# ── HISTORY (GRAPH DATA) ──────────────────────────────
@app.route("/history", methods=["GET"])
@jwt_required()
def history():
    user_id = get_jwt_identity()

    records = Prediction.query.filter_by(user_id=user_id)\
        .order_by(Prediction.timestamp).all()

    return jsonify([
        {
            "aqi": r.predicted_aqi,
            "time": r.timestamp.isoformat()
        }
        for r in records
    ])

# ── HEALTH ────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_OK})

# ── RUN ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)