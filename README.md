# 🌫️ AQI Risk Engine
### Forecasting and Respiratory Health Risk Assessment System

An end-to-end full-stack web application that predicts next-day Air Quality Index (AQI) using a Random Forest ML model, generates personalized health advisories, and provides hospital-level risk alerts — deployed live on Render.

---

## 📌 Overview

Most AQI systems are **reactive** — they tell you how bad the air is *right now*. The AQI Risk Engine is **proactive**: it takes current pollutant readings, calculates today's AQI using EPA-standard formulas, and forecasts **next-day AQI** using machine learning.

Beyond numbers, the system translates predictions into **actionable health guidance** tailored to the user's profile — making complex environmental data accessible to everyone.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Next-Day AQI Prediction** | Random Forest model forecasts AQI 24 hours ahead |
| 🧮 **Current AQI Calculation** | EPA-standard formula using pollutant sub-indices |
| 💊 **Personalized Health Advisory** | Tailored recommendations for asthma patients, elderly, general public |
| 🏥 **Hospital Alert System** | Classifies AQI into Normal / Moderate Risk / High Risk for healthcare preparedness |
| 📊 **Trend Visualization** | Chart.js line graphs showing AQI history over time |
| 🔐 **User Authentication** | JWT-based secure login/signup with per-user prediction history |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML, CSS, JavaScript, Chart.js |
| **Backend** | Python, Flask (REST API) |
| **Machine Learning** | Scikit-learn (Random Forest Regression), joblib |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite |
| **Authentication** | JWT (JSON Web Tokens) |
| **Deployment** | Render (Cloud Platform) |

---


## 🧠 Machine Learning Details

**Model:** Random Forest Regression (Scikit-learn)

**Input Features:**
- Raw pollutant concentrations: PM2.5, PM10, NO₂, CO, SO₂, O₃
- Calculated current AQI (EPA formula)
- Engineered features: lag values, rolling averages, seasonal indicators (Monsoon flag, Winter flag)

**Output:** Predicted next-day AQI (constrained to 0–500 standard range)

**Performance Metrics:** RMSE, MAE (update below with your results)

| Metric | Value |
|---|---|
| RMSE | 29.8 |
| MAE | 18..63 |
| Prediction Latency | < 2 seconds |

---

## 📊 AQI Risk Categories

| AQI Range | Category | Hospital Alert |
|---|---|---|
| 0 – 50 | Good | Normal |
| 51 – 100 | Moderate | Normal |
| 101 – 150 | Unhealthy for Sensitive Groups | Moderate Risk |
| 151 – 200 | Unhealthy | Moderate Risk |
| 201 – 300 | Very Unhealthy | High Risk |
| 301 – 500 | Hazardous | Emergency |

---

## 👥 User Profiles & Advisories

The system generates personalized advisories based on the selected user profile:

- **Asthma Patients** — Warnings about outdoor exposure, N95 mask recommendations
- **Elderly** — Precautionary measures, indoor air quality tips
- **General Public** — Activity guidance based on current AQI level

---

## 🧪 Test Cases

| TC | Scenario | Result |
|---|---|---|
| TC-01 | User Signup | ✅ Pass |
| TC-02 | User Login | ✅ Pass |
| TC-03 | AQI Prediction | ✅ Pass |
| TC-04 | Health Advisory Generation | ✅ Pass |
| TC-05 | AQI History Graph | ✅ Pass |

---

## ⚠️ Known Limitations

- Relies on **manual user inputs** rather than real-time API data
- Weather parameters (temperature, humidity) not yet integrated
- SQLite not suitable for large-scale concurrent usage
- No offline / mobile app support currently

---

## 🔮 Future Enhancements

- [ ] Integration with OpenAQ / WAQI real-time APIs
- [ ] Weather parameter integration (temperature, humidity, wind speed)
- [ ] Advanced models: LSTM, XGBoost
- [ ] Email/SMS notification system for AQI alerts
- [ ] Mobile application
- [ ] PostgreSQL migration for scalability


---

## 🙋 Author

**Shristi Singh**  

----
**Live Demo-** https://aqi-risk-engine.onrender.com/
