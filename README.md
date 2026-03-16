# Login Risk Ensemble Predictor API

A combined FastAPI + Flask backend that loads three pre-trained `.pkl` models and exposes an **ensemble prediction** API to assess login-event risk in real time.

## Models Used

| Model File | Algorithm | Role | Features | Weight |
|---|---|---|---|---|
| `login_risk_model.pkl` | LightGBM (LGBMClassifier) | Primary classifier — highly precise binary risk prediction | 14 | 45% |
| `xgboost_optimized_model.pkl` | XGBoost (XGBClassifier) | Secondary classifier — tuned binary risk prediction | 17 | 35% |
| `isolation_model.pkl` | EllipticEnvelope (sklearn) | Anomaly detector — flags statistically unusual login events | 15 | 20% |

The final risk score is a weighted average of all three model probabilities, producing a single score between 0 and 1.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

The server starts on **http://127.0.0.1:8000** with the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | FastAPI home page with links |
| `/predict` | POST | FastAPI JSON prediction endpoint |
| `/docs` | GET | Swagger interactive API documentation |
| `/redoc` | GET | ReDoc API documentation |
| `/health` | GET | Health check — confirms all models are loaded |
| `/flask/` | GET | Flask-served HTML form (interactive UI) |
| `/flask/predict` | POST | Flask JSON prediction endpoint |

## API Usage

### POST `/predict` — FastAPI Endpoint

Send a JSON body describing the login event:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "round_trip_time_ms": 350.0,
    "asn": 15169,
    "device_type": "desktop",
    "login_successful": 1,
    "is_attack_ip": 0,
    "timestamp": "2026-03-16T10:30:00",
    "country": "US",
    "region": "California",
    "city": "Mountain View",
    "os_name_version": "Mac OS X 10.14.6",
    "browser_name_version": "Chrome 84.0.4147.338.339"
  }'
```

### Request Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `round_trip_time_ms` | float | Yes | Network round-trip time in milliseconds |
| `asn` | int | Yes | Autonomous System Number of the client IP |
| `device_type` | string | No | `bot`, `desktop`, `mobile`, `tablet`, or `unknown` (default: `unknown`) |
| `login_successful` | int | Yes | `1` if login succeeded, `0` if not |
| `is_attack_ip` | int | Yes | `1` if IP is flagged as malicious, `0` if not |
| `timestamp` | string | No | ISO-8601 datetime (defaults to current UTC time) |
| `country` | string | No | Country code or name (e.g. `US`, `NO`, `RU`) |
| `region` | string | No | Region or state (e.g. `Oslo County`, `California`) |
| `city` | string | No | City name (e.g. `Oslo`, `Mountain View`) |
| `os_name_version` | string | No | OS identifier (e.g. `Mac OS X 10.14.6`, `iOS 13.4`) |
| `browser_name_version` | string | No | Browser identifier (e.g. `Chrome 84.0.4147.338.339`) |

### Response Format

```json
{
  "ensemble": {
    "risk_score": 0.1261,
    "prediction": 0,
    "risk_level": "Low",
    "verdict": "Legitimate Login"
  },
  "models": {
    "lightgbm": {
      "probability": 0.0,
      "prediction": 0,
      "weight": 0.45
    },
    "xgboost": {
      "probability": 0.3603,
      "prediction": 0,
      "weight": 0.35
    },
    "elliptic_envelope": {
      "anomaly_score_raw": -85520681926.24,
      "risk_probability": 0.0,
      "prediction": 0,
      "raw_model_label": 1,
      "weight": 0.2
    }
  },
  "input_summary": {
    "round_trip_time_ms": 350.0,
    "device_type": "desktop",
    "country": "US",
    "is_attack_ip": 0,
    "login_successful": 1
  }
}
```

### Risk Levels

| Score Range | Level | Meaning |
|---|---|---|
| 0.00 – 0.39 | **Low** | Login appears legitimate |
| 0.40 – 0.64 | **Medium** | Some risk signals detected |
| 0.65 – 0.84 | **High** | Strong risk indicators |
| 0.85 – 1.00 | **Critical** | Very high confidence of malicious activity |

## Interactive Web Form

Visit **http://127.0.0.1:8000/flask/** to access a full-featured HTML form that lets you fill in login event details and see the ensemble prediction results with visual risk indicators, per-model probability breakdowns, and colour-coded severity badges.

## Architecture

```
User Input (JSON or HTML form)
        │
        ▼
┌─────────────────────────────┐
│     Feature Preprocessing    │
│  • Time extraction           │
│  • RTT capping/filling       │
│  • Device label encoding     │
│  • Frequency encoding        │
│    (Country, Region, City,   │
│     OS, Browser)             │
└────────┬────────────────────┘
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
┌────────┐ ┌────────┐ ┌───────────────┐
│LightGBM│ │XGBoost │ │ Elliptic      │
│(14 ft) │ │(17 ft) │ │ Envelope(15ft)│
│prob    │ │prob    │ │ anomaly score │
└───┬────┘ └───┬────┘ └──────┬────────┘
    │          │              │
    ▼          ▼              ▼
┌─────────────────────────────────┐
│   Weighted Ensemble Combiner    │
│   0.45·LGBM + 0.35·XGB + 0.20·AE│
└────────────┬────────────────────┘
             ▼
      Final Risk Score
   (0–1 with risk level)
```

## Project Structure

```
backend/
├── app.py                          # Main application (FastAPI + Flask)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── isolation_model.pkl             # EllipticEnvelope anomaly detection model
├── login_risk_model.pkl            # LightGBM classification pipeline
└── xgboost_optimized_model.pkl     # XGBoost binary classifier
```
