# Login Risk Ensemble Predictor API

A combined FastAPI + Flask backend that loads pre-trained ML models and exposes an **ensemble prediction** API to assess login-event risk in real time. The system is split into two inference tiers — a **Fast Path** (cheap, always-on) and a **Deep Path** (expensive, triggered only when the Fast Path is uncertain) — with **Trust Entropy** quantifying how much the models disagree.

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
│ 0.45   │ │ 0.35   │ │  0.20         │
└───┬────┘ └───┬────┘ └──────┬────────┘
    │          │              │
    ▼          ▼              ▼
┌─────────────────────────────────┐
│     Fast Path Ensemble Score    │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │ 0.4 < s < 0.7│ ── No ──► Trust Entropy ──► Response
      └──────┬──────┘
             │ Yes
    ┌────────┴────────┐
    ▼                 ▼
┌────────────┐ ┌──────────────┐
│ LSTM       │ │ GraphSAGE    │
│ Sequential │ │ Link Predict │
│ (0.15)     │ │ (0.15)       │
└─────┬──────┘ └──────┬───────┘
      │               │
      ▼               ▼
┌─────────────────────────────────┐
│  Blended Score                  │
│  0.70·Fast + 0.15·LSTM + 0.15·G│
└────────────┬────────────────────┘
             │
             ▼
      Trust Entropy ──► Response
```

## Models Used

### Fast Path (always-on)

| Model File | Algorithm | Role | Features | Weight |
|---|---|---|---|---|
| `login_risk_model.pkl` | LightGBM (LGBMClassifier) | Primary classifier — highly precise binary risk prediction | 14 | 45% |
| `xgboost_optimized_model.pkl` | XGBoost (XGBClassifier) | Secondary classifier — tuned binary risk prediction | 17 | 35% |
| `isolation_model.pkl` | EllipticEnvelope (sklearn) | Anomaly detector — flags statistically unusual login events | 15 | 20% |

### Deep Path (triggered when Fast Path score is Medium: 0.4 < score < 0.7)

| Model File | Algorithm | Role | Weight (when active) |
|---|---|---|---|
| `lstm_action_model.pt` | 2-layer LSTM | Sequential anomaly — predicts next resource from user history | 15% |
| `graphsage_link_model.pt` | GraphSAGE (2-layer) | Relational anomaly — scores (user, device, resource) link plausibility | 15% |

When the Deep Path fires, the final score is blended: **70% Fast Path + 15% LSTM + 15% GraphSAGE**.

### Trust Entropy

All contributing model probabilities are normalised into a discrete distribution and scored with **Shannon Entropy**:

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

The raw entropy is divided by log₂(n) to produce a **normalised uncertainty score** in [0, 1]:

| Normalised Entropy | Uncertainty Level | Meaning |
|---|---|---|
| 0.00 – 0.29 | **Low** | Models strongly agree |
| 0.30 – 0.69 | **Medium** | Moderate disagreement |
| 0.70 – 1.00 | **High** | Models strongly disagree |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Deep Path Models (optional — the server runs without them)

```bash
python train_deep_models.py
```

This generates synthetic user-resource-device interaction data (500 users, 50 resources, 10 devices), trains the LSTM and GraphSAGE models, and saves four artefact files:

- `lstm_action_model.pt` — LSTM state dict
- `lstm_vocab.json` — resource vocabulary + hyper-parameters
- `graphsage_link_model.pt` — GraphSAGE state dict
- `graphsage_meta.json` — graph metadata + hyper-parameters

### 3. Run the Server

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
    "browser_name_version": "Chrome 84.0.4147.338.339",
    "user_id": "user_42",
    "resource_id": "resource_7"
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
| `user_id` | string | No | Opaque user identifier — enables Deep Path sequential analysis |
| `resource_id` | string | No | Resource being accessed — enables Deep Path behavioural models |

### Response Format

```json
{
  "ensemble": {
    "risk_score": 0.52,
    "prediction": 1,
    "risk_level": "Medium",
    "verdict": "Malicious / Risky Login",
    "uncertainty_score": 0.74,
    "uncertainty_level": "High"
  },
  "models": {
    "lightgbm": {
      "probability": 0.80,
      "prediction": 1,
      "weight": 0.45
    },
    "xgboost": {
      "probability": 0.36,
      "prediction": 0,
      "weight": 0.35
    },
    "elliptic_envelope": {
      "anomaly_score_raw": -85520681926.24,
      "risk_probability": 0.10,
      "prediction": 0,
      "raw_model_label": 1,
      "weight": 0.2
    }
  },
  "deep_path": {
    "triggered": true,
    "lstm_sequential": {
      "risk_score": 0.65
    },
    "gnn_link_predictor": {
      "risk_score": 0.45
    }
  },
  "input_summary": {
    "round_trip_time_ms": 350.0,
    "device_type": "desktop",
    "country": "US",
    "is_attack_ip": 0,
    "login_successful": 1,
    "user_id": "user_42",
    "resource_id": "resource_7"
  }
}
```

### Risk Levels

| Score Range | Level | Meaning |
|---|---|---|
| 0.00 – 0.39 | **Low** | Login appears legitimate |
| 0.40 – 0.64 | **Medium** | Some risk signals detected — Deep Path triggered if models available |
| 0.65 – 0.84 | **High** | Strong risk indicators |
| 0.85 – 1.00 | **Critical** | Very high confidence of malicious activity |

### Deep Path Behaviour

The Deep Path models are only invoked when **all** of the following conditions are met:

1. The Fast Path ensemble score falls in the Medium band (`0.4 < score < 0.7`)
2. `user_id` and `resource_id` are provided in the request
3. The LSTM and GraphSAGE model artefacts are present and loaded

If any condition is not met, `deep_path.triggered` will be `false` and the Deep Path model scores will be `null`. The final `risk_score` will be the unmodified Fast Path score.

## Interactive Web Form

Visit **http://127.0.0.1:8000/flask/** to access a full-featured HTML form that lets you fill in login event details (including the new `user_id` and `resource_id` fields) and see ensemble prediction results with visual risk indicators, per-model probability breakdowns, Deep Path results, Trust Entropy uncertainty, and colour-coded severity badges.

## Project Structure

```
backend/
├── app.py                          # Main application (FastAPI + Flask)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── train_deep_models.py            # Training script for LSTM + GraphSAGE
├── models/
│   ├── __init__.py                 # Package exports
│   ├── lstm_sequential.py          # LSTM model definition + scorer
│   ├── gnn_link_predictor.py       # GraphSAGE model definition + scorer
│   ├── deep_path.py                # Deep Path trigger orchestrator
│   └── trust_entropy.py            # Shannon Entropy implementation
├── isolation_model.pkl             # EllipticEnvelope anomaly detection model
├── login_risk_model.pkl            # LightGBM classification pipeline
├── xgboost_optimized_model.pkl     # XGBoost binary classifier
├── lstm_action_model.pt            # LSTM state dict (after training)
├── lstm_vocab.json                 # LSTM vocabulary + hyperparams (after training)
├── graphsage_link_model.pt         # GraphSAGE state dict (after training)
└── graphsage_meta.json             # GraphSAGE metadata (after training)
```
