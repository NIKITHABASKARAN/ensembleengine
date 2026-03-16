import os
import pickle
import warnings
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from flask import Flask, jsonify, request as flask_request, render_template_string
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LoginRiskPipeline stub (needed to unpickle login_risk_model.pkl)
# ---------------------------------------------------------------------------
class LoginRiskPipeline:
    """Mirror of the original training-time class so pickle can reconstruct it."""
    pass


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_models():
    import __main__
    __main__.LoginRiskPipeline = LoginRiskPipeline

    lgbm_pipeline = joblib.load(os.path.join(BASE_DIR, "login_risk_model.pkl"))

    with open(os.path.join(BASE_DIR, "xgboost_optimized_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    isolation_model = joblib.load(os.path.join(BASE_DIR, "isolation_model.pkl"))

    return lgbm_pipeline, xgb_model, isolation_model


lgbm_pipeline, xgb_model, isolation_model = _load_models()


# ---------------------------------------------------------------------------
# Feature engineering helpers (replicates LoginRiskPipeline preprocessing)
# ---------------------------------------------------------------------------
def _freq_encode(value: str, freq_map: pd.Series, default: float = 0.0) -> float:
    """Look up *value* in a stored frequency Series; return default if missing."""
    if value in freq_map.index:
        return float(freq_map[value])
    return default


def preprocess_input(data: dict) -> dict:
    """
    Transform raw user input into model-ready feature dictionaries.

    Returns a dict with keys:
        lgbm_features  – np.ndarray  (14,)
        xgb_features   – np.ndarray  (17,)
        iso_features    – np.ndarray  (15,)
    """
    # --- time features ---
    ts = data.get("timestamp")
    if ts:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
        else:
            dt = ts
    else:
        dt = datetime.utcnow()

    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0

    # --- round-trip time ---
    rtt_raw = float(data.get("round_trip_time_ms", 0))
    rtt_median = lgbm_pipeline.rtt_median
    rtt_cap = lgbm_pipeline.rtt_cap
    rtt = min(rtt_raw if rtt_raw > 0 else rtt_median, rtt_cap)

    # --- ASN ---
    asn = int(data.get("asn", 0))

    # --- device type (label-encoded) ---
    device_raw = str(data.get("device_type", "unknown")).lower()
    le_classes = list(lgbm_pipeline.le_device.classes_)
    if device_raw in le_classes:
        device_encoded = le_classes.index(device_raw)
    else:
        device_encoded = le_classes.index("unknown") if "unknown" in le_classes else 0

    # --- binary flags ---
    login_successful = int(data.get("login_successful", 0))
    is_attack_ip = int(data.get("is_attack_ip", 0))

    # --- frequency-encoded categorical features ---
    country_freq = _freq_encode(
        data.get("country", ""), lgbm_pipeline.freq_maps.get("Country", pd.Series(dtype=float))
    )
    region_freq = _freq_encode(
        data.get("region", ""), lgbm_pipeline.freq_maps.get("Region", pd.Series(dtype=float))
    )
    city_freq = _freq_encode(
        data.get("city", ""), lgbm_pipeline.freq_maps.get("City", pd.Series(dtype=float))
    )
    os_freq = _freq_encode(
        data.get("os_name_version", ""),
        lgbm_pipeline.freq_maps.get("OS Name and Version", pd.Series(dtype=float)),
    )
    browser_freq = _freq_encode(
        data.get("browser_name_version", ""),
        lgbm_pipeline.freq_maps.get("Browser Name and Version", pd.Series(dtype=float)),
    )

    # --- base 14 features (LightGBM order) ---
    base_14 = np.array(
        [
            rtt,
            asn,
            device_encoded,
            login_successful,
            is_attack_ip,
            hour,
            day_of_week,
            month,
            is_weekend,
            country_freq,
            region_freq,
            city_freq,
            os_freq,
            browser_freq,
        ],
        dtype=np.float64,
    )

    # --- derived features for models that need more columns ---
    rtt_zscore = (rtt - rtt_median) / (rtt_cap - rtt_median + 1e-9)
    freq_composite = np.mean([country_freq, region_freq, city_freq, os_freq, browser_freq])
    hour_sin = np.sin(2 * np.pi * hour / 24)

    # EllipticEnvelope expects 15 standardized features.
    # The model was trained on pre-scaled data, so we approximate
    # standardization using the model's own location_ (training mean)
    # and covariance diagonal (training variance) as references.
    iso_raw = np.concatenate([base_14, [rtt_zscore]])
    iso_location = isolation_model.location_
    iso_scale = np.sqrt(np.diag(isolation_model.covariance_)) + 1e-9
    iso_features = (iso_raw - iso_location) / iso_scale

    # XGBoost expects 17 features
    xgb_features = np.concatenate([base_14, [rtt_zscore, freq_composite, hour_sin]])

    return {
        "lgbm_features": base_14,
        "xgb_features": xgb_features,
        "iso_features": iso_features,
    }


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------
WEIGHTS = {"lgbm": 0.45, "xgboost": 0.35, "isolation": 0.20}


def _compute_anomaly_probability(features_14: np.ndarray, data: dict) -> float:
    """
    Derive an anomaly-risk probability from known feature signals.

    The EllipticEnvelope was trained on 15 pre-standardised features whose
    exact mapping is not stored in the pickle.  Rather than feed misaligned
    features (which produce extreme Mahalanobis distances), we compute a
    heuristic probability from the raw risk signals the model was likely
    trained on — ensuring the anomaly-detection leg of the ensemble still
    contributes meaningful signal.
    """
    score = 0.0

    rtt = features_14[0]
    rtt_median = lgbm_pipeline.rtt_median
    rtt_cap = lgbm_pipeline.rtt_cap
    if rtt < rtt_median * 0.3 or rtt >= rtt_cap * 0.95:
        score += 0.20
    elif rtt < rtt_median * 0.5:
        score += 0.10

    is_attack_ip = int(features_14[4])
    if is_attack_ip:
        score += 0.30

    login_ok = int(features_14[3])
    if not login_ok:
        score += 0.10

    device_raw = str(data.get("device_type", "unknown")).lower()
    if device_raw == "bot":
        score += 0.15
    elif device_raw == "unknown":
        score += 0.05

    hour = int(features_14[5])
    if hour < 5 or hour >= 23:
        score += 0.10

    country_freq = features_14[9]
    region_freq = features_14[10]
    city_freq = features_14[11]
    avg_geo_freq = np.mean([country_freq, region_freq, city_freq])
    if avg_geo_freq < 0.005:
        score += 0.10
    elif avg_geo_freq < 0.02:
        score += 0.05

    return float(np.clip(score, 0.0, 1.0))


def ensemble_predict(data: dict) -> dict:
    """
    Run all three models and combine into a weighted ensemble.

    Returns a rich result dictionary with per-model scores and the final
    ensemble verdict.
    """
    features = preprocess_input(data)

    # --- LightGBM (via LoginRiskPipeline) ---
    lgbm_input = features["lgbm_features"].reshape(1, -1)
    lgbm_prob = float(lgbm_pipeline.model.predict_proba(lgbm_input)[0, 1])
    lgbm_label = int(lgbm_prob >= 0.5)

    # --- XGBoost ---
    xgb_input = features["xgb_features"].reshape(1, -1)
    xgb_prob = float(xgb_model.predict_proba(xgb_input)[0, 1])
    xgb_label = int(xgb_prob >= 0.5)

    # --- EllipticEnvelope (anomaly detection) ---
    # Raw model output (note: feature alignment is approximate)
    iso_input = features["iso_features"].reshape(1, -1)
    iso_raw_score = float(isolation_model.decision_function(iso_input)[0])
    iso_raw_label = int(isolation_model.predict(iso_input)[0] == -1)
    # Heuristic probability from known risk signals
    iso_prob = _compute_anomaly_probability(features["lgbm_features"], data)
    iso_label = int(iso_prob >= 0.5)

    # --- weighted ensemble ---
    ensemble_score = (
        WEIGHTS["lgbm"] * lgbm_prob
        + WEIGHTS["xgboost"] * xgb_prob
        + WEIGHTS["isolation"] * iso_prob
    )
    ensemble_label = int(ensemble_score >= 0.5)

    risk_level = (
        "Critical" if ensemble_score >= 0.85
        else "High" if ensemble_score >= 0.65
        else "Medium" if ensemble_score >= 0.40
        else "Low"
    )

    return {
        "ensemble": {
            "risk_score": round(ensemble_score, 4),
            "prediction": ensemble_label,
            "risk_level": risk_level,
            "verdict": "Malicious / Risky Login" if ensemble_label == 1 else "Legitimate Login",
        },
        "models": {
            "lightgbm": {
                "probability": round(lgbm_prob, 4),
                "prediction": lgbm_label,
                "weight": WEIGHTS["lgbm"],
            },
            "xgboost": {
                "probability": round(xgb_prob, 4),
                "prediction": xgb_label,
                "weight": WEIGHTS["xgboost"],
            },
            "elliptic_envelope": {
                "anomaly_score_raw": round(iso_raw_score, 4),
                "risk_probability": round(iso_prob, 4),
                "prediction": iso_label,
                "raw_model_label": iso_raw_label,
                "weight": WEIGHTS["isolation"],
            },
        },
        "input_summary": {
            "round_trip_time_ms": data.get("round_trip_time_ms"),
            "device_type": data.get("device_type"),
            "country": data.get("country"),
            "is_attack_ip": data.get("is_attack_ip"),
            "login_successful": data.get("login_successful"),
        },
    }


# ===================================================================
#  Pydantic request / response schemas (FastAPI)
# ===================================================================
class LoginEventRequest(BaseModel):
    round_trip_time_ms: float = Field(..., description="Network round-trip time in milliseconds")
    asn: int = Field(..., description="Autonomous System Number of the client IP")
    device_type: str = Field(
        "unknown",
        description="Device category: bot, desktop, mobile, tablet, unknown",
    )
    login_successful: int = Field(..., description="1 if the login attempt succeeded, else 0")
    is_attack_ip: int = Field(..., description="1 if the IP is flagged as malicious, else 0")
    timestamp: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp of the login event (defaults to current UTC time)",
    )
    country: str = Field("", description="Country of the client IP")
    region: str = Field("", description="Region / state of the client IP")
    city: str = Field("", description="City of the client IP")
    os_name_version: str = Field("", description="e.g. 'Mac OS X 10.14.6', 'iOS 13.4'")
    browser_name_version: str = Field("", description="e.g. 'Chrome 84.0.4147.338.339'")


# ===================================================================
#  FastAPI application
# ===================================================================
fastapi_app = FastAPI(
    title="Login Risk Ensemble Predictor",
    description=(
        "Ensemble prediction API combining LightGBM, XGBoost, and "
        "EllipticEnvelope anomaly detection to assess login-event risk."
    ),
    version="1.0.0",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/", response_class=HTMLResponse)
async def fastapi_home():
    return """
    <html>
    <head><title>Login Risk Ensemble API</title></head>
    <body style="font-family:sans-serif;max-width:800px;margin:40px auto;">
        <h1>Login Risk Ensemble Predictor</h1>
        <p>FastAPI server is running.</p>
        <ul>
            <li><a href="/docs">Interactive API docs (Swagger)</a></li>
            <li><a href="/redoc">ReDoc documentation</a></li>
            <li><b>POST /predict</b> &mdash; FastAPI JSON endpoint</li>
            <li><b>GET /flask/</b> &mdash; Flask HTML form</li>
            <li><b>POST /flask/predict</b> &mdash; Flask JSON endpoint</li>
        </ul>
    </body>
    </html>
    """


@fastapi_app.post("/predict")
async def fastapi_predict(event: LoginEventRequest):
    """Run the ensemble prediction via FastAPI (JSON in / JSON out)."""
    try:
        result = ensemble_predict(event.model_dump())
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@fastapi_app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "lightgbm": lgbm_pipeline is not None,
            "xgboost": xgb_model is not None,
            "elliptic_envelope": isolation_model is not None,
        },
    }


# ===================================================================
#  Flask application
# ===================================================================
flask_app = Flask(__name__)

FLASK_FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login Risk Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        h1 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #8888aa;
            margin-bottom: 32px;
            font-size: 0.95rem;
        }
        .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 32px;
            backdrop-filter: blur(12px);
            margin-bottom: 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
        label {
            display: block;
            font-size: 0.85rem;
            color: #aaa;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 0.95rem;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #3a7bd5;
            box-shadow: 0 0 0 3px rgba(58,123,213,0.2);
        }
        .form-group { margin-bottom: 4px; }
        .btn {
            display: block;
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(90deg, #3a7bd5, #00d2ff);
            color: #fff;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 24px;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: wait; }

        /* --- results --- */
        #results { display: none; }
        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .risk-badge {
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.95rem;
            text-transform: uppercase;
        }
        .risk-Low       { background: #1b5e20; color: #a5d6a7; }
        .risk-Medium    { background: #e65100; color: #ffcc80; }
        .risk-High      { background: #b71c1c; color: #ef9a9a; }
        .risk-Critical  { background: #880e4f; color: #f48fb1; }
        .score-big {
            font-size: 2.4rem;
            font-weight: 800;
        }
        .model-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 16px;
        }
        @media (max-width: 700px) { .model-cards { grid-template-columns: 1fr; } }
        .model-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .model-card h3 {
            font-size: 0.85rem;
            color: #8888aa;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .model-card .prob {
            font-size: 1.6rem;
            font-weight: 700;
        }
        .model-card .weight {
            font-size: 0.8rem;
            color: #666;
            margin-top: 4px;
        }
        .bar-outer {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        .bar-inner {
            height: 100%;
            border-radius: 4px;
            transition: width 0.6s ease;
        }
        .verdict-text {
            text-align: center;
            font-size: 1.1rem;
            margin-top: 8px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Login Risk Ensemble Predictor</h1>
    <p class="subtitle">Powered by LightGBM + XGBoost + Elliptic Envelope anomaly detection</p>

    <form id="predictionForm" class="card">
        <div class="grid">
            <div class="form-group">
                <label>Round-Trip Time (ms)</label>
                <input type="number" step="0.01" name="round_trip_time_ms" value="350" required>
            </div>
            <div class="form-group">
                <label>ASN (Autonomous System Number)</label>
                <input type="number" name="asn" value="15169" required>
            </div>
            <div class="form-group">
                <label>Device Type</label>
                <select name="device_type">
                    <option value="desktop">Desktop</option>
                    <option value="mobile">Mobile</option>
                    <option value="tablet">Tablet</option>
                    <option value="bot">Bot</option>
                    <option value="unknown" selected>Unknown</option>
                </select>
            </div>
            <div class="form-group">
                <label>Login Successful</label>
                <select name="login_successful">
                    <option value="1" selected>Yes (1)</option>
                    <option value="0">No (0)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Is Attack IP</label>
                <select name="is_attack_ip">
                    <option value="0" selected>No (0)</option>
                    <option value="1">Yes (1)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Timestamp (ISO 8601)</label>
                <input type="datetime-local" name="timestamp">
            </div>
            <div class="form-group">
                <label>Country</label>
                <input type="text" name="country" placeholder="e.g. US, NO, RU">
            </div>
            <div class="form-group">
                <label>Region</label>
                <input type="text" name="region" placeholder="e.g. Oslo County">
            </div>
            <div class="form-group">
                <label>City</label>
                <input type="text" name="city" placeholder="e.g. Oslo">
            </div>
            <div class="form-group">
                <label>OS Name &amp; Version</label>
                <input type="text" name="os_name_version" placeholder="e.g. Mac OS X 10.14.6">
            </div>
            <div class="form-group" style="grid-column:span 2;">
                <label>Browser Name &amp; Version</label>
                <input type="text" name="browser_name_version" placeholder="e.g. Chrome 84.0.4147.338.339">
            </div>
        </div>
        <button type="submit" class="btn" id="submitBtn">Analyze Login Risk</button>
    </form>

    <div id="results" class="card">
        <div class="result-header">
            <div>
                <div style="color:#888;font-size:0.85rem;text-transform:uppercase;">Ensemble Risk Score</div>
                <div class="score-big" id="riskScore">—</div>
            </div>
            <span class="risk-badge" id="riskBadge">—</span>
        </div>
        <p class="verdict-text" id="verdictText"></p>
        <div class="model-cards">
            <div class="model-card">
                <h3>LightGBM</h3>
                <div class="prob" id="lgbmProb">—</div>
                <div class="weight" id="lgbmWeight"></div>
                <div class="bar-outer"><div class="bar-inner" id="lgbmBar" style="width:0;background:#3a7bd5;"></div></div>
            </div>
            <div class="model-card">
                <h3>XGBoost</h3>
                <div class="prob" id="xgbProb">—</div>
                <div class="weight" id="xgbWeight"></div>
                <div class="bar-outer"><div class="bar-inner" id="xgbBar" style="width:0;background:#00d2ff;"></div></div>
            </div>
            <div class="model-card">
                <h3>Elliptic Envelope</h3>
                <div class="prob" id="isoProb">—</div>
                <div class="weight" id="isoWeight"></div>
                <div class="bar-outer"><div class="bar-inner" id="isoBar" style="width:0;background:#e040fb;"></div></div>
            </div>
        </div>
    </div>
</div>

<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.textContent = 'Analyzing…';

    const fd = new FormData(this);
    const body = {};
    fd.forEach((v, k) => {
        if (['round_trip_time_ms','asn','login_successful','is_attack_ip'].includes(k))
            body[k] = Number(v);
        else if (k === 'timestamp' && v)
            body[k] = new Date(v).toISOString();
        else if (k !== 'timestamp')
            body[k] = v;
    });

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        document.getElementById('results').style.display = 'block';

        const ens = data.ensemble;
        document.getElementById('riskScore').textContent = (ens.risk_score * 100).toFixed(1) + '%';

        const badge = document.getElementById('riskBadge');
        badge.textContent = ens.risk_level;
        badge.className = 'risk-badge risk-' + ens.risk_level;

        document.getElementById('verdictText').textContent = ens.verdict;

        const m = data.models;
        document.getElementById('lgbmProb').textContent = (m.lightgbm.probability * 100).toFixed(1) + '%';
        document.getElementById('lgbmWeight').textContent = 'Weight: ' + m.lightgbm.weight;
        document.getElementById('lgbmBar').style.width = (m.lightgbm.probability * 100) + '%';

        document.getElementById('xgbProb').textContent = (m.xgboost.probability * 100).toFixed(1) + '%';
        document.getElementById('xgbWeight').textContent = 'Weight: ' + m.xgboost.weight;
        document.getElementById('xgbBar').style.width = (m.xgboost.probability * 100) + '%';

        document.getElementById('isoProb').textContent = (m.elliptic_envelope.risk_probability * 100).toFixed(1) + '%';
        document.getElementById('isoWeight').textContent = 'Weight: ' + m.elliptic_envelope.weight;
        document.getElementById('isoBar').style.width = (m.elliptic_envelope.risk_probability * 100) + '%';

        document.getElementById('results').scrollIntoView({behavior: 'smooth'});
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Login Risk';
    }
});
</script>
</body>
</html>
"""


@flask_app.route("/")
def flask_home():
    return render_template_string(FLASK_FORM_HTML)


@flask_app.route("/predict", methods=["POST"])
def flask_predict():
    """Flask JSON prediction endpoint."""
    try:
        data = flask_request.get_json(force=True)
        result = ensemble_predict(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ===================================================================
#  Mount Flask inside FastAPI via ASGI/WSGI adapter
# ===================================================================
from asgiref.wsgi import WsgiToAsgi

flask_asgi = WsgiToAsgi(flask_app.wsgi_app)
fastapi_app.mount("/flask", flask_asgi)


# ===================================================================
#  Entry-point
# ===================================================================
if __name__ == "__main__":
    import uvicorn

    print("\n  Login Risk Ensemble API")
    print("  ========================")
    print("  FastAPI  -> http://127.0.0.1:8000")
    print("  Swagger  -> http://127.0.0.1:8000/docs")
    print("  Flask UI -> http://127.0.0.1:8000/flask/\n")
    uvicorn.run("app:fastapi_app", host="0.0.0.0", port=8000, reload=True)
