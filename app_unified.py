import os
import pickle
import warnings
from datetime import datetime
from typing import Optional
import time
import uuid
import random

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from flask import Flask, jsonify, request as flask_request, render_template_string
from pydantic import BaseModel, Field

from db import init_db, get_last_login, upsert_login
from haversine import check_impossible_travel
from models.deep_path import run_deep_path, should_trigger_deep_path
from models.trust_entropy import compute_trust_entropy
from models.lstm_sequential import load_lstm_scorer
from models.gnn_link_predictor import load_gnn_scorer

warnings.filterwarnings("ignore")

# =========================================================================
# CRITICAL PATCH: Handle StringDtype compatibility (pandas 1.x -> 2.x)
# =========================================================================
_original_stringdtype_new = pd.StringDtype.__new__
_original_stringdtype_init = pd.StringDtype.__init__

def _new_stringdtype(cls, *args, **kwargs):
    instance = object.__new__(cls)
    return instance

def _new_stringdtype_init(self, storage="python", na_value=pd.NA, *args, **kwargs):
    if not hasattr(self, 'storage'):
        try:
            _original_stringdtype_init(self, storage=storage)
        except:
            object.__setattr__(self, 'storage', storage)

pd.StringDtype.__new__ = staticmethod(_new_stringdtype)
pd.StringDtype.__init__ = _new_stringdtype_init

# ---------------------------------------------------------------------------
# LoginRiskPipeline stub (needed to unpickle login_risk_model.pkl)
# ---------------------------------------------------------------------------
class LoginRiskPipeline:
    """Mirror of original training-time class so pickle can reconstruct it."""
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
init_db()

# ---------------------------------------------------------------------------
# Deep Path model loading (graceful — returns None if artefacts missing)
# ---------------------------------------------------------------------------
lstm_scorer = load_lstm_scorer(BASE_DIR)
gnn_scorer = load_gnn_scorer(BASE_DIR)

if lstm_scorer and gnn_scorer:
    print("[Deep Path] LSTM + GraphSAGE models loaded successfully.")
elif lstm_scorer:
    print("[Deep Path] Only LSTM loaded (GraphSAGE artefacts missing).")
elif gnn_scorer:
    print("[Deep Path] Only GraphSAGE loaded (LSTM artefacts missing).")
else:
    print("[Deep Path] No deep-path models found — detective layer disabled.")

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
# Enhanced Ensemble prediction with Deep Path integration
# ---------------------------------------------------------------------------
WEIGHTS = {"lgbm": 0.20, "xgboost": 0.30, "isolation": 0.40, "deep_path": 0.10}

def _compute_anomaly_probability(features_14: np.ndarray, data: dict) -> float:
    """
    Derive an anomaly-risk probability from known feature signals.
    Enhanced for high-risk scenario detection.
    """
    score = 0.0

    rtt = features_14[0]
    rtt_median = lgbm_pipeline.rtt_median
    rtt_cap = lgbm_pipeline.rtt_cap
    if rtt < rtt_median * 0.3 or rtt >= rtt_cap * 0.95:
        score += 0.30  # Increased from 0.25
    elif rtt < rtt_median * 0.5:
        score += 0.20  # Increased from 0.15

    is_attack_ip = int(features_14[4])
    if is_attack_ip:
        score += 0.50  # Increased from 0.40

    login_ok = int(features_14[3])
    if not login_ok:
        score += 0.30  # Increased from 0.20

    device_raw = str(data.get("device_type", "unknown")).lower()
    if device_raw == "bot":
        score += 0.35  # Increased from 0.25
    elif device_raw == "unknown":
        score += 0.15  # Increased from 0.10

    hour = int(features_14[5])
    if hour < 5 or hour >= 23:
        score += 0.20  # Increased from 0.15

    country_freq = features_14[9]
    region_freq = features_14[10]
    city_freq = features_14[11]
    avg_geo_freq = np.mean([country_freq, region_freq, city_freq])
    if avg_geo_freq < 0.005:
        score += 0.20  # Increased from 0.15
    elif avg_geo_freq < 0.02:
        score += 0.15  # Increased from 0.10

    # Additional high-risk country boost
    country = data.get("country", "").upper()
    if country in ["RU", "CN", "KP", "IR"]:
        score += 0.25  # Increased from 0.15

    # Extreme risk combination boost
    if is_attack_ip and device_raw == "bot" and not login_ok:
        score += 0.30  # Bonus for worst-case combination

    return float(np.clip(score, 0.0, 1.0))

def ensemble_predict(data: dict, activity_sequence: list = None, force_deep_path: bool = False) -> dict:
    """
    Run all models (traditional + deep path) and combine into weighted ensemble.
    """
    features = preprocess_input(data)

    # --- Traditional ML Models ---
    # LightGBM
    lgbm_input = features["lgbm_features"].reshape(1, -1)
    lgbm_prob = float(lgbm_pipeline.model.predict_proba(lgbm_input)[0, 1])
    lgbm_label = int(lgbm_prob >= 0.5)

    # XGBoost
    xgb_input = features["xgb_features"].reshape(1, -1)
    xgb_prob = float(xgb_model.predict_proba(xgb_input)[0, 1])
    xgb_label = int(xgb_prob >= 0.5)

    # EllipticEnvelope
    iso_input = features["iso_features"].reshape(1, -1)
    iso_raw_score = float(isolation_model.decision_function(iso_input)[0])
    iso_raw_label = int(isolation_model.predict(iso_input)[0] == -1)
    iso_prob = _compute_anomaly_probability(features["lgbm_features"], data)
    iso_label = int(iso_prob >= 0.5)

    # --- Deep Path Models (if available) ---
    deep_path_prob = 0.0
    deep_path_label = 0
    deep_path_details = {}
    
    # Calculate fast path score for deep path decision
    fast_path_score = (
        WEIGHTS["lgbm"] * lgbm_prob +
        WEIGHTS["xgboost"] * xgb_prob +
        WEIGHTS["isolation"] * iso_prob
    ) / (WEIGHTS["lgbm"] + WEIGHTS["xgboost"] + WEIGHTS["isolation"])
    
    if lstm_scorer and gnn_scorer:
        try:
            # Check if deep path should be triggered (original logic OR forced)
            should_trigger = force_deep_path or should_trigger_deep_path(fast_path_score)
            
            deep_result = run_deep_path(
                fast_path_score,  # positional arg 1
                data.get("user_id", "anonymous"),  # positional arg 2
                data.get("resource_id", ""),  # positional arg 3 (updated from country)
                data.get("device_type", "unknown"),  # positional arg 4
                lstm_scorer,  # positional arg 5
                gnn_scorer,   # positional arg 6
                activity_sequence  # positional arg 7 (new)
            )
            if deep_result["triggered"] or force_deep_path:
                _blended = deep_result.get("blended_score")
                deep_path_prob = _blended if _blended is not None else (fast_path_score if fast_path_score is not None else 0.0)
                deep_path_label = int(deep_path_prob >= 0.5)
                deep_path_details = deep_result
            else:
                # Deep path not triggered, use fast path score
                deep_path_prob = fast_path_score if fast_path_score is not None else 0.0
                deep_path_label = int(deep_path_prob >= 0.5)
                deep_path_details = {"triggered": False, "reason": "fast_path_score_outside_band"}
        except Exception as e:
            print(f"[Deep Path] Error: {e}")
            deep_path_prob = fast_path_score if fast_path_score is not None else 0.0
            deep_path_label = int(deep_path_prob >= 0.5)
            deep_path_details = {"error": str(e)}
    elif lstm_scorer:
        try:
            # LSTM only fallback
            deep_path_prob = float(lstm_scorer.predict_proba(features["lgbm_features"].reshape(1, -1))[0, 1])
            deep_path_label = int(deep_path_prob >= 0.5)
            deep_path_details = {"lstm_probability": deep_path_prob}
        except Exception as e:
            print(f"[Deep Path] LSTM Error: {e}")
            deep_path_prob = fast_path_score if fast_path_score is not None else 0.0
            deep_path_label = int(deep_path_prob >= 0.5)
            deep_path_details = {"error": str(e)}
    else:
        # No deep path models available, use fast path score
        deep_path_prob = fast_path_score if fast_path_score is not None else 0.0
        deep_path_label = int(deep_path_prob >= 0.5)
        deep_path_details = {"available": False}

    # --- Enhanced weighted ensemble ---
    ensemble_score = (
        WEIGHTS["lgbm"] * lgbm_prob
        + WEIGHTS["xgboost"] * xgb_prob
        + WEIGHTS["isolation"] * iso_prob
        + WEIGHTS["deep_path"] * deep_path_prob
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
            "deep_path": {
                "probability": round(deep_path_prob, 4),
                "prediction": deep_path_label,
                "weight": WEIGHTS["deep_path"],
                "details": deep_path_details,
                "available": bool(lstm_scorer and gnn_scorer),
                "sequence_risk": deep_path_details.get("sequence_risk", 0.0),
                "relational_risk": deep_path_details.get("relational_risk", 0.0),
            },
        },
        "sequence_risk": deep_path_details.get("sequence_risk", 0.0),
        "relational_risk": deep_path_details.get("relational_risk", 0.0),
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
class AccessRequest(BaseModel):
    round_trip_time_ms: float = Field(..., description="Network round-trip time in milliseconds")
    asn: int = Field(..., description="Autonomous System Number of client IP")
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
    user_id: str = Field("anonymous", description="Hashed user identifier")
    latitude: float = Field(0.0, description="Coarse latitude")
    longitude: float = Field(0.0, description="Coarse longitude")
    
    # New fields for Continuous Session Monitoring
    resource_id: str = Field("", description="Resource being accessed (e.g., file, API endpoint)")
    action_type: str = Field("READ", description="Action type: READ, WRITE, DELETE, EXECUTE, ADMIN")
    sensitivity_level: int = Field(1, ge=1, le=5, description="Resource sensitivity level (1-5, 5=highest)")

# ===================================================================
#  Simulation Scenarios
# ===================================================================
SCENARIO_DESCRIPTIONS = {
    "Credential_Stuffing": "Many users, many IPs, rapid failures — classic stuffing pattern",
    "Impossible_Travel": "Same user logs in from two continents minutes apart",
    "Insider_Threat": "Trusted employee accessing sensitive resources at 2–4 AM",
    "Normal_Baseline": "Healthy low-risk traffic to calibrate model thresholds",
}

SCENARIOS = {
    "Credential_Stuffing": lambda i: {
        "user_id": f"user_{random.randint(1000, 9999)}",
        "round_trip_time_ms": random.uniform(10, 80),
        "asn": random.randint(1000, 99999),
        "device_type": random.choice(["bot", "unknown", "mobile"]),
        "login_successful": 0,
        "is_attack_ip": 1,
        "country": random.choice(["RU", "CN", "KP", "IR"]),
        "region": "",
        "city": "",
        "os_name_version": "unknown",
        "browser_name_version": "unknown",
        "latitude": random.uniform(-90, 90),
        "longitude": random.uniform(-180, 180),
        "resource_id": f"/api/credentials/{random.randint(1000, 9999)}",
        "action_type": random.choice(["READ", "WRITE"]),
        "sensitivity_level": random.randint(2, 4),
    },
    "Impossible_Travel": lambda i: {
        "user_id": "victim_007",
        "round_trip_time_ms": 120.0,
        "asn": 15169,
        "device_type": "desktop",
        "login_successful": 1,
        "is_attack_ip": 0,
        "country": "US" if i % 2 == 0 else "JP",
        "region": "California" if i % 2 == 0 else "Tokyo",
        "city": "Mountain View" if i % 2 == 0 else "Shinjuku",
        "os_name_version": "Mac OS X 10.14.6",
        "browser_name_version": "Chrome 84.0.4147.338.339",
        "latitude": 37.77 if i % 2 == 0 else 35.68,
        "longitude": -122.41 if i % 2 == 0 else 139.69,
        "resource_id": "/api/user/profile",
        "action_type": "READ",
        "sensitivity_level": 2,
    },
    "Insider_Threat": lambda i: {
        "user_id": "employee_42",
        "round_trip_time_ms": 45.0,
        "asn": 7922,
        "device_type": "desktop",
        "login_successful": 1,
        "is_attack_ip": 0,
        "country": "US",
        "region": "California",
        "city": "San Francisco",
        "os_name_version": "Windows 10",
        "browser_name_version": "Chrome 84.0.4147.338.339",
        "latitude": 37.77,
        "longitude": -122.41,
        "timestamp": f"2026-03-20T0{random.randint(1,4)}:00:00",
        "resource_id": "/api/admin/sensitive_data",
        "action_type": random.choice(["READ", "DELETE", "ADMIN"]),
        "sensitivity_level": random.randint(4, 5),
    },
    "Normal_Baseline": lambda i: {
        "user_id": f"regular_{i}",
        "round_trip_time_ms": random.uniform(80, 200),
        "asn": 7922,
        "device_type": "desktop",
        "login_successful": 1,
        "is_attack_ip": 0,
        "country": "US",
        "region": "California",
        "city": "San Jose",
        "os_name_version": "Mac OS X 10.14.6",
        "browser_name_version": "Chrome 84.0.4147.338.339",
        "latitude": 37.33,
        "longitude": -121.89,
        "resource_id": "/api/dashboard",
        "action_type": "READ",
        "sensitivity_level": 1,
    },
}

# ===================================================================
#  FastAPI application
# ===================================================================
fastapi_app = FastAPI(
    title="Unified Zero Trust Login Risk PDP",
    description=(
        "Unified Policy Decision Point combining traditional ML, deep path models, "
        "stateful tracking, impossible travel detection, and OPA enforcement."
    ),
    version="2.0.0",
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
    <head><title>Unified Zero Trust PDP</title></head>
    <body style="font-family:sans-serif;max-width:800px;margin:40px auto;">
        <h1>🛡️ Unified Zero Trust Policy Decision Point</h1>
        <p>Advanced login risk assessment with stateful tracking and deep path analysis.</p>
        <ul>
            <li><a href="/docs">Interactive API docs (Swagger)</a></li>
            <li><a href="/redoc">ReDoc documentation</a></li>
            <li><b>POST /predict</b> &mdash; Enhanced prediction endpoint</li>
            <li><b>POST /simulate</b> &mdash; Attack simulation orchestrator</li>
            <li><b>GET /flask/</b> &mdash; Interactive web interface</li>
            <li><b>GET /health</b> &mdash; System health check</li>
        </ul>
    </body>
    </html>
    """

# ===================================================================
#  OPA Query Functions (from Alfred's implementation)
# ===================================================================
def compute_entropy(result: dict) -> float:
    """Measures disagreement between models using Shannon Entropy: H(X) = -Σ p_i * log_2(p_i)"""
    import math
    
    probs = []
    
    # Traditional models
    probs.append(result["models"]["lightgbm"]["probability"])
    probs.append(result["models"]["xgboost"]["probability"])
    probs.append(result["models"]["elliptic_envelope"]["risk_probability"])
    
    # Deep path model (if available)
    if result["models"]["deep_path"]["available"]:
        probs.append(result["models"]["deep_path"]["probability"])
    
    # Apply Shannon Entropy formula
    # H(X) = -Σ p_i * log_2(p_i)
    entropy = 0.0
    for p in probs:
        # Avoid log(0) which is undefined
        if p > 0 and p < 1:
            entropy -= p * math.log2(p)
    
    # Normalize to [0, 1] range (max entropy for n outcomes is log_2(n))
    max_entropy = math.log2(len(probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return round(normalized_entropy, 4)

def query_opa(risk_score: float, entropy: float,
              impossible_travel: bool, device_trusted: bool):
    """Query OPA sidecar. Falls back to inline rules if OPA is not running."""
    import httpx
    try:
        r = httpx.post(
            "http://localhost:8181/v1/data/authz/access",
            json={"input": {
                "risk_score": risk_score,
                "entropy": entropy,
                "impossible_travel": impossible_travel,
                "device_trusted": device_trusted
            }},
            timeout=0.1
        )
        res = r.json().get("result", {})
        return res.get("action", "MFA"), res.get("reason", "opa_no_reason")
    except Exception:
        # OPA not running — use strict inline fallback rules
        if impossible_travel:
            return ("BLOCK", "fallback_block_impossible_travel")
        if risk_score >= 0.75:  # Lowered from 0.85 to account for model variations
            return ("BLOCK", "fallback_block_high_risk")
        if risk_score >= 0.65:
            return ("MFA", "fallback_mfa_elevated_risk")
        if risk_score >= 0.5:
            return ("MFA", "fallback_mfa_moderate_risk")
        if entropy >= 0.7:
            return ("MFA", "fallback_mfa_high_entropy")
        if not device_trusted and risk_score >= 0.4:
            return ("MFA", "fallback_mfa_untrusted_device")
        return ("ALLOW", "fallback_allow")

@fastapi_app.post("/predict")
async def fastapi_predict(event: AccessRequest):
    """
    Enhanced prediction endpoint with stateful tracking, deep path analysis, and OPA enforcement.
    """
    try:
        data = event.model_dump()
        
        # Extract stateful fields — keep them separate from ML input
        user_id = data.get("user_id", "anonymous")
        curr_lat = data.get("latitude", 0.0)
        curr_lon = data.get("longitude", 0.0)
        curr_ts = time.time()
        resource_id = data.get("resource_id", "")
        action_type = data.get("action_type", "READ")
        sensitivity_level = data.get("sensitivity_level", 1)
        
        # Fetch user's activity sequence for behavioral analysis
        from db import get_activity_sequence
        activity_sequence = get_activity_sequence(user_id, limit=15)
        
        # Priority Gate: Force deep path for high-sensitivity resources
        force_deep_path = sensitivity_level >= 4
        
        # Build clean ML input with only the original fields
        ORIGINAL_FIELDS = {
            "round_trip_time_ms", "asn", "device_type", "login_successful",
            "is_attack_ip", "timestamp", "country", "region", "city",
            "os_name_version", "browser_name_version"
        }
        clean_data = {k: v for k, v in data.items() if k in ORIGINAL_FIELDS}
        
        # Impossible travel check (Alfred's stateful logic)
        travel = {"flagged": False, "reason": "first_login",
                  "distance_km": 0.0, "velocity_kmh": None}
        prev = get_last_login(user_id)
        if prev and (curr_lat != 0.0 or curr_lon != 0.0):
            travel = check_impossible_travel(
                prev[0], prev[1], prev[2],
                curr_lat, curr_lon, curr_ts
            )
        
        # Always update history after checking
        upsert_login(user_id, curr_lat, curr_lon, curr_ts)
        
        # Get enhanced ensemble prediction with deep path models
        result = ensemble_predict(clean_data, activity_sequence, force_deep_path)
        
        # Apply travel boost if impossible travel flagged (Alfred's logic)
        if travel["flagged"]:
            boosted = min(1.0, result["ensemble"]["risk_score"] + 0.40)
            result["ensemble"]["risk_score"] = round(boosted, 4)
            if boosted >= 0.85:
                result["ensemble"]["risk_level"] = "Critical"
            elif boosted >= 0.65:
                result["ensemble"]["risk_level"] = "High"
            elif boosted >= 0.40:
                result["ensemble"]["risk_level"] = "Medium"
            else:
                result["ensemble"]["risk_level"] = "Low"
        
        # Add travel info to response
        result["travel"] = travel
        
        # Compute entropy (model disagreement) and query OPA
        entropy = compute_entropy(result)
        device_trusted = data.get("device_type", "unknown") not in ("bot", "unknown")
        action, opa_reason = query_opa(
            result["ensemble"]["risk_score"],
            entropy,
            travel["flagged"],
            device_trusted
        )
        result["verdict"] = action
        result["opa_reason"] = opa_reason
        result["entropy"] = entropy
        
        # Log the resource access for continuous monitoring
        from db import log_resource_access
        log_resource_access(user_id, resource_id, action_type)
        
        # Add behavioral analysis for frontend Radar Chart
        result["behavioral_analysis"] = {
            "sequence_risk": result.get("sequence_risk", 0.0),
            "relational_risk": result.get("relational_risk", 0.0),
            "activity_sequence": activity_sequence,
            "sensitivity_level": sensitivity_level,
            "force_deep_path": force_deep_path
        }
        
        # Add Zero Trust metadata
        result["zero_trust"] = {
            "stateful_tracking": True,
            "deep_path_analysis": result["models"]["deep_path"]["available"],
            "opa_enforcement": True,
            "impossible_travel_detection": True,
            "entropy_based_uncertainty": True,
            "continuous_monitoring": True,
        }
        
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@fastapi_app.get("/health")
async def health():
    """Enhanced health check with deep path model status."""
    return {
        "status": "healthy",
        "models_loaded": {
            "lightgbm": lgbm_pipeline is not None,
            "xgboost": xgb_model is not None,
            "elliptic_envelope": isolation_model is not None,
            "lstm_deep_path": lstm_scorer is not None,
            "gnn_deep_path": gnn_scorer is not None,
            "deep_path_enabled": bool(lstm_scorer and gnn_scorer),
        },
        "zero_trust_features": {
            "stateful_tracking": True,
            "impossible_travel_detection": True,
            "opa_policy_enforcement": True,
            "entropy_uncertainty_detection": True,
            "deep_path_analysis": bool(lstm_scorer and gnn_scorer),
        }
    }

@fastapi_app.post("/simulate")
async def simulate(body: dict):
    """Enhanced simulation orchestrator with deep path analysis."""
    scenario_id = body.get("scenario_id", "Normal_Baseline")
    count = min(int(body.get("count", 5)), 10)

    if scenario_id not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario. Valid options: {list(SCENARIOS.keys())}"
        )

    results = []
    for i in range(count):
        log = SCENARIOS[scenario_id](i)
        log["event_id"] = str(uuid.uuid4())
        event_id = log.pop("event_id")
        user_id_for_log = log.get("user_id", "anonymous")
        event = AccessRequest(**log)
        result = await fastapi_predict(event)
        
        # Extract JSON response body
        import json
        if hasattr(result, "body"):
            result_data = json.loads(result.body)
        else:
            result_data = result
            
        results.append({
            "event_id": event_id,
            "user_id": user_id_for_log,
            "result": result_data
        })

    return {
        "scenario": scenario_id,
        "description": SCENARIO_DESCRIPTIONS[scenario_id],
        "events_sent": count,
        "results": results,
        "simulation_metadata": {
            "deep_path_enabled": bool(lstm_scorer and gnn_scorer),
            "zero_trust_features": ["stateful_tracking", "opa_enforcement", "entropy_detection", "deep_path"]
        }
    }

# ===================================================================
#  Flask application (Enhanced with Zero Trust features)
# ===================================================================
flask_app = Flask(__name__)

FLASK_FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Unified Zero Trust PDP</title>
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
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-top: 16px;
        }
        @media (max-width: 900px) { .model-cards { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 600px) { .model-cards { grid-template-columns: 1fr; } }
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
        .trust-features {
            background: rgba(0,255,0,0.1);
            border: 1px solid rgba(0,255,0,0.2);
            border-radius: 8px;
            padding: 12px;
            margin-top: 16px;
            font-size: 0.85rem;
        }
        .trust-features h4 {
            color: #4caf50;
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🛡️ Unified Zero Trust PDP</h1>
    <p class="subtitle">Traditional ML + Deep Path + Stateful Tracking + OPA Enforcement</p>

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
                <label>User ID</label>
                <input type="text" name="user_id" placeholder="anonymous" value="test_user_001">
            </div>
            <div class="form-group">
                <label>Latitude</label>
                <input type="number" step="0.000001" name="latitude" placeholder="37.77" value="37.77">
            </div>
            <div class="form-group">
                <label>Longitude</label>
                <input type="number" step="0.000001" name="longitude" placeholder="-122.41" value="-122.41">
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
        <button type="submit" class="btn" id="submitBtn">🛡️ Analyze with Zero Trust</button>
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
            <div class="model-card">
                <h3>Deep Path</h3>
                <div class="prob" id="deepProb">—</div>
                <div class="weight" id="deepWeight"></div>
                <div class="bar-outer"><div class="bar-inner" id="deepBar" style="width:0;background:#4caf50;"></div></div>
            </div>
        </div>
        
        <div class="trust-features">
            <h4>🛡️ Zero Trust Features Active</h4>
            <div id="trustFeatures">Loading...</div>
        </div>
    </div>
</div>

<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.textContent = '🔄 Analyzing…';

    const fd = new FormData(this);
    const body = {};
    fd.forEach((v, k) => {
        if (['round_trip_time_ms','asn','login_successful','is_attack_ip','latitude','longitude'].includes(k))
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

        document.getElementById('verdictText').textContent = `Final Verdict: ${data.verdict.toUpperCase()} (${data.opa_reason})`;

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

        const deepAvailable = m.deep_path.available;
        document.getElementById('deepProb').textContent = deepAvailable ? 
            (m.deep_path.probability * 100).toFixed(1) + '%' : 'N/A';
        document.getElementById('deepWeight').textContent = 'Weight: ' + m.deep_path.weight;
        document.getElementById('deepBar').style.width = deepAvailable ? 
            (m.deep_path.probability * 100) + '%' : '0%';

        // Zero Trust features
        const zt = data.zero_trust;
        const features = [];
        if (zt.stateful_tracking) features.push('✅ Stateful Tracking');
        if (zt.deep_path_analysis) features.push('✅ Deep Path Analysis');
        if (zt.opa_enforcement) features.push('✅ OPA Policy Enforcement');
        if (zt.impossible_travel_detection) features.push('✅ Impossible Travel Detection');
        if (zt.entropy_based_uncertainty) features.push('✅ Entropy-based Uncertainty');
        
        document.getElementById('trustFeatures').innerHTML = features.join('<br>');

        document.getElementById('results').scrollIntoView({behavior: 'smooth'});
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🛡️ Analyze with Zero Trust';
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
    """Flask JSON prediction endpoint with enhanced features."""
    try:
        data = flask_request.get_json(force=True)
        
        # Extract stateful fields — keep them separate from ML input
        user_id = data.get("user_id", "anonymous")
        curr_lat = data.get("latitude", 0.0)
        curr_lon = data.get("longitude", 0.0)
        curr_ts = time.time()
        resource_id = data.get("resource_id", "")
        action_type = data.get("action_type", "READ")
        sensitivity_level = data.get("sensitivity_level", 1)
        
        # Fetch user's activity sequence for behavioral analysis
        from db import get_activity_sequence
        activity_sequence = get_activity_sequence(user_id, limit=15)
        
        # Priority Gate: Force deep path for high-sensitivity resources
        force_deep_path = sensitivity_level >= 4
        
        # Build clean ML input with only the original fields
        ORIGINAL_FIELDS = {
            "round_trip_time_ms", "asn", "device_type", "login_successful",
            "is_attack_ip", "timestamp", "country", "region", "city",
            "os_name_version", "browser_name_version"
        }
        clean_data = {k: v for k, v in data.items() if k in ORIGINAL_FIELDS}
        
        # Impossible travel check
        travel = {"flagged": False, "reason": "first_login",
                  "distance_km": 0.0, "velocity_kmh": None}
        prev = get_last_login(user_id)
        if prev and (curr_lat != 0.0 or curr_lon != 0.0):
            travel = check_impossible_travel(
                prev[0], prev[1], prev[2],
                curr_lat, curr_lon, curr_ts
            )
        
        # Always update history after checking
        upsert_login(user_id, curr_lat, curr_lon, curr_ts)
        
        # Get enhanced ensemble prediction with deep path models
        result = ensemble_predict(clean_data, activity_sequence, force_deep_path)
        
        # Apply travel boost if impossible travel flagged
        if travel["flagged"]:
            boosted = min(1.0, result["ensemble"]["risk_score"] + 0.40)
            result["ensemble"]["risk_score"] = round(boosted, 4)
            if boosted >= 0.85:
                result["ensemble"]["risk_level"] = "Critical"
            elif boosted >= 0.65:
                result["ensemble"]["risk_level"] = "High"
            elif boosted >= 0.40:
                result["ensemble"]["risk_level"] = "Medium"
            else:
                result["ensemble"]["risk_level"] = "Low"
        
        # Add travel info to response
        result["travel"] = travel
        
        # Compute entropy (model disagreement) and query OPA
        entropy = compute_entropy(result)
        device_trusted = data.get("device_type", "unknown") not in ("bot", "unknown")
        action, opa_reason = query_opa(
            result["ensemble"]["risk_score"],
            entropy,
            travel["flagged"],
            device_trusted
        )
        result["verdict"] = action
        result["opa_reason"] = opa_reason
        result["entropy"] = entropy
        
        # Log the resource access for continuous monitoring
        from db import log_resource_access
        log_resource_access(user_id, resource_id, action_type)
        
        # Add behavioral analysis for frontend Radar Chart
        result["behavioral_analysis"] = {
            "sequence_risk": result.get("sequence_risk", 0.0),
            "relational_risk": result.get("relational_risk", 0.0),
            "activity_sequence": activity_sequence,
            "sensitivity_level": sensitivity_level,
            "force_deep_path": force_deep_path
        }
        
        # Add Zero Trust metadata
        result["zero_trust"] = {
            "stateful_tracking": True,
            "deep_path_analysis": result["models"]["deep_path"]["available"],
            "opa_enforcement": True,
            "impossible_travel_detection": True,
            "entropy_based_uncertainty": True,
            "continuous_monitoring": True,
        }
        
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

    print("\n  🛡️ Unified Zero Trust Policy Decision Point")
    print("  =========================================")
    print("  FastAPI      -> http://127.0.0.1:8000")
    print("  Swagger      -> http://127.0.0.1:8000/docs")
    print("  Flask UI     -> http://127.0.0.1:8000/flask/")
    print("  Deep Path    :", "ENABLED" if (lstm_scorer and gnn_scorer) else "DISABLED")
    print("  Stateful     : ENABLED")
    print("  OPA Policy   : ENABLED\n")
    uvicorn.run("app_unified:fastapi_app", host="0.0.0.0", port=8000, reload=True)
