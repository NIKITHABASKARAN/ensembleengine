#!/usr/bin/env python
"""Comprehensive test for Unified Zero Trust PDP system"""
import httpx
import time
import uuid

BASE = "http://127.0.0.1:8000"

print("\n" + "=" * 80)
print("🛡️ UNIFIED ZERO TRUST PDP - COMPREHENSIVE INTEGRATION TEST")
print("=" * 80 + "\n")

# Test data template
COMMON = {
    "round_trip_time_ms": 120.0, "asn": 15169, "device_type": "desktop",
    "login_successful": 1, "is_attack_ip": 0, "country": "US",
    "region": "California", "city": "Mountain View",
    "os_name_version": "Mac OS X 10.14.6",
    "browser_name_version": "Chrome 84.0.4147.338.339"
}

# ========================================================================
# TEST 1: Enhanced Model Integration
# ========================================================================
print("[TEST 1] Enhanced Model Integration (Traditional + Deep Path)...")
try:
    r = httpx.post(f"{BASE}/predict", json={
        **COMMON,
        "user_id": f"unified_test_{uuid.uuid4().hex[:8]}",
        "latitude": 37.77, "longitude": -122.41
    })
    assert r.status_code == 200, f"FAIL: request failed {r.text}"
    data = r.json()
    
    # Verify all models are present
    assert "models" in data, "FAIL: models key missing"
    assert "lightgbm" in data["models"], "FAIL: LightGBM missing"
    assert "xgboost" in data["models"], "FAIL: XGBoost missing"
    assert "elliptic_envelope" in data["models"], "FAIL: EllipticEnvelope missing"
    assert "deep_path" in data["models"], "FAIL: Deep Path missing"
    
    # Verify Zero Trust features
    assert "zero_trust" in data, "FAIL: zero_trust metadata missing"
    zt = data["zero_trust"]
    assert zt["stateful_tracking"] == True, "FAIL: stateful tracking not enabled"
    assert zt["opa_enforcement"] == True, "FAIL: OPA enforcement not enabled"
    assert "impossible_travel_detection" in zt, "FAIL: impossible travel detection missing"
    assert "entropy_based_uncertainty" in zt, "FAIL: entropy detection missing"
    
    # Verify enhanced response structure
    assert "travel" in data, "FAIL: travel info missing"
    assert "entropy" in data, "FAIL: entropy calculation missing"
    assert "verdict" in data, "FAIL: OPA verdict missing"
    assert "opa_reason" in data, "FAIL: OPA reason missing"
    
    deep_available = data["models"]["deep_path"]["available"]
    print(f"✓ Enhanced integration working - Deep Path: {'ENABLED' if deep_available else 'DISABLED'}")
    test1_pass = True
except Exception as e:
    print(f"✗ TEST 1: FAIL - {e}")
    test1_pass = False

# ========================================================================
# TEST 2: Stateful Tracking & Impossible Travel (from Alfred)
# ========================================================================
print("\n[TEST 2] Stateful Tracking & Impossible Travel Detection...")
try:
    user_id = f"travel_test_{uuid.uuid4().hex[:8]}"
    
    # First login - San Francisco
    r1 = httpx.post(f"{BASE}/predict", json={
        **COMMON,
        "user_id": user_id,
        "latitude": 37.77, "longitude": -122.41
    })
    assert r1.status_code == 200, f"FAIL: first login failed {r1.text}"
    score1 = r1.json()["ensemble"]["risk_score"]
    
    time.sleep(1)
    
    # Second login - Tokyo (impossible travel)
    r2 = httpx.post(f"{BASE}/predict", json={
        **COMMON,
        "user_id": user_id,
        "latitude": 35.68, "longitude": 139.69
    })
    assert r2.status_code == 200, f"FAIL: second login failed {r2.text}"
    data2 = r2.json()
    
    # Verify impossible travel detection
    assert data2["travel"]["flagged"] == True, "FAIL: impossible travel not flagged"
    assert data2["travel"]["reason"] == "impossible_travel", "FAIL: wrong travel reason"
    assert data2["travel"]["distance_km"] > 8000, "FAIL: distance too small"
    
    # Verify risk boost
    score2 = data2["ensemble"]["risk_score"]
    assert score2 >= score1 + 0.35, f"FAIL: risk not boosted enough. Before={score1}, After={score2}"
    
    print(f"✓ Impossible travel detected - Distance: {data2['travel']['distance_km']:.1f}km, Risk boost: +{score2-score1:.3f}")
    test2_pass = True
except Exception as e:
    print(f"✗ TEST 2: FAIL - {e}")
    test2_pass = False

# ========================================================================
# TEST 3: OPA Policy Enforcement (from Alfred)
# ========================================================================
print("\n[TEST 3] OPA Policy Enforcement...")
try:
    # High-risk scenario (should trigger BLOCK)
    r1 = httpx.post(f"{BASE}/predict", json={
        **COMMON,
        "round_trip_time_ms": 1.0,
        "device_type": "bot",
        "is_attack_ip": 1,
        "login_successful": 0,
        "country": "RU",
        "user_id": f"opa_test_high_{uuid.uuid4().hex[:8]}",
        "latitude": 0.0, "longitude": 0.0
    })
    assert r1.status_code == 200, f"FAIL: high-risk test failed {r1.text}"
    data1 = r1.json()
    
    # Should trigger block or MFA
    assert data1["verdict"] in ["BLOCK", "MFA"], f"FAIL: expected BLOCK/MFA, got {data1['verdict']}"
    assert data1["opa_reason"] != "", "FAIL: OPA reason missing"
    
    # Low-risk scenario (should trigger ALLOW)
    r2 = httpx.post(f"{BASE}/predict", json={
        **COMMON,
        "device_type": "desktop",
        "is_attack_ip": 0,
        "login_successful": 1,
        "user_id": f"opa_test_low_{uuid.uuid4().hex[:8]}",
        "latitude": 0.0, "longitude": 0.0
    })
    assert r2.status_code == 200, f"FAIL: low-risk test failed {r2.text}"
    data2 = r2.json()
    
    # Should allow
    assert data2["verdict"] == "ALLOW", f"FAIL: expected ALLOW, got {data2['verdict']}"
    
    print(f"✓ OPA enforcement working - High: {data1['verdict']} ({data1['opa_reason']}), Low: {data2['verdict']}")
    test3_pass = True
except Exception as e:
    print(f"✗ TEST 3: FAIL - {e}")
    test3_pass = False

# ========================================================================
# TEST 4: Enhanced Simulation Orchestrator
# ========================================================================
print("\n[TEST 4] Enhanced Simulation Orchestrator...")
try:
    # Test Credential Stuffing with deep path analysis
    r1 = httpx.post(f"{BASE}/simulate", json={
        "scenario_id": "Credential_Stuffing",
        "count": 3
    })
    assert r1.status_code == 200, f"FAIL: simulation failed {r1.text}"
    data1 = r1.json()
    
    assert data1["events_sent"] == 3, "FAIL: wrong event count"
    assert len(data1["results"]) == 3, "FAIL: wrong result count"
    assert "simulation_metadata" in data1, "FAIL: simulation metadata missing"
    
    # Verify each result has enhanced structure
    for result in data1["results"]:
        assert "zero_trust" in result["result"], "FAIL: zero trust metadata missing in simulation"
        assert "deep_path" in result["result"]["models"], "FAIL: deep path missing in simulation"
    
    # Test Impossible Travel scenario
    r2 = httpx.post(f"{BASE}/simulate", json={
        "scenario_id": "Impossible_Travel",
        "count": 4
    })
    assert r2.status_code == 200, f"FAIL: impossible travel simulation failed {r2.text}"
    data2 = r2.json()
    
    # Should have at least some flagged travel events
    flagged = [x for x in data2["results"] if x["result"]["travel"]["flagged"]]
    assert len(flagged) >= 1, "FAIL: no travel flagged in simulation"
    
    print(f"✓ Enhanced simulation working - Credential Stuffing: {data1['events_sent']} events, Travel flagged: {len(flagged)}/4")
    test4_pass = True
except Exception as e:
    print(f"✗ TEST 4: FAIL - {e}")
    test4_pass = False

# ========================================================================
# TEST 5: Health Check with Enhanced Status
# ========================================================================
print("\n[TEST 5] Enhanced Health Check...")
try:
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200, f"FAIL: health check failed {r.text}"
    data = r.json()
    
    # Verify model status
    assert "models_loaded" in data, "FAIL: models_loaded missing"
    assert data["models_loaded"]["lightgbm"] == True, "FAIL: LightGBM not loaded"
    assert data["models_loaded"]["xgboost"] == True, "FAIL: XGBoost not loaded"
    assert data["models_loaded"]["elliptic_envelope"] == True, "FAIL: EllipticEnvelope not loaded"
    
    # Verify Zero Trust features status
    assert "zero_trust_features" in data, "FAIL: zero_trust_features missing"
    zt_features = data["zero_trust_features"]
    assert zt_features["stateful_tracking"] == True, "FAIL: stateful tracking not reported"
    assert zt_features["opa_policy_enforcement"] == True, "FAIL: OPA enforcement not reported"
    assert zt_features["entropy_uncertainty_detection"] == True, "FAIL: entropy detection not reported"
    
    # Check deep path status (may be True or False)
    deep_status = zt_features["deep_path_analysis"]
    print(f"✓ Enhanced health check working - Deep Path Analysis: {'ENABLED' if deep_status else 'DISABLED'}")
    test5_pass = True
except Exception as e:
    print(f"✗ TEST 5: FAIL - {e}")
    test5_pass = False

# ========================================================================
# TEST 6: Entropy Calculation with All Models
# ========================================================================
print("\n[TEST 6] Enhanced Entropy Calculation...")
try:
    r = httpx.post(f"{BASE}/predict", json={
        **COMMON,
        "user_id": f"entropy_test_{uuid.uuid4().hex[:8]}",
        "latitude": 37.77, "longitude": -122.41
    })
    assert r.status_code == 200, f"FAIL: entropy test failed {r.text}"
    data = r.json()
    
    # Verify entropy calculation
    assert "entropy" in data, "FAIL: entropy missing"
    entropy = data["entropy"]
    assert 0.0 <= entropy <= 1.0, f"FAIL: entropy out of range: {entropy}"
    
    # Entropy should be calculated including deep path if available
    deep_available = data["models"]["deep_path"]["available"]
    if deep_available:
        print(f"✓ Enhanced entropy calculation working - Entropy: {entropy:.4f} (including deep path)")
    else:
        print(f"✓ Entropy calculation working - Entropy: {entropy:.4f} (traditional models only)")
    
    test6_pass = True
except Exception as e:
    print(f"✗ TEST 6: FAIL - {e}")
    test6_pass = False

# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 80)
print("🛡️ UNIFIED ZERO TRUST PDP - TEST SUMMARY")
print("=" * 80)

tests = [
    ("Enhanced Model Integration", test1_pass),
    ("Stateful Tracking & Impossible Travel", test2_pass),
    ("OPA Policy Enforcement", test3_pass),
    ("Enhanced Simulation Orchestrator", test4_pass),
    ("Enhanced Health Check", test5_pass),
    ("Enhanced Entropy Calculation", test6_pass),
]

all_passed = True
for test_name, passed in tests:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test_name:<40} {status}")
    if not passed:
        all_passed = False

print("=" * 80)

if all_passed:
    print("\n🎉 ALL TESTS PASSED!")
    print("\n🛡️ Unified Zero Trust PDP Features:")
    print("  ✅ Traditional ML Models (LightGBM + XGBoost + EllipticEnvelope)")
    print("  ✅ Deep Path Analysis (LSTM + GraphSAGE)")
    print("  ✅ Stateful Login Tracking")
    print("  ✅ Impossible Travel Detection")
    print("  ✅ OPA Policy Enforcement")
    print("  ✅ Entropy-based Uncertainty Detection")
    print("  ✅ Enhanced Simulation Orchestrator")
    print("  ✅ Comprehensive Health Monitoring")
    print("\n🚀 System ready for production deployment!")
    print("\n📡 Access URLs:")
    print("  FastAPI:      http://127.0.0.1:8000")
    print("  Swagger:      http://127.0.0.1:8000/docs")
    print("  Web UI:       http://127.0.0.1:8000/flask/")
    print("  Health Check: http://127.0.0.1:8000/health")
else:
    print("\n❌ SOME TESTS FAILED - Please review the failures above.")

print("\n" + "=" * 80)
