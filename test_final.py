#!/usr/bin/env python
"""Full integration test for all TASK 1, 2, 3"""
import httpx
import time
import uuid

BASE = "http://127.0.0.1:8000"

print("\n" + "=" * 70)
print("RUNNING FULL INTEGRATION TEST")
print("=" * 70 + "\n")

# ========================================================================
# TASK 1: Impossible Travel Detection
# ========================================================================
print("[TASK 1] Testing Impossible Travel Detection...")
COMMON = {
    "round_trip_time_ms": 120.0, "asn": 15169, "device_type": "desktop",
    "login_successful": 1, "is_attack_ip": 0, "country": "US",
    "region": "California", "city": "Mountain View",
    "os_name_version": "Mac OS X 10.14.6",
    "browser_name_version": "Chrome 84.0.4147.338.339"
}

try:
    # Use unique user ID for this test to avoid DB state conflicts
    task1_user = f"final_task1_user_{uuid.uuid4().hex[:12]}"
    
    # First login — San Francisco
    r1 = httpx.post(f"{BASE}/predict", json={**COMMON,
        "user_id": task1_user, "latitude": 37.77, "longitude": -122.41})
    assert r1.status_code == 200, f"FAIL: first request failed {r1.text}"
    score1 = r1.json()["ensemble"]["risk_score"]
    
    time.sleep(1)
    
    # Second login — Tokyo (impossible in 1 second)
    r2 = httpx.post(f"{BASE}/predict", json={**COMMON,
        "user_id": task1_user, "latitude": 35.68, "longitude": 139.69})
    assert r2.status_code == 200, f"FAIL: second request failed {r2.text}"
    j2 = r2.json()
    assert "travel" in j2, "FAIL: 'travel' key missing"
    assert j2["travel"]["flagged"] == True, f"FAIL: travel not flagged"
    assert j2["travel"]["reason"] == "impossible_travel", f"FAIL: wrong reason"
    assert j2["ensemble"]["risk_score"] >= score1 + 0.35, "FAIL: score not boosted"
    
    print("✓ TASK 1: PASS - Impossible travel detection working")
    task1_pass = True
except Exception as e:
    print(f"✗ TASK 1: FAIL - {e}")
    task1_pass = False

# ========================================================================
# TASK 2: Policy-as-Code (OPA) with Fallback
# ========================================================================
print("\n[TASK 2] Testing Policy-as-Code & OPA Query...")
try:
    # High-risk test
    r1 = httpx.post(f"{BASE}/predict", json={**COMMON,
        "round_trip_time_ms": 1.0, "device_type": "bot", "is_attack_ip": 1, 
        "login_successful": 0, "country": "RU", "city": "Moscow",
        "user_id": f"final_task2_high_{uuid.uuid4().hex[:12]}", "latitude": 0.0, "longitude": 0.0})
    assert r1.status_code == 200, f"FAIL: {r1.text}"
    j1 = r1.json()
    assert "verdict" in j1, "FAIL: verdict missing"
    assert "opa_reason" in j1, "FAIL: opa_reason missing"
    assert "entropy" in j1, "FAIL: entropy missing"
    assert 0.0 <= j1["entropy"] <= 1.0, f"FAIL: entropy out of range {j1['entropy']}"
    
    # Low-risk test
    r2 = httpx.post(f"{BASE}/predict", json={**COMMON,
        "device_type": "desktop", "is_attack_ip": 0, "login_successful": 1,
        "user_id": f"final_task2_low_{uuid.uuid4().hex[:12]}", "latitude": 0.0, "longitude": 0.0})
    assert r2.status_code == 200, f"FAIL: {r2.text}"
    j2 = r2.json()
    assert j2["verdict"] == "ALLOW", f"FAIL: expected ALLOW for low-risk"
    
    print("✓ TASK 2: PASS - OPA policy & entropy computation working")
    task2_pass = True
except Exception as e:
    print(f"✗ TASK 2: FAIL - {e}")
    task2_pass = False

# ========================================================================
# TASK 3: Simulation Log Orchestrator
# ========================================================================
print("\n[TASK 3] Testing Simulation Scenarios...")
try:
    # Credential Stuffing
    r1 = httpx.post(f"{BASE}/simulate",
        json={"scenario_id": "Credential_Stuffing", "count": 3}, timeout=30)
    assert r1.status_code == 200, f"FAIL: {r1.text}"
    j1 = r1.json()
    assert j1["events_sent"] == 3, f"FAIL: wrong event count"
    assert len(j1["results"]) == 3, f"FAIL: wrong result count"
    for item in j1["results"]:
        assert "event_id" in item and "user_id" in item and "result" in item
    
    # Impossible Travel scenario
    r2 = httpx.post(f"{BASE}/simulate",
        json={"scenario_id": "Impossible_Travel", "count": 4}, timeout=30)
    assert r2.status_code == 200, f"FAIL: {r2.text}"
    j2 = r2.json()
    flagged = [x for x in j2["results"] if x["result"]["travel"]["flagged"]]
    assert len(flagged) >= 1, f"FAIL: no travel flagged"
    
    # Invalid scenario (should be 400)
    r3 = httpx.post(f"{BASE}/simulate",
        json={"scenario_id": "INVALID", "count": 1}, timeout=10)
    assert r3.status_code == 400, f"FAIL: expected 400 for invalid scenario"
    
    # Count capping
    r4 = httpx.post(f"{BASE}/simulate",
        json={"scenario_id": "Normal_Baseline", "count": 20}, timeout=60)
    assert r4.status_code == 200, f"FAIL: {r4.text}"
    assert r4.json()["events_sent"] == 10, f"FAIL: count not capped at 10"
    
    print("✓ TASK 3: PASS - Simulation scenarios working")
    task3_pass = True
except Exception as e:
    print(f"✗ TASK 3: FAIL - {e}")
    task3_pass = False

# ========================================================================
# Summary
# ========================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"TASK 1 (State Management + Impossible Travel):  {'PASS' if task1_pass else 'FAIL'}")
print(f"TASK 2 (Policy-as-Code + OPA):               {'PASS' if task2_pass else 'FAIL'}")
print(f"TASK 3 (Simulation Log Orchestrator):         {'PASS' if task3_pass else 'FAIL'}")
print("=" * 70)

if task1_pass and task2_pass and task3_pass:
    print("\n✅ All 3 tasks complete.")
    print("\nFiles created/modified:")
    print("  - app.py (modified)")
    print("  - db.py (new)")
    print("  - haversine.py (new)")
    print("  - access.rego (new)")
    print("  - requirements.txt (modified)")
    print("\nServer running at http://127.0.0.1:8000")
    print("Endpoints: POST /predict | POST /simulate | GET /health | GET /docs\n")
else:
    print("\n❌ Some tasks failed. Please review above.\n")
