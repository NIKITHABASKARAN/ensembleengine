import httpx
import time
import uuid

BASE = "http://127.0.0.1:8000"
COMMON = {
    "round_trip_time_ms": 120.0, "asn": 15169,
    "login_successful": 1, "is_attack_ip": 0, "country": "US",
    "region": "California", "city": "Mountain View",
    "os_name_version": "Mac OS X 10.14.6",
    "browser_name_version": "Chrome 84.0.4147.338.339"
}

# Wait for server
time.sleep(3)

# High-risk test: very low RTT + attack IP + failed login + bot
r1 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "round_trip_time_ms": 1.0, "device_type": "bot", "is_attack_ip": 1, "login_successful": 0,
    "country": "RU", "city": "Moscow",
    "user_id": f"opa_test_high_{uuid.uuid4().hex[:8]}", "latitude": 0.0, "longitude": 0.0})
assert r1.status_code == 200, f"FAIL: {r1.text}"
j1 = r1.json()
print(f"High-risk test: verdict={j1['verdict']}, risk_score={j1['ensemble']['risk_score']}, entropy={j1['entropy']}")
assert "verdict" in j1, "FAIL: 'verdict' key missing"
assert "opa_reason" in j1, "FAIL: 'opa_reason' key missing"
assert "entropy" in j1, "FAIL: 'entropy' key missing"
assert 0.0 <= j1["entropy"] <= 1.0, f"FAIL: entropy out of range {j1['entropy']}"
# Risk verdict should be MFA or higher based on risk_score
if j1['ensemble']['risk_score'] >= 0.85:
    assert j1["verdict"] in ("BLOCK", "MFA"), f"FAIL: expected BLOCK/MFA for score >= 0.85, got {j1['verdict']}"
elif j1['ensemble']['risk_score'] >= 0.5:
    assert j1["verdict"] in ("BLOCK", "MFA"), f"FAIL: expected BLOCK/MFA for score >= 0.5, got {j1['verdict']}"
print(f"✓ High-risk test passed")

# Low risk — should ALLOW
r2 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "device_type": "desktop", "is_attack_ip": 0, "login_successful": 1,
    "user_id": f"opa_test_low_{uuid.uuid4().hex[:8]}", "latitude": 0.0, "longitude": 0.0})
assert r2.status_code == 200, f"FAIL: {r2.text}"
j2 = r2.json()
assert j2["verdict"] == "ALLOW", f"FAIL: expected ALLOW, got {j2['verdict']}"
print(f"✓ Low-risk test: verdict={j2['verdict']}")
print("TASK 2: PASS")
