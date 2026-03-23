import httpx
import time
import uuid

BASE = "http://127.0.0.1:8000"
COMMON = {
    "round_trip_time_ms": 120.0, "asn": 15169, "device_type": "desktop",
    "login_successful": 1, "is_attack_ip": 0, "country": "US",
    "region": "California", "city": "Mountain View",
    "os_name_version": "Mac OS X 10.14.6",
    "browser_name_version": "Chrome 84.0.4147.338.339"
}

# Use unique user ID to avoid DB conflicts from previous runs
unique_user = f"test_user_travel_{uuid.uuid4().hex[:8]}"

# Wait for server
time.sleep(3)

# First login — San Francisco
r1 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "user_id": unique_user, "latitude": 37.77, "longitude": -122.41})
assert r1.status_code == 200, f"FAIL: first request failed {r1.text}"
score1 = r1.json()["ensemble"]["risk_score"]
print(f"✓ First login: score={score1}")

time.sleep(1)

# Second login — Tokyo (8800km away, 1 second later = impossible)
r2 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "user_id": unique_user, "latitude": 35.68, "longitude": 139.69})
assert r2.status_code == 200, f"FAIL: second request failed {r2.text}"
j2 = r2.json()
assert "travel" in j2, "FAIL: 'travel' key missing from response"
assert j2["travel"]["flagged"] == True, f"FAIL: expected flagged=True, got {j2['travel']}"
assert j2["travel"]["reason"] == "impossible_travel", f"FAIL: wrong reason {j2['travel']['reason']}"
assert j2["ensemble"]["risk_score"] >= score1 + 0.35, \
    f"FAIL: score not boosted enough. Before={score1}, After={j2['ensemble']['risk_score']}"
print(f"✓ Second login: travel flagged={j2['travel']['flagged']}, score boosted to {j2['ensemble']['risk_score']}")
print("TASK 1: PASS")
