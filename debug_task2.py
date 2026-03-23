import httpx
import json
import time

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

# High risk test
r1 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "device_type": "bot", "is_attack_ip": 1, "login_successful": 0,
    "user_id": "opa_test_1", "latitude": 0.0, "longitude": 0.0})
j1 = r1.json()
print("High-risk response:")
print(json.dumps(j1, indent=2))
