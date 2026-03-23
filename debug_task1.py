#!/usr/bin/env python
"""Debug TASK 1 failure"""
import httpx
import time

BASE = "http://127.0.0.1:8000"

COMMON = {
    "round_trip_time_ms": 120.0, "asn": 15169, "device_type": "desktop",
    "login_successful": 1, "is_attack_ip": 0, "country": "US",
    "region": "California", "city": "Mountain View",
    "os_name_version": "Mac OS X 10.14.6",
    "browser_name_version": "Chrome 84.0.4147.338.339"
}

# Use a unique user ID for this test
user_id = f"debug_user_{int(time.time())}"

# First login
r1 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "user_id": user_id, "latitude": 37.77, "longitude": -122.41})
print(f"First login response status: {r1.status_code}")
j1 = r1.json()
print(f"First login score: {j1['ensemble']['risk_score']}")
score1 = j1['ensemble']['risk_score']

time.sleep(1)

# Second login from Tokyo
r2 = httpx.post(f"{BASE}/predict", json={**COMMON,
    "user_id": user_id, "latitude": 35.68, "longitude": 139.69})
print(f"Second login response status: {r2.status_code}")
j2 = r2.json()
print(f"Second login score: {j2['ensemble']['risk_score']}")
print(f"Travel flagged: {j2['travel']['flagged']}")
print(f"Travel reason: {j2['travel']['reason']}")
print(f"Distance KM: {j2['travel']['distance_km']}")
print(f"Velocity KMH: {j2['travel']['velocity_kmh']}")
print(f"Score boost: {j2['ensemble']['risk_score'] - score1:.4f}")
print(f"Expected boost >= 0.35: {j2['ensemble']['risk_score'] >= score1 + 0.35}")
