import httpx
import time

BASE = "http://127.0.0.1:8000"

# Wait for server
time.sleep(3)

# Test 1: Credential Stuffing — 5 events
r1 = httpx.post(f"{BASE}/simulate",
    json={"scenario_id": "Credential_Stuffing", "count": 5}, timeout=30)
assert r1.status_code == 200, f"FAIL T3-1: {r1.text}"
j1 = r1.json()
assert j1["events_sent"] == 5, f"FAIL T3-1: events_sent={j1['events_sent']}"
assert len(j1["results"]) == 5, f"FAIL T3-1: results len={len(j1['results'])}"
for item in j1["results"]:
    assert "event_id" in item and "user_id" in item and "result" in item
    assert "ensemble" in item["result"], f"FAIL T3-1: no ensemble in result"
    assert "verdict"  in item["result"], f"FAIL T3-1: no verdict in result"
    assert "travel"   in item["result"], f"FAIL T3-1: no travel in result"
print("✓ TASK 3 Test 1 (Credential_Stuffing): PASS")

# Test 2: Impossible Travel — at least 1 flagged
r2 = httpx.post(f"{BASE}/simulate",
    json={"scenario_id": "Impossible_Travel", "count": 4}, timeout=30)
assert r2.status_code == 200, f"FAIL T3-2: {r2.text}"
j2 = r2.json()
flagged = [x for x in j2["results"] if x["result"]["travel"]["flagged"]]
assert len(flagged) >= 1, f"FAIL T3-2: no impossible travel flagged"
print("✓ TASK 3 Test 2 (Impossible_Travel): PASS")

# Test 3: Invalid scenario — expect 400
r3 = httpx.post(f"{BASE}/simulate",
    json={"scenario_id": "INVALID_SCENARIO", "count": 1}, timeout=10)
assert r3.status_code == 400, f"FAIL T3-3: expected 400, got {r3.status_code}"
print("✓ TASK 3 Test 3 (invalid scenario): PASS")

# Test 4: Count cap at 10
r4 = httpx.post(f"{BASE}/simulate",
    json={"scenario_id": "Normal_Baseline", "count": 10}, timeout=60)
assert r4.status_code == 200, f"FAIL T3-4: {r4.text}"
assert r4.json()["events_sent"] == 10, f"FAIL T3-4: count not 10"
print("✓ TASK 3 Test 4 (Normal_Baseline count=10): PASS")

print("TASK 3: PASS")
