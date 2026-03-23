#!/usr/bin/env python
"""Quick test to verify the fixes for unified app"""
import httpx
import uuid

BASE = "http://127.0.0.1:8000"

print("\n🔧 TESTING UNIFIED APP FIXES")
print("=" * 50)

# Test the high-risk scenario that should trigger BLOCK
print("\n[TEST] High-Risk Scenario (RU + Bot + Attack IP)...")
try:
    response = httpx.post(f"{BASE}/predict", json={
        "round_trip_time_ms": 1.0,
        "asn": 15169,
        "device_type": "bot",
        "login_successful": 0,
        "is_attack_ip": 1,
        "country": "RU",
        "region": "Moscow Oblast",
        "city": "Moscow",
        "os_name_version": "unknown",
        "browser_name_version": "unknown",
        "user_id": f"test_high_{uuid.uuid4().hex[:8]}",
        "latitude": 0.0,
        "longitude": 0.0
    }, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        score = data["ensemble"]["risk_score"]
        verdict = data["verdict"]
        reason = data.get("opa_reason", "N/A")
        
        print(f"✅ Request successful")
        print(f"📊 Risk Score: {score:.4f}")
        print(f"🛡️ Verdict: {verdict}")
        print(f"📝 Reason: {reason}")
        
        if score > 0.85:
            print("✅ SUCCESS: Score > 0.85 achieved!")
        else:
            print(f"❌ ISSUE: Score {score:.4f} is not > 0.85")
            
        if verdict == "BLOCK":
            print("✅ SUCCESS: BLOCK verdict achieved!")
        else:
            print(f"❌ ISSUE: Expected BLOCK, got {verdict}")
            
        # Check entropy calculation
        entropy = data.get("entropy", 0)
        if 0.0 <= entropy <= 1.0:
            print(f"✅ Entropy calculation working: {entropy:.4f}")
        else:
            print(f"❌ Entropy issue: {entropy}")
            
        # Check deep path
        deep_available = data["models"]["deep_path"]["available"]
        print(f"🧠 Deep Path: {'ENABLED' if deep_available else 'DISABLED'}")
        
    else:
        print(f"❌ Request failed: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🔧 FIXES VERIFICATION COMPLETE")
print("=" * 50)
