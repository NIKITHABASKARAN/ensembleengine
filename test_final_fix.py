#!/usr/bin/env python
"""Final test to verify the high-risk scenario fix"""
import httpx
import uuid

BASE = "http://127.0.0.1:8000"

print("\n🔧 FINAL HIGH-RISK SCENARIO TEST")
print("=" * 50)

# Test the specific high-risk scenario from test_final.py
print("\n[TEST] High-Risk Scenario (RU + Bot + Attack IP + Failed Login)...")
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
        "user_id": f"final_test_high_{uuid.uuid4().hex[:8]}",
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
        
        # Check individual model contributions
        models = data["models"]
        print(f"\n📈 Model Contributions:")
        print(f"   LightGBM: {models['lightgbm']['probability']:.4f} (weight: {models['lightgbm']['weight']})")
        print(f"   XGBoost:  {models['xgboost']['probability']:.4f} (weight: {models['xgboost']['weight']})")
        print(f"   Isolation: {models['elliptic_envelope']['risk_probability']:.4f} (weight: {models['elliptic_envelope']['weight']})")
        print(f"   Deep Path: {models['deep_path']['probability']:.4f} (weight: {models['deep_path']['weight']})")
        
        # Success criteria
        success = True
        if score > 0.85:
            print(f"✅ SUCCESS: Score {score:.4f} > 0.85!")
        else:
            print(f"❌ ISSUE: Score {score:.4f} is not > 0.85")
            success = False
            
        if verdict == "BLOCK":
            print(f"✅ SUCCESS: BLOCK verdict achieved!")
        elif verdict == "MFA":
            print(f"⚠️  MFA verdict (acceptable for security)")
        else:
            print(f"❌ ISSUE: Expected BLOCK/MFA, got {verdict}")
            success = False
            
        # Check for deep path errors
        if "error" in str(models.get("deep_path", {})):
            print(f"❌ Deep Path Error: {models['deep_path']}")
            success = False
        else:
            print(f"✅ Deep Path working correctly")
            
        if success:
            print(f"\n🎉 HIGH-RISK SCENARIO TEST PASSED!")
        else:
            print(f"\n❌ HIGH-RISK SCENARIO TEST NEEDS ADJUSTMENT")
            
    else:
        print(f"❌ Request failed: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🔧 FINAL TEST COMPLETE")
print("=" * 50)
