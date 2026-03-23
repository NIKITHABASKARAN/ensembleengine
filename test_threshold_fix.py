#!/usr/bin/env python
"""Test the updated thresholds for high-risk scenario"""
import httpx
import uuid

BASE = "http://127.0.0.1:8000"

print("\n🔧 THRESHOLD FIX TEST")
print("=" * 50)

# Test the high-risk scenario with new thresholds
print("\n[TEST] High-Risk Scenario with Updated Thresholds...")
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
        "user_id": f"threshold_test_{uuid.uuid4().hex[:8]}",
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
        
        # Calculate expected score with new weights
        expected_score = (
            models['lightgbm']['probability'] * models['lightgbm']['weight'] +
            models['xgboost']['probability'] * models['xgboost']['weight'] +
            models['elliptic_envelope']['risk_probability'] * models['elliptic_envelope']['weight'] +
            models['deep_path']['probability'] * models['deep_path']['weight']
        )
        print(f"\n🧮 Calculated Score: {expected_score:.4f}")
        
        # Success criteria
        success = True
        
        # Check if score meets threshold for BLOCK (>=0.75)
        if score >= 0.75:
            print(f"✅ SUCCESS: Score {score:.4f} >= 0.75 (BLOCK threshold)!")
        elif score >= 0.65:
            print(f"⚠️  SCORE: {score:.4f} >= 0.65 (MFA elevated threshold)")
        else:
            print(f"❌ ISSUE: Score {score:.4f} below all thresholds")
            success = False
            
        # Check verdict
        if verdict == "BLOCK":
            print(f"✅ SUCCESS: BLOCK verdict achieved!")
        elif verdict == "MFA":
            print(f"⚠️  MFA verdict (still good for security)")
        else:
            print(f"❌ ISSUE: Expected BLOCK/MFA, got {verdict}")
            success = False
            
        # Check if reason matches expected
        if score >= 0.75 and verdict == "BLOCK":
            if "high_risk" in reason:
                print(f"✅ SUCCESS: Correct reason for high-risk BLOCK")
            else:
                print(f"⚠️  Unexpected reason: {reason}")
        elif score >= 0.65 and verdict == "MFA":
            if "elevated_risk" in reason:
                print(f"✅ SUCCESS: Correct reason for elevated MFA")
            else:
                print(f"⚠️  Unexpected reason: {reason}")
                
        if success:
            print(f"\n🎉 THRESHOLD FIX TEST PASSED!")
            print(f"   High-risk scenario properly handled")
        else:
            print(f"\n❌ THRESHOLD FIX TEST NEEDS FURTHER ADJUSTMENT")
            
    else:
        print(f"❌ Request failed: {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🔧 THRESHOLD TEST COMPLETE")
print("=" * 50)
