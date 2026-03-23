#!/usr/bin/env python3
"""
Test script for Continuous Session Monitoring features.

Tests the upgraded Luminaries system with:
1. Activity sequence tracking
2. Priority gate for high-sensitivity resources
3. LSTM sequence anomaly detection
4. GNN relational analysis
5. Behavioral analysis in response
"""

import requests
import json
import time
import random

BASE_URL = "http://127.0.0.1:8000"

def test_continuous_monitoring():
    """Test the complete continuous monitoring system."""
    
    print("🧪 Testing Continuous Session Monitoring System")
    print("=" * 60)
    
    # Test 1: Normal low-sensitivity access
    print("\n📋 Test 1: Normal Low-Sensitivity Access")
    test_request = {
        "user_id": "user_001",
        "round_trip_time_ms": 120.0,
        "asn": 15169,
        "device_type": "desktop",
        "login_successful": 1,
        "is_attack_ip": 0,
        "country": "US",
        "region": "California",
        "city": "San Francisco",
        "os_name_version": "Mac OS X 10.14.6",
        "browser_name_version": "Chrome 84.0.4147.338.339",
        "latitude": 37.77,
        "longitude": -122.41,
        "resource_id": "/api/dashboard",
        "action_type": "READ",
        "sensitivity_level": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_request)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Status: {result.get('verdict', 'Unknown')}")
        print(f"📊 Risk Score: {result.get('ensemble', {}).get('risk_score', 0):.4f}")
        print(f"🧠 Entropy: {result.get('entropy', 0):.4f}")
        
        behavioral = result.get('behavioral_analysis', {})
        print(f"🔍 Sequence Risk: {behavioral.get('sequence_risk', 0):.4f}")
        print(f"🌐 Relational Risk: {behavioral.get('relational_risk', 0):.4f}")
        print(f"🎯 Sensitivity Level: {behavioral.get('sensitivity_level', 0)}")
        print(f"⚡ Force Deep Path: {behavioral.get('force_deep_path', False)}")
    else:
        print(f"❌ Failed: {response.status_code}")
    
    # Test 2: High-sensitivity resource (should force deep path)
    print("\n📋 Test 2: High-Sensitivity Resource Access")
    high_sensitivity_request = test_request.copy()
    high_sensitivity_request.update({
        "resource_id": "/api/admin/sensitive_data",
        "action_type": "DELETE",
        "sensitivity_level": 5
    })
    
    response = requests.post(f"{BASE_URL}/predict", json=high_sensitivity_request)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Status: {result.get('verdict', 'Unknown')}")
        print(f"📊 Risk Score: {result.get('ensemble', {}).get('risk_score', 0):.4f}")
        
        behavioral = result.get('behavioral_analysis', {})
        print(f"🎯 Sensitivity Level: {behavioral.get('sensitivity_level', 0)}")
        print(f"⚡ Force Deep Path: {behavioral.get('force_deep_path', False)}")
        print(f"🔬 Deep Path Available: {result.get('models', {}).get('deep_path', {}).get('available', False)}")
        
        if behavioral.get('force_deep_path'):
            print("✅ Priority gate correctly triggered deep path analysis")
        else:
            print("❌ Priority gate failed to trigger deep path")
    else:
        print(f"❌ Failed: {response.status_code}")
    
    # Test 3: Build activity sequence and test anomaly detection
    print("\n📋 Test 3: Activity Sequence Building")
    user_id = "user_sequence_test"
    
    # Build a normal sequence
    normal_resources = [
        "/api/dashboard",
        "/api/profile",
        "/api/settings",
        "/api/reports",
        "/api/messages"
    ]
    
    for i, resource in enumerate(normal_resources):
        seq_request = {
            "user_id": user_id,
            "round_trip_time_ms": 100.0 + i * 10,
            "asn": 15169,
            "device_type": "desktop",
            "login_successful": 1,
            "is_attack_ip": 0,
            "country": "US",
            "region": "California",
            "city": "San Francisco",
            "os_name_version": "Mac OS X 10.14.6",
            "browser_name_version": "Chrome 84.0.4147.338.339",
            "latitude": 37.77,
            "longitude": -122.41,
            "resource_id": resource,
            "action_type": "READ",
            "sensitivity_level": 2
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=seq_request)
        if response.status_code == 200:
            result = response.json()
            behavioral = result.get('behavioral_analysis', {})
            sequence = behavioral.get('activity_sequence', [])
            print(f"📝 Activity {i+1}: {resource} | Sequence length: {len(sequence)}")
        else:
            print(f"❌ Failed to log activity {i+1}")
        
        time.sleep(0.1)  # Small delay between requests
    
    # Test anomaly with unusual resource
    print("\n📋 Test 4: Anomaly Detection")
    anomaly_request = seq_request.copy()
    anomaly_request.update({
        "resource_id": "/api/admin/delete_all_users",  # Unusual resource
        "action_type": "DELETE",
        "sensitivity_level": 5
    })
    
    response = requests.post(f"{BASE_URL}/predict", json=anomaly_request)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Status: {result.get('verdict', 'Unknown')}")
        print(f"📊 Risk Score: {result.get('ensemble', {}).get('risk_score', 0):.4f}")
        
        behavioral = result.get('behavioral_analysis', {})
        print(f"🔍 Sequence Risk: {behavioral.get('sequence_risk', 0):.4f}")
        print(f"🌐 Relational Risk: {behavioral.get('relational_risk', 0):.4f}")
        
        sequence = behavioral.get('activity_sequence', [])
        print(f"📝 Total Activities Logged: {len(sequence)}")
        
        # Check if sequence risk is elevated (indicating anomaly detection)
        if behavioral.get('sequence_risk', 0) > 0.3:
            print("✅ LSTM correctly detected sequence anomaly")
        else:
            print("⚠️ LSTM may not have detected strong sequence anomaly")
    else:
        print(f"❌ Failed: {response.status_code}")
    
    # Test 5: Verify database logging
    print("\n📋 Test 5: Database Activity Verification")
    try:
        import sqlite3
        import os
        
        db_path = "login_history.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM user_activity_log WHERE user_id = ?
            """, (user_id,))
            count = cursor.fetchone()[0]
            conn.close()
            
            print(f"📊 Database logged {count} activities for user {user_id}")
            
            if count >= len(normal_resources) + 1:  # Normal sequence + anomaly
                print("✅ Activity logging working correctly")
            else:
                print("⚠️ Some activities may not have been logged")
        else:
            print("❌ Database file not found")
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
    
    # Test 6: Health check with new features
    print("\n📋 Test 6: System Health Check")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print("✅ System Health: OK")
        
        zero_trust = health.get('zero_trust_features', {})
        print("🛡️ Zero Trust Features:")
        for feature, enabled in zero_trust.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {feature}")
        
        if zero_trust.get('deep_path_analysis'):
            print("✅ Deep path analysis is enabled")
        else:
            print("⚠️ Deep path analysis not available (models may be missing)")
    else:
        print(f"❌ Health check failed: {response.status_code}")
    
    print("\n🎉 Continuous Session Monitoring Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_continuous_monitoring()
