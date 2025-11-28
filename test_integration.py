#!/usr/bin/env python3
"""
Integration Test Script for Fall Detection System
Tests the connection between Python fall detection and Node.js backend
"""

import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:4000')
FALL_ALERT_TOKEN = os.getenv('FALL_ALERT_TOKEN', 'secret_token_for_fall_detection')
USER_ID = int(os.getenv('USER_ID', '1'))

def test_backend_health():
    """Test if backend is running"""
    print("\n🔍 Test 1: Backend Health Check")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running")
            print(f"   Status: {data.get('status')}")
            print(f"   Database: {data.get('database')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend")
        print(f"   Make sure backend is running at {BACKEND_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_invalid_token():
    """Test with invalid API token (should fail)"""
    print("\n🔍 Test 2: Invalid API Token (Should Fail)")
    print("=" * 50)
    
    try:
        headers = {
            'X-API-KEY': 'wrong_token',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/notify/fall-alert",
            json={'userId': USER_ID, 'timestamp': datetime.now().isoformat()},
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 403:
            print("✅ Invalid token correctly rejected (403 Forbidden)")
            return True
        else:
            print(f"⚠️  Expected 403, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_missing_token():
    """Test without API token (should fail)"""
    print("\n🔍 Test 3: Missing API Token (Should Fail)")
    print("=" * 50)
    
    try:
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(
            f"{BACKEND_URL}/api/notify/fall-alert",
            json={'userId': USER_ID, 'timestamp': datetime.now().isoformat()},
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 401:
            print("✅ Missing token correctly rejected (401 Unauthorized)")
            return True
        else:
            print(f"⚠️  Expected 401, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_valid_fall_alert():
    """Test sending a valid fall alert"""
    print("\n🔍 Test 4: Send Valid Fall Alert")
    print("=" * 50)
    
    try:
        alert_data = {
            'userId': USER_ID,
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.92
        }
        
        headers = {
            'X-API-KEY': FALL_ALERT_TOKEN,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/notify/fall-alert",
            json=alert_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            print("✅ Fall alert sent successfully!")
            print(f"   Alert ID: {data.get('fallAlertId')}")
            print(f"   Caregivers notified: {data.get('notificationsSent', {}).get('caregivers', 0)}")
            print(f"   Emergency contacts: {data.get('notificationsSent', {}).get('emergencyContacts', 0)}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_missing_fields():
    """Test with missing required fields"""
    print("\n🔍 Test 5: Missing Required Fields (Should Fail)")
    print("=" * 50)
    
    try:
        headers = {
            'X-API-KEY': FALL_ALERT_TOKEN,
            'Content-Type': 'application/json'
        }
        
        # Missing timestamp
        response = requests.post(
            f"{BACKEND_URL}/api/notify/fall-alert",
            json={'userId': USER_ID},
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 400:
            print("✅ Missing fields correctly rejected (400 Bad Request)")
            return True
        else:
            print(f"⚠️  Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 50)
    print("🚀 FALL DETECTION INTEGRATION TEST SUITE")
    print("=" * 50)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"User ID: {USER_ID}")
    print(f"Token configured: {'Yes' if FALL_ALERT_TOKEN else 'No'}")
    
    results = {
        'Health Check': test_backend_health(),
        'Invalid Token': test_invalid_token(),
        'Missing Token': test_missing_token(),
        'Valid Alert': test_valid_fall_alert(),
        'Missing Fields': test_missing_fields()
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 50)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)