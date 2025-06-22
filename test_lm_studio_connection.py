#!/usr/bin/env python3
"""Test LM Studio connection and response time"""

import requests
import time

def test_lm_studio():
    print("Testing LM Studio connection...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        print(f"✅ Server status: {response.status_code}")
        models = response.json()
        print(f"✅ Available models: {models}")
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return
    
    # Test 2: Simple completion
    print("\nTesting simple completion...")
    
    payload = {
        "messages": [
            {"role": "system", "content": "Reply with one word only."},
            {"role": "user", "content": "Say hello"}
        ],
        "max_tokens": 10,
        "temperature": 0.1,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Response time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Request failed after {elapsed:.2f}s: {e}")
    
    # Test 3: Check server load
    print("\nChecking server metrics...")
    try:
        # Niektóre modele LM Studio mogą mieć endpoint metrics
        response = requests.get("http://127.0.0.1:1234/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Server metrics available")
        else:
            print("ℹ️ Server metrics not available (normal)")
    except:
        print("ℹ️ Server metrics endpoint not found (normal)")

if __name__ == "__main__":
    test_lm_studio() 