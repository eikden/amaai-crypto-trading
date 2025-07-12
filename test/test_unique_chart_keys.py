#!/usr/bin/env python3
"""
Test script to verify unique chart keys are generated
"""

import time
import uuid

def generate_chart_key(current_index, decisions_len):
    """Test the chart key generation logic"""
    return f"trading_chart_{current_index}_{decisions_len}_{int(time.time() * 1000000)}_{str(uuid.uuid4())[:8]}"

def test_unique_keys():
    """Test that multiple calls generate unique keys"""
    keys = set()
    
    # Generate 100 keys with same parameters
    for i in range(100):
        key = generate_chart_key(1, 5)
        if key in keys:
            print(f"❌ Duplicate key found: {key}")
            return False
        keys.add(key)
        
        # Small delay to ensure timestamp differences
        time.sleep(0.001)
    
    print(f"✅ Generated {len(keys)} unique keys")
    return True

def test_key_format():
    """Test that keys have the expected format"""
    key = generate_chart_key(10, 25)
    parts = key.split('_')
    
    if len(parts) != 6:
        print(f"❌ Expected 6 parts, got {len(parts)}: {key}")
        return False
    
    if parts[0] != "trading" or parts[1] != "chart":
        print(f"❌ Wrong prefix: {parts[0]}_{parts[1]}")
        return False
    
    try:
        current_index = int(parts[2])
        decisions_len = int(parts[3])
        timestamp = int(parts[4])
        uuid_part = parts[5]
        
        print(f"✅ Key format correct: index={current_index}, decisions={decisions_len}, timestamp={timestamp}, uuid={uuid_part}")
        return True
    except ValueError as e:
        print(f"❌ Invalid number format in key: {e}")
        return False

if __name__ == "__main__":
    print("Testing unique chart key generation...")
    
    success = True
    success &= test_unique_keys()
    success &= test_key_format()
    
    if success:
        print("\n🎉 All tests passed! Chart keys should be unique now.")
    else:
        print("\n❌ Some tests failed.")
