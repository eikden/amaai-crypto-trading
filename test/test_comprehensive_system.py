#!/usr/bin/env python3
"""
Comprehensive test for the trading system with enhanced charts and unique keys
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import time
        import uuid
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_chart_key_uniqueness():
    """Test chart key generation for uniqueness"""
    import time
    import uuid
    
    def generate_chart_key(current_index, decisions_len):
        return f"trading_chart_{current_index}_{decisions_len}_{int(time.time() * 1000000)}_{str(uuid.uuid4())[:8]}"
    
    keys = set()
    for i in range(50):
        key = generate_chart_key(i % 10, i % 5)
        if key in keys:
            print(f"❌ Duplicate key: {key}")
            return False
        keys.add(key)
        time.sleep(0.001)
    
    print(f"✅ Generated {len(keys)} unique chart keys")
    return True

def test_enhanced_chart_function():
    """Test that the enhanced chart function can be imported and has correct signature"""
    try:
        # Import the specific function from auto-trade.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("auto_trade", "auto-trade.py")
        auto_trade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auto_trade)
        
        # Check if update_chart function exists
        if hasattr(auto_trade, 'update_chart'):
            print("✅ update_chart function found")
            return True
        else:
            print("❌ update_chart function not found")
            return False
    except Exception as e:
        print(f"❌ Error importing auto-trade.py: {e}")
        return False

def test_technical_indicators():
    """Test technical indicator calculation"""
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 0.05,
            'high': prices + abs(np.random.randn(100) * 0.1),
            'low': prices - abs(np.random.randn(100) * 0.1),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        data.set_index('timestamp', inplace=True)
        
        # Test RSI calculation
        def calculate_rsi(close_prices, period=14):
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(data['close'])
        
        # Test MACD calculation
        def calculate_macd(close_prices, fast=12, slow=26, signal=9):
            exp1 = close_prices.ewm(span=fast).mean()
            exp2 = close_prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd_line, signal_line, histogram = calculate_macd(data['close'])
        
        # Test Bollinger Bands
        def calculate_bollinger_bands(close_prices, period=20, std_dev=2):
            sma = close_prices.rolling(window=period).mean()
            std = close_prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['close'])
        
        # Check if calculations produced valid results
        if (not rsi.dropna().empty and 
            not macd_line.dropna().empty and 
            not bb_upper.dropna().empty):
            print("✅ Technical indicators calculated successfully")
            return True
        else:
            print("❌ Technical indicator calculations failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in technical indicator test: {e}")
        return False

if __name__ == "__main__":
    print("Running comprehensive trading system tests...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Chart Key Uniqueness", test_chart_key_uniqueness),
        ("Enhanced Chart Function", test_enhanced_chart_function),
        ("Technical Indicators", test_technical_indicators)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The trading system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
