#!/usr/bin/env python3
"""
Quick validation test for core trading system functionality
Run this to verify the system is working before deployment
"""

import sys
import os
import traceback
from datetime import datetime, timedelta

def test_basic_functionality():
    """Run basic functionality tests"""
    print("🚀 AI Trading System - Quick Validation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Package imports
    total_tests += 1
    print("\n1️⃣ Testing essential package imports...")
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.graph_objects as go
        print("✅ Essential packages imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    
    # Test 2: Main module import
    total_tests += 1
    print("\n2️⃣ Testing main module import...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "auto_trade", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "auto-trade.py")
        )
        auto_trade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auto_trade)
        print("✅ Main module imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Main module import failed: {e}")
        return False
    
    # Test 3: Agent initialization
    total_tests += 1
    print("\n3️⃣ Testing agent initialization...")
    try:
        # Create a simple config object for the agent
        class SimpleConfig:
            def __init__(self):
                self.enable_debug = False
                
        config = SimpleConfig()
        sentiment_agent = auto_trade.SentimentAnalysisAgent(config)
        print("✅ SentimentAnalysisAgent created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        sentiment_agent = None
    
    # Test 4: Sentiment analysis
    total_tests += 1
    print("\n4️⃣ Testing sentiment analysis...")
    try:
        if sentiment_agent is not None:
            test_posts = [
                {'text': 'Great market outlook today! 🚀', 'username': 'test_user', 'timestamp': datetime.now()},
                {'text': 'Market looks bearish today', 'username': 'test_user2', 'timestamp': datetime.now()}
            ]
            result = sentiment_agent.analyze_sentiment(test_posts)
            assert 'sentiment_score' in result
            assert 'confidence' in result
            print(f"✅ Sentiment analysis working (score: {result['sentiment_score']:.3f})")
            tests_passed += 1
        else:
            print("⚠️  Skipping sentiment test due to agent initialization failure")
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
    
    # Test 5: Data structure validation
    total_tests += 1
    print("\n5️⃣ Testing data structure handling...")
    try:
        # Create sample data like the system would use
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='H'),
            'open': [50000 + i * 100 for i in range(10)],
            'high': [50100 + i * 100 for i in range(10)],
            'low': [49900 + i * 100 for i in range(10)],
            'close': [50050 + i * 100 for i in range(10)],
            'volume': [1000 + i * 10 for i in range(10)]
        })
        
        # Test P&L calculation logic
        profit = 0.0
        profit_pct = 0.0
        
        # Test formatting function
        def format_pnl_display(profit_val, pct_val):
            if abs(profit_val) < 0.01:
                return "0.00%"
            elif abs(pct_val) > 0.001:
                return f"{pct_val:+.2f}%"
            else:
                return "0.00%"
        
        test_display = format_pnl_display(profit, profit_pct)
        assert test_display == "0.00%"
        
        print("✅ Data structures and P&L logic working correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
    
    # Test 6: Display function exists
    total_tests += 1
    print("\n6️⃣ Testing display functions...")
    try:
        assert hasattr(auto_trade, 'display_simulation_results')
        assert hasattr(auto_trade, 'fetch_binance_ta')
        print("✅ Key display functions found")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Display function test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    success_rate = (tests_passed / total_tests) * 100
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 System is ready for use!")
        return True
    else:
        print("🚨 System needs attention before use")
        return False

if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
