#!/usr/bin/env python3
"""
Test script for data fetching and indicator calculation
"""

import logging
import pandas as pd
from datetime import datetime
import sys
import os

# Add the current directory to sys.path to import from auto_trade
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_fetch():
    """Test data fetching with various scenarios"""
    try:
        # Import after adding to path
        import importlib.util
        spec = importlib.util.spec_from_file_location("auto_trade", "auto-trade.py")
        auto_trade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auto_trade)
        fetch_binance_ta = auto_trade.fetch_binance_ta
        
        print("🚀 Testing Data Fetch with Technical Indicators")
        print("=" * 60)
        
        # Test with a simple date range
        symbol = 'BTC/USDT'
        interval = '1h'
        start_date = pd.Timestamp('2024-01-01')
        end_date = pd.Timestamp('2024-01-02')
        
        print(f"📊 Testing {symbol} from {start_date} to {end_date} with {interval} interval")
        
        df = fetch_binance_ta(symbol, interval, start_date, end_date)
        
        print(f"✅ Successfully fetched data: {len(df)} rows")
        print(f"📈 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Check indicators
        indicators = ['MA20', 'MA50', 'UpperBB', 'LowerBB', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist']
        
        print("\n📋 Technical Indicators Status:")
        for indicator in indicators:
            if indicator in df.columns:
                valid_count = df[indicator].notna().sum()
                total_count = len(df)
                nan_pct = ((total_count - valid_count) / total_count) * 100
                
                if valid_count > 0:
                    value_range = f"{df[indicator].min():.4f} to {df[indicator].max():.4f}"
                    print(f"  ✅ {indicator}: {valid_count}/{total_count} valid values ({nan_pct:.1f}% NaN), range: {value_range}")
                else:
                    print(f"  ❌ {indicator}: No valid values")
            else:
                print(f"  ❓ {indicator}: Column not found")
        
        # Show sample data
        print(f"\n📊 Sample data (last 5 rows):")
        sample_cols = ['close', 'MA20', 'RSI', 'UpperBB', 'LowerBB']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].tail())
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_fetch()
    if success:
        print("\n🎉 Data fetch test completed successfully!")
    else:
        print("\n💥 Data fetch test failed!")
        sys.exit(1)
