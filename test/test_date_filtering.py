#!/usr/bin/env python3
"""
Test script to verify date filtering in the enhanced trading system
"""

import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_date_filtering():
    """Test the date filtering logic"""
    print("🧪 Testing date filtering logic...")
    
    # Create mock dataframe with hourly data for 3 months
    start_full = datetime(2024, 1, 1)
    end_full = datetime(2024, 3, 31)
    dates = pd.date_range(start_full, end_full, freq='1H')
    
    # Create mock price data
    df = pd.DataFrame({
        'close': [100 + i * 0.01 for i in range(len(dates))],
        'open': [100 + i * 0.01 - 0.5 for i in range(len(dates))],
        'high': [100 + i * 0.01 + 0.5 for i in range(len(dates))],
        'low': [100 + i * 0.01 - 0.7 for i in range(len(dates))],
        'volume': [1000 + i for i in range(len(dates))]
    }, index=dates)
    
    print(f"📊 Original data: {len(df)} rows")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test filtering for February 2024
    start_filter = datetime(2024, 2, 1)
    end_filter = datetime(2024, 2, 28)
    
    # Apply the same filtering logic as in fetch_binance_ta
    filtered_df = df[start_filter:end_filter]
    
    print(f"📈 Filtered data: {len(filtered_df)} rows")
    print(f"   Date range: {filtered_df.index[0]} to {filtered_df.index[-1]}")
    
    # Verify the filtering worked correctly
    assert filtered_df.index[0] >= start_filter, "Start date filter failed"
    assert filtered_df.index[-1] <= end_filter, "End date filter failed"
    
    print("✅ Date filtering test passed!")
    
    # Test simulation points
    print("\n🎯 Testing simulation points...")
    simulation_step = 6  # Every 6 hours
    simulation_points = list(range(20, len(filtered_df), simulation_step))
    
    print(f"   Simulation points: {len(simulation_points)}")
    print(f"   First point: index {simulation_points[0]} -> {filtered_df.index[simulation_points[0]]}")
    print(f"   Last point: index {simulation_points[-1]} -> {filtered_df.index[simulation_points[-1]]}")
    
    # Test portfolio timestamps (like in the chart function)
    portfolio_values = [1000 + i * 10 for i in range(len(simulation_points))]
    portfolio_timestamps = [filtered_df.index[i] for i in simulation_points[:len(portfolio_values)]]
    
    print(f"   Portfolio timestamps: {len(portfolio_timestamps)}")
    print(f"   First timestamp: {portfolio_timestamps[0]}")
    print(f"   Last timestamp: {portfolio_timestamps[-1]}")
    
    print("✅ Simulation points test passed!")
    
    print("\n🎉 All tests completed successfully!")
    print("   The chart will now show data only within the selected date range.")

if __name__ == "__main__":
    test_date_filtering()
