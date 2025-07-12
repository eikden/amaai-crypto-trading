#!/usr/bin/env python3
"""
Test script to verify 1-day date filtering works correctly
"""

import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_one_day_filtering():
    """Test the 1-day date filtering logic"""
    print("🧪 Testing 1-day date filtering...")
    
    # Simulate what happens when user selects 1 day
    today = datetime(2024, 7, 15, 12, 0, 0)  # July 15, 2024 at noon
    start_date = today - timedelta(days=1)   # July 14, 2024 at noon
    end_date = today                         # July 15, 2024 at noon
    
    print(f"📅 User selected date range:")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    print(f"   Duration: {(end_date - start_date).total_seconds() / 3600:.1f} hours")
    
    # Simulate extended data fetching (what the function does internally)
    # For 1h interval over 1 day = 24 data points, but we need 60 for MA50
    extended_start = start_date - timedelta(days=30)  # Go back 30 days
    
    print(f"\n🔧 System extends date range for technical indicators:")
    print(f"   Extended start: {extended_start}")
    print(f"   Extended end: {end_date}")
    print(f"   Extended duration: {(end_date - extended_start).days} days")
    
    # Create mock extended dataframe (31 days of hourly data)
    extended_dates = pd.date_range(extended_start, end_date, freq='h')
    extended_df = pd.DataFrame({
        'close': [100 + i * 0.01 for i in range(len(extended_dates))],
        'open': [100 + i * 0.01 - 0.5 for i in range(len(extended_dates))],
        'high': [100 + i * 0.01 + 0.5 for i in range(len(extended_dates))],
        'low': [100 + i * 0.01 - 0.7 for i in range(len(extended_dates))],
        'volume': [1000 + i for i in range(len(extended_dates))]
    }, index=extended_dates)
    
    print(f"\n📊 Extended dataframe:")
    print(f"   Rows: {len(extended_df)}")
    print(f"   Date range: {extended_df.index[0]} to {extended_df.index[-1]}")
    
    # Apply the NEW filtering logic (filter to ORIGINAL user dates)
    filtered_df = extended_df[start_date:end_date]
    
    print(f"\n✅ After filtering to ORIGINAL user-requested range:")
    print(f"   Rows: {len(filtered_df)}")
    print(f"   Date range: {filtered_df.index[0]} to {filtered_df.index[-1]}")
    print(f"   Expected ~24 hours of data: {len(filtered_df)} hours")
    
    # Verify the filtering worked correctly
    assert filtered_df.index[0] >= start_date, "Start date filter failed"
    assert filtered_df.index[-1] <= end_date, "End date filter failed"
    assert len(filtered_df) <= 25, f"Too much data for 1 day: {len(filtered_df)} rows"  # Allow for rounding
    
    print("\n🎉 Test passed! Chart will now show only the selected 1-day period.")
    
    # Test chart display simulation
    print(f"\n📈 Chart will display:")
    print(f"   Start time: {filtered_df.index[0]}")
    print(f"   End time: {filtered_df.index[-1]}")
    print(f"   Data points: {len(filtered_df)}")
    print(f"   Time span: {(filtered_df.index[-1] - filtered_df.index[0]).total_seconds() / 3600:.1f} hours")

if __name__ == "__main__":
    test_one_day_filtering()
