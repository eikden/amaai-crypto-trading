#!/usr/bin/env python3
"""
Test script for enhanced charts with RSI, MACD, and Bollinger Bands
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.subplots as sp

def create_test_data_with_indicators():
    """Create test data with all technical indicators"""
    np.random.seed(42)
    base_price = 100
    prices = []
    dates = []
    volumes = []

    # Create realistic price data
    for i in range(100):
        if i == 0:
            price = base_price
        else:
            # Add some trend and volatility
            trend = 0.001 * np.sin(i * 0.1)  # Gentle sine wave trend
            noise = np.random.normal(0, 0.02)  # 2% daily volatility
            price = prices[-1] * (1 + trend + noise)
        
        prices.append(price)
        dates.append(datetime.now() - timedelta(hours=100-i))
        volumes.append(np.random.randint(1000, 10000))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'close': prices,
        'volume': volumes
    })

    # Calculate all technical indicators
    # Simple Moving Averages
    df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['close'], window=50)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['UpperBB'] = bollinger.bollinger_hband()
    df['MidBB'] = bollinger.bollinger_mavg()
    df['LowerBB'] = bollinger.bollinger_lband()
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Remove NaN values
    df = df.dropna()
    df.set_index('timestamp', inplace=True)
    
    return df

def create_enhanced_chart(df):
    """Create enhanced chart similar to the main application"""
    # Create subplots
    fig = sp.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.15, 0.15, 0.25],
        subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD', 'Volume'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}], 
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # ===== MAIN PRICE CHART WITH BOLLINGER BANDS =====
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['UpperBB'],
            name='Upper BB',
            line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
            fill=None,
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['LowerBB'],
            name='Lower BB',
            line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
            fill='tonexty',
            fillcolor='rgba(173, 204, 255, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA20'],
            name='MA20',
            line=dict(color='orange', width=2),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # ===== RSI CHART =====
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # RSI overbought/oversold levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2)
    
    # ===== MACD CHART =====
    # MACD Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_line'],
            name='MACD Line',
            line=dict(color='blue', width=2),
            showlegend=True
        ),
        row=3, col=1
    )
    
    # MACD Signal Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_signal'],
            name='MACD Signal',
            line=dict(color='red', width=2),
            showlegend=True
        ),
        row=3, col=1
    )
    
    # MACD Histogram
    colors = ['green' if x >= 0 else 'red' for x in df['MACD_hist']]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_hist'],
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.6,
            showlegend=True
        ),
        row=3, col=1
    )
    
    # MACD zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=3)
    
    # ===== VOLUME =====
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7,
            showlegend=True
        ),
        row=4, col=1
    )
    
    # ===== LAYOUT CONFIGURATION =====
    fig.update_layout(
        title='📈 Enhanced Trading Chart - Test Data',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update subplot titles and axis labels
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def test_enhanced_charts():
    """Test the enhanced charting functionality"""
    print("🚀 Testing Enhanced Charts with Technical Indicators")
    print("=" * 60)
    
    # Create test data
    df = create_test_data_with_indicators()
    print(f"📊 Created {len(df)} data points")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📈 RSI range: {df['RSI'].min():.1f} - {df['RSI'].max():.1f}")
    print(f"📉 MACD range: {df['MACD_line'].min():.3f} - {df['MACD_line'].max():.3f}")
    
    # Check technical indicators
    print("\n📋 Technical Indicators Summary:")
    print(f"  • MA20: ${df['MA20'].iloc[-1]:.2f}")
    print(f"  • MA50: ${df['MA50'].iloc[-1]:.2f}")
    print(f"  • Upper BB: ${df['UpperBB'].iloc[-1]:.2f}")
    print(f"  • Lower BB: ${df['LowerBB'].iloc[-1]:.2f}")
    print(f"  • RSI: {df['RSI'].iloc[-1]:.1f}")
    print(f"  • MACD Line: {df['MACD_line'].iloc[-1]:.4f}")
    print(f"  • MACD Signal: {df['MACD_signal'].iloc[-1]:.4f}")
    print(f"  • MACD Histogram: {df['MACD_hist'].iloc[-1]:.4f}")
    
    # Create chart
    fig = create_enhanced_chart(df)
    
    # Save as HTML
    fig.write_html("enhanced_chart_test.html")
    print(f"\n✅ Enhanced chart saved as 'enhanced_chart_test.html'")
    print("✅ Chart includes:")
    print("  • 📈 Candlestick price chart")
    print("  • 📊 Bollinger Bands (Upper, Lower, and fill)")
    print("  • 📉 Moving Average (MA20)")
    print("  • 📈 RSI with overbought/oversold levels")
    print("  • 📊 MACD with line, signal, and histogram")
    print("  • 📊 Volume chart")
    
    # Show some trading signals based on indicators
    print(f"\n🎯 Current Market Signals:")
    current_rsi = df['RSI'].iloc[-1]
    current_price = df['close'].iloc[-1]
    current_upper_bb = df['UpperBB'].iloc[-1]
    current_lower_bb = df['LowerBB'].iloc[-1]
    current_macd_hist = df['MACD_hist'].iloc[-1]
    
    signals = []
    if current_rsi < 30:
        signals.append("🟢 RSI Oversold - Potential Buy Signal")
    elif current_rsi > 70:
        signals.append("🔴 RSI Overbought - Potential Sell Signal")
    
    if current_price <= current_lower_bb:
        signals.append("🟢 Price at Lower Bollinger Band - Potential Buy")
    elif current_price >= current_upper_bb:
        signals.append("🔴 Price at Upper Bollinger Band - Potential Sell")
    
    if current_macd_hist > 0:
        signals.append("🟢 MACD Histogram Positive - Bullish Momentum")
    else:
        signals.append("🔴 MACD Histogram Negative - Bearish Momentum")
    
    if signals:
        for signal in signals:
            print(f"  • {signal}")
    else:
        print("  • 🟡 No clear signals - Market in neutral zone")

if __name__ == "__main__":
    test_enhanced_charts()
