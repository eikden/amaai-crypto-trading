#!/usr/bin/env python3
"""
Test script for the new active trading strategy
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path to import from auto-trade.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trade import (
    TradingConfig, TradingDecisionAgent, FastMarketAnalystAgent, 
    FastPatternRecognitionAgent, FastRiskManagementAgent
)

def create_test_data():
    """Create realistic test data for trading"""
    np.random.seed(42)
    base_price = 100
    prices = []
    dates = []

    for i in range(50):  # Shorter test data
        if i == 0:
            price = base_price
        else:
            # Create more realistic price movement
            change = np.random.normal(0, 0.025)  # 2.5% daily volatility
            if i < 15:  # Downtrend - should trigger buy signals
                change -= 0.008
            elif i < 30:  # Sideways
                change *= 0.3
            else:  # Uptrend - should trigger sell signals
                change += 0.005
            price = prices[-1] * (1 + change)
        prices.append(price)
        dates.append(datetime.now() - timedelta(days=50-i))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000, 10000) for _ in prices]
    })

    # Add technical indicators
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['UpperBB'] = bollinger.bollinger_hband()
    df['LowerBB'] = bollinger.bollinger_lband()
    df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['close'], window=50)
    
    df = df.dropna()  # Remove NaN values from indicators
    df.set_index('timestamp', inplace=True)
    
    return df

def test_active_strategy():
    """Test the new active trading strategy"""
    print("🚀 Testing Active Trading Strategy")
    print("=" * 50)
    
    # Create test data
    df = create_test_data()
    print(f"📊 Created {len(df)} data points")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📈 RSI range: {df['RSI'].min():.1f} - {df['RSI'].max():.1f}")
    
    # Create config with active strategy settings
    config = TradingConfig(
        target_win_rate=0.75,  # 75% target
        min_confidence=0.65,   # Lower confidence threshold
        conservative_mode=False,
        initial_capital=10000
    )
    
    # Initialize agents
    market_agent = FastMarketAnalystAgent(df)
    pattern_agent = FastPatternRecognitionAgent(df)
    risk_agent = FastRiskManagementAgent()
    trading_agent = TradingDecisionAgent(config)
    
    # Simulate trading
    capital = config.initial_capital
    position = {"holding": False, "entry_price": 0, "shares": 0}
    trades = []
    trade_signals = []
    
    print(f"\n🎯 Starting simulation with ${capital:,.2f}")
    print(f"🎯 Target Win Rate: {config.target_win_rate:.1%}")
    print(f"🎯 Min Confidence: {config.min_confidence:.1%}")
    print()
    
    for i in range(20, len(df)):  # Start after enough data for indicators
        row = df.iloc[i]
        
        # Get agent analyses
        market_analysis = market_agent.analyze(i)
        pattern_analysis = pattern_agent.analyze(i)
        risk_assessment = risk_agent.analyze(
            current_price=row.close,
            entry_price=position.get("entry_price", row.close),
            position_size=position.get("shares", 0),
            account_balance=capital
        )
        
        # Calculate current win rate
        wins = sum(1 for t in trades if t["profit_pct"] > 0)
        current_win_rate = wins / len(trades) if trades else 0.8  # Start optimistic
        
        # Get trading decision
        decision = trading_agent.make_decision(
            market_analysis, pattern_analysis, risk_assessment, position, current_win_rate
        )
        
        trade_signals.append({
            'timestamp': row.name,
            'price': row.close,
            'action': decision.action.value,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning[:100] + "..." if len(decision.reasoning) > 100 else decision.reasoning
        })
        
        # Execute trades
        if decision.action.value == "BUY" and not position["holding"]:
            shares = capital * 0.9 / row.close  # Use 90% of capital (10% fee)
            position = {
                "holding": True,
                "entry_price": row.close,
                "shares": shares
            }
            capital *= 0.1  # Keep 10% as cash (simulating fee)
            print(f"📈 BUY  @ ${row.close:.2f} | Conf: {decision.confidence:.1%} | {decision.reasoning[:60]}...")
            
        elif decision.action.value == "SELL" and position["holding"]:
            sell_value = position["shares"] * row.close
            profit_pct = ((row.close - position["entry_price"]) / position["entry_price"]) * 100
            
            trades.append({
                "entry_price": position["entry_price"],
                "exit_price": row.close,
                "profit_pct": profit_pct,
                "profit_amount": sell_value - (position["shares"] * position["entry_price"])
            })
            
            capital = sell_value
            position = {"holding": False, "entry_price": 0, "shares": 0}
            print(f"📉 SELL @ ${row.close:.2f} | Conf: {decision.confidence:.1%} | Profit: {profit_pct:+.1f}% | {decision.reasoning[:40]}...")
    
    # Final results
    print("\n" + "=" * 50)
    print("📊 TRADING RESULTS")
    print("=" * 50)
    
    total_signals = len([s for s in trade_signals if s['action'] != 'HOLD'])
    buy_signals = len([s for s in trade_signals if s['action'] == 'BUY'])
    sell_signals = len([s for s in trade_signals if s['action'] == 'SELL'])
    
    print(f"📊 Total Signals Generated: {total_signals}")
    print(f"📈 Buy Signals: {buy_signals}")
    print(f"📉 Sell Signals: {sell_signals}")
    print(f"🔄 Completed Trades: {len(trades)}")
    
    if trades:
        wins = sum(1 for t in trades if t["profit_pct"] > 0)
        win_rate = wins / len(trades)
        avg_profit = sum(t["profit_pct"] for t in trades) / len(trades)
        total_return = (capital - config.initial_capital) / config.initial_capital * 100
        
        print(f"🎯 Win Rate: {win_rate:.1%}")
        print(f"📈 Average Trade: {avg_profit:+.2f}%")
        print(f"💰 Total Return: {total_return:+.2f}%")
        print(f"💵 Final Capital: ${capital:,.2f}")
        
        if win_rate >= 0.7:
            print("✅ SUCCESS: Win rate target achieved!")
        else:
            print("⚠️  Below 70% win rate target")
            
        if total_signals >= 5:
            print("✅ SUCCESS: Strategy is generating trades!")
        else:
            print("⚠️  Strategy may be too conservative")
    else:
        print("⚠️  No completed trades - strategy too conservative")
    
    # Show some sample signals
    print(f"\n📋 Sample Signals:")
    for signal in trade_signals[-10:]:
        if signal['action'] != 'HOLD':
            print(f"  {signal['timestamp'].strftime('%Y-%m-%d')} | {signal['action']} @ ${signal['price']:.2f} | {signal['confidence']:.1%}")

if __name__ == "__main__":
    test_active_strategy()
