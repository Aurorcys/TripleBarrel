import yfinance as yf
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("🚀 STARTING SPY/QQQ PAIRS TRADING BACKTEST")
print("=" * 60)

# ============================================================================
# 1. GET DATA PROPERLY - SIMPLEST WAY
# ============================================================================
print("📥 Downloading SPY and QQQ data...")

# Download with auto_adjust=True (default now) - fewer columns
spy = yf.download('SPY', period='5y', interval='1d', progress=False)
qqq = yf.download('QQQ', period='5y', interval='1d', progress=False)

print(f"\nSPY columns: {spy.columns.tolist()}")
print(f"QQQ columns: {qqq.columns.tolist()}")

# Just use whatever columns we get - typically ['Open', 'High', 'Low', 'Close', 'Volume']
# Create aligned DataFrame
data = pd.DataFrame(index=spy.index)

# Use .iloc[:, #] to avoid column name issues
data['SPY_Close'] = spy.iloc[:, 3]  # Usually column 3 is Close
data['QQQ_Close'] = qqq.iloc[:, 3]
data['SPY_Volume'] = spy.iloc[:, 4]  # Usually column 4 is Volume
data['QQQ_Volume'] = qqq.iloc[:, 4]

print(f"✅ Data loaded: {len(data)} trading days")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")

# ============================================================================
# 2. CALCULATE RATIO AND Z-SCORE
# ============================================================================
print("\n📊 Calculating QQQ/SPY ratio...")

data['Close_Ratio'] = data['QQQ_Close'] / data['SPY_Close']

# Z-score on ratio
lookback = 20
data['ratio_mean'] = data['Close_Ratio'].rolling(lookback).mean()
data['ratio_std'] = data['Close_Ratio'].rolling(lookback).std()
data['ratio_z'] = (data['Close_Ratio'] - data['ratio_mean']) / data['ratio_std'].replace(0, 0.001)

print(f"Ratio stats: Mean={data['Close_Ratio'].mean():.3f}, Std={data['Close_Ratio'].std():.3f}")
print(f"Z-score range: [{data['ratio_z'].min():.2f}, {data['ratio_z'].max():.2f}]")

# ============================================================================
# 3. GET VOLATILITY DATA
# ============================================================================
print("\n📈 Fetching volatility data (VIX & VXN)...")

vix = yf.download('^VIX', period='5y', interval='1d', progress=False)
vxn = yf.download('^VXN', period='5y', interval='1d', progress=False)

# Align with main data
data['VIX'] = vix.iloc[:, 3].reindex(data.index, method='ffill')  # Close price
data['VXN'] = vxn.iloc[:, 3].reindex(data.index, method='ffill')

# Volatility spread
data['vol_spread'] = data['VXN'] - data['VIX']
data['vol_spread_z'] = (data['vol_spread'] - data['vol_spread'].rolling(20).mean()) / data['vol_spread'].rolling(20).std().replace(0, 0.001)

print(f"VIX range: [{data['VIX'].min():.1f}, {data['VIX'].max():.1f}]")
print(f"VXN range: [{data['VXN'].min():.1f}, {data['VXN'].max():.1f}]")

# ============================================================================
# 4. VOLUME AND CONFIDENCE SCORING
# ============================================================================
print("\n⚖️ Calculating confidence scores...")

# Volume ratio
data['volume_ratio'] = data['QQQ_Volume'] / data['SPY_Volume'].replace(0, 0.001)

# Confidence scoring (SIMPLIFIED - 0 to 100 points)
# 1. Ratio extremity (0-40 points)
data['conf_ratio'] = 40 * (np.minimum(abs(data['ratio_z']), 4) / 4)

# 2. Volatility alignment (0-30 points)
# If ratio_z and vol_spread_z have same sign, that's confirmation
vol_alignment = np.sign(data['ratio_z']) == np.sign(data['vol_spread_z'])
data['conf_vol'] = np.where(vol_alignment, 30 * np.minimum(abs(data['vol_spread_z']), 3) / 3, 0)

# 3. Volume confirmation (0-30 points)
data['conf_volume'] = np.where(
    data['volume_ratio'] > 1.2,
    30 * np.minimum((data['volume_ratio'] - 1.2) / 1.0, 1),
    0
)

# Total confidence
data['confidence'] = data['conf_ratio'] + data['conf_vol'] + data['conf_volume']

print(f"Confidence range: [{data['confidence'].min():.0f}, {data['confidence'].max():.0f}]")

# ============================================================================
# 5. SIGNAL GENERATION
# ============================================================================
print("\n🎯 Generating trading signals...")

entry_thres = 2.0  # Z > 2 for entry
exit_thres = 0.5   # Z < 0.5 for exit

data['entry_signal'] = (abs(data['ratio_z']) > entry_thres).astype(int)

# Direction: If QQQ/SPY ratio is high (z > 0), QQQ is expensive relative to SPY
# So we short QQQ and long SPY (direction = -1)
# If ratio is low (z < 0), SPY is expensive relative to QQQ
# So we short SPY and long QQQ (direction = 1)
data['direction'] = np.where(data['ratio_z'] > 0, -1, 1)

# Signal tiers
data['signal_tier'] = 'NO_SIGNAL'
signal_mask = data['entry_signal'] == 1
data.loc[signal_mask, 'signal_tier'] = np.where(
    data.loc[signal_mask, 'confidence'] >= 70, 'TIER_1',
    np.where(data.loc[signal_mask, 'confidence'] >= 50, 'TIER_2', 'TIER_3')
)

print(f"📊 SIGNAL STATISTICS:")
print(f"Total days: {len(data):,}")
print(f"Entry signals: {data['entry_signal'].sum()}")
print(f"Tier 1 signals: {(data['signal_tier'] == 'TIER_1').sum()}")
print(f"Tier 2 signals: {(data['signal_tier'] == 'TIER_2').sum()}")
print(f"Tier 3 signals: {(data['signal_tier'] == 'TIER_3').sum()}")
print(f"No signal: {(data['signal_tier'] == 'NO_SIGNAL').sum()}")

# ============================================================================
# 6. BACKTEST - SIMPLE VERSION (DOLLAR NEUTRAL)
# ============================================================================
print(f"\n💰 BACKTESTING PAIRS STRATEGY...")

initial_capital = 10000
cash = initial_capital
position = 0  # 0: flat, 1: position active
entry_idx = 0
trades = []
portfolio_value = []
trade_size_per_side = 5000  # $5000 long, $5000 short = $10k total exposure

for i in range(lookback, len(data)-1):
    spy_price = data['SPY_Close'].iloc[i]
    qqq_price = data['QQQ_Close'].iloc[i]
    current_z = data['ratio_z'].iloc[i]
    
    # ENTRY: Signal with confidence > 30
    if position == 0 and data['entry_signal'].iloc[i] == 1 and data['confidence'].iloc[i] > 30:
        direction = data['direction'].iloc[i]
        
        if direction == -1:  # QQQ expensive, SPY cheap
            # Long SPY, Short QQQ
            spy_shares = trade_size_per_side / spy_price
            qqq_shares = -trade_size_per_side / qqq_price  # Negative for short
            position_type = 'Short QQQ/Long SPY'
        else:  # direction == 1, SPY expensive, QQQ cheap
            # Short SPY, Long QQQ
            spy_shares = -trade_size_per_side / spy_price
            qqq_shares = trade_size_per_side / qqq_price
            position_type = 'Long QQQ/Short SPY'
        
        trades.append({
            'entry_idx': i,
            'entry_date': data.index[i],
            'direction': direction,
            'position_type': position_type,
            'spy_shares': spy_shares,
            'qqq_shares': qqq_shares,
            'spy_entry': spy_price,
            'qqq_entry': qqq_price,
            'entry_z': current_z,
            'confidence': data['confidence'].iloc[i],
            'tier': data['signal_tier'].iloc[i]
        })
        entry_idx = i
        position = 1
    
    elif position == 1:
        days_held = i - entry_idx
        trade = trades[-1]
        
        # Exit conditions
        exit_condition = (
            abs(current_z) < exit_thres or  # Returned to mean
            days_held > 21 or  # Time stop (21 days)
            (trade['direction'] == -1 and current_z < -3) or  # Stop loss for short QQQ
            (trade['direction'] == 1 and current_z > 3)  # Stop loss for long QQQ
        )
        
        if exit_condition:
            # Calculate P&L - FIXED to ensure it's always calculated
            spy_pnl = trade['spy_shares'] * (spy_price - trade['spy_entry'])
            qqq_pnl = trade['qqq_shares'] * (qqq_price - trade['qqq_entry'])
            total_pnl = spy_pnl + qqq_pnl
            
            # Update cash
            cash += total_pnl
            
            # Update trade record - FIXED to always include P&L
            trade['exit_idx'] = i
            trade['exit_date'] = data.index[i]
            trade['spy_exit'] = spy_price
            trade['qqq_exit'] = qqq_price
            trade['exit_z'] = current_z
            trade['pnl'] = total_pnl  # This was missing for some trades
            trade['days_held'] = days_held
            trade['return_pct'] = (total_pnl / (2 * trade_size_per_side)) * 100  # Based on total exposure
            
            position = 0


    
    # Calculate current portfolio value
    if position == 0:
        current_value = cash
    else:
        trade = trades[-1]
        spy_value = trade['spy_shares'] * spy_price
        qqq_value = trade['qqq_shares'] * qqq_price
        current_value = cash + spy_value + qqq_value
    
    portfolio_value.append(current_value)

# Also, check for any open trades at the end and close them
if position == 1:
    # Close any open position at the end
    trade = trades[-1]
    spy_pnl = trade['spy_shares'] * (data['SPY_Close'].iloc[-1] - trade['spy_entry'])
    qqq_pnl = trade['qqq_shares'] * (data['QQQ_Close'].iloc[-1] - trade['qqq_entry'])
    total_pnl = spy_pnl + qqq_pnl
    cash += total_pnl
    
    trade['exit_idx'] = len(data) - 1
    trade['exit_date'] = data.index[-1]
    trade['spy_exit'] = data['SPY_Close'].iloc[-1]
    trade['qqq_exit'] = data['QQQ_Close'].iloc[-1]
    trade['exit_z'] = data['ratio_z'].iloc[-1]
    trade['pnl'] = total_pnl
    trade['days_held'] = len(data) - 1 - trade['entry_idx']
    trade['return_pct'] = (total_pnl / (2 * trade_size_per_side)) * 100

# ============================================================================
# 7. RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("📈 BACKTEST RESULTS")
print("=" * 60)

# Strategy results
final_value = portfolio_value[-1] if portfolio_value else initial_capital
total_return_pct = (final_value / initial_capital - 1) * 100

# Buy and hold benchmark
spy_bh = initial_capital * (data['SPY_Close'] / data['SPY_Close'].iloc[lookback])
bh_value = spy_bh.iloc[-1]
bh_return_pct = (bh_value / initial_capital - 1) * 100

print(f"\n💰 STRATEGY PERFORMANCE:")
print(f"Initial capital: ${initial_capital:,.0f}")
print(f"Final capital: ${final_value:,.2f}")
print(f"Total return: {total_return_pct:.2f}%")
print(f"Total trades: {len(trades)}")

if trades:
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
    
    print(f"\n📊 TRADE STATISTICS:")
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"Losing trades: {len(losing_trades)}")
    
    if winning_trades:
        avg_win = np.mean([t['pnl'] for t in winning_trades])
        avg_win_pct = np.mean([t['return_pct'] for t in winning_trades])
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        avg_loss_pct = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        print(f"Average win: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
        print(f"Average loss: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
        
        if losing_trades and sum([abs(t['pnl']) for t in losing_trades]) > 0:
            profit_factor = sum([t['pnl'] for t in winning_trades]) / sum([abs(t['pnl']) for t in losing_trades])
            print(f"Profit factor: {profit_factor:.2f}")
    
    # Analyze by tier
    tier1_trades = [t for t in trades if t.get('tier') == 'TIER_1']
    tier2_trades = [t for t in trades if t.get('tier') == 'TIER_2']
    tier3_trades = [t for t in trades if t.get('tier') == 'TIER_3']
    
    if tier1_trades:
        tier1_wins = sum(1 for t in tier1_trades if t['pnl'] > 0)
        print(f"\n🏆 TIER 1 TRADES ({len(tier1_trades)} total):")
        print(f"  Win rate: {tier1_wins/len(tier1_trades)*100:.1f}%")
    
    if tier2_trades:
        tier2_wins = sum(1 for t in tier2_trades if t['pnl'] > 0)
        print(f"🥈 TIER 2 TRADES ({len(tier2_trades)} total):")
        print(f"  Win rate: {tier2_wins/len(tier2_trades)*100:.1f}%")
    
    if tier3_trades:
        tier3_wins = sum(1 for t in tier3_trades if t['pnl'] > 0)
        print(f"🥉 TIER 3 TRADES ({len(tier3_trades)} total):")
        print(f"  Win rate: {tier3_wins/len(tier3_trades)*100:.1f}%")

print(f"\n📊 BUY & HOLD BENCHMARK (SPY):")
print(f"Final value: ${bh_value:,.2f}")
print(f"Total return: {bh_return_pct:.2f}%")
print(f"Strategy vs B&H: {total_return_pct - bh_return_pct:.2f}% difference")

# ============================================================================
# 8. RECENT SIGNALS
# ============================================================================
print(f"\n🔍 RECENT SIGNALS (last 10 trading days):")
print("-" * 60)

recent = data.tail(10)
for idx, row in recent.iterrows():
    if row['entry_signal'] == 1:
        dir_text = "Short QQQ/Long SPY" if row['direction'] == -1 else "Long QQQ/Short SPY"
        print(f"{idx.date()}: {dir_text}")
        print(f"  Ratio Z: {row['ratio_z']:.2f} | VIX: {row['VIX']:.1f} | VXN: {row['VXN']:.1f}")
        print(f"  Confidence: {row['confidence']:.0f}/100 | Tier: {row['signal_tier']}")
        print(f"  QQQ/SPY Ratio: {row['Close_Ratio']:.3f}")
        print()

# Current market state
last_row = data.iloc[-1]
print(f"\n📅 CURRENT MARKET STATE ({data.index[-1].date()}):")
print(f"QQQ/SPY Ratio: {last_row['Close_Ratio']:.3f} (Z-score: {last_row['ratio_z']:.2f})")
print(f"VIX: {last_row['VIX']:.1f} | VXN: {last_row['VXN']:.1f} | Vol Spread: {last_row['vol_spread']:.2f}")
print(f"Volume Ratio (QQQ/SPY): {last_row['volume_ratio']:.2f}")
print(f"Confidence Score: {last_row['confidence']:.0f}/100")

if last_row['entry_signal'] == 1:
    action = "Short QQQ/Long SPY" if last_row['direction'] == -1 else "Long QQQ/Short SPY"
    print(f"🎯 ACTIVE SIGNAL: {action} (Tier: {last_row['signal_tier']})")
else:
    print(f"🎯 NO ACTIVE SIGNAL (|Z| < {entry_thres})")

print("\n" + "=" * 60)
print("✅ BACKTEST COMPLETE")
print("=" * 60)