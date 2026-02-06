import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ======================================
# DATA DOWNLOAD
# ======================================

print("\n📥 DOWNLOADING SPY DATA...")

# Get 5 years of daily data
spy = yf.download('SPY', period='5y', interval='1d', progress=False)
print(f"✅ Downloaded {len(spy)} days of SPY data")

if isinstance(spy.columns, pd.MultiIndex):
    print(f"⚠️ Detected MultiIndex columns: {spy.columns.names}")
    # KEEP THE ORIGINAL COLUMN NAMES - just use the first level
    spy.columns = spy.columns.get_level_values(0)
    print(f"✅ Fixed columns: {spy.columns.tolist()[:10]}...")

# Calculate daily returns
spy['Return'] = spy['Close'].pct_change()
spy['Log_Return'] = np.log(spy['Close'] / spy['Close'].shift(1))

# ======================================
# DATA FEATURE ENGINEERING (VOLATILITY)
# ======================================

print("\n🔧 CALCULATING VOLATILITY FEATURES...")

# 5-day realized volatility (short-term)
spy['Vol_5d'] = spy['Log_Return'].rolling(5).std() * np.sqrt(252)

# 20-day realized volatility (medium-term)
spy['Vol_20d'] = spy['Log_Return'].rolling(20).std() * np.sqrt(252)

# Volatility ratio (short/medium)
spy['Vol_Ratio'] = spy['Vol_5d'] / spy['Vol_20d'].replace(0, 0.001)

# Bollinger Band width (another volatility measure) - FIXED
rolling_mean = spy['Close'].rolling(20).mean()
rolling_std = spy['Close'].rolling(20).std()

# Extract as Series if they're DataFrames
if isinstance(rolling_mean, pd.DataFrame):
    rolling_mean = rolling_mean.iloc[:, 0]
if isinstance(rolling_std, pd.DataFrame):
    rolling_std = rolling_std.iloc[:, 0]

spy['BB_Upper'] = rolling_mean + (rolling_std * 2)
spy['BB_Lower'] = rolling_mean - (rolling_std * 2)
spy['BB_Width'] = (spy['BB_Upper'] - spy['BB_Lower']) / rolling_mean

# ATR (Average True Range) for volatility
high_low = spy['High'] - spy['Low']
high_close_prev = abs(spy['High'] - spy['Close'].shift(1))
low_close_prev = abs(spy['Low'] - spy['Close'].shift(1))
spy['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
spy['ATR_14'] = spy['TR'].rolling(14).mean()

print(f"✅ Created {sum(['Vol' in col or 'BB' in col or 'ATR' in col for col in spy.columns])} volatility features")
print(spy.columns)

# ======================================
# MEAN REVERSION SIGNALS
# ======================================

print("\n🎯 GENERATING MEAN REVERSION SIGNALS...")

# Signal 1: Extreme volatility spike (Vol_Ratio > 2.0)
spy['Signal_Vol_Spike'] = (spy['Vol_Ratio'] > 2.0).astype(int)

# Signal 2: Bollinger Band squeeze expansion (BB width > 95th percentile)
bb_width_95 = spy['BB_Width'].quantile(0.95)
spy['Signal_BB_Expand'] = (spy['BB_Width'] > bb_width_95).astype(int)

# Signal 3: High ATR relative to 20-day average
spy['ATR_Ratio'] = spy['ATR_14'] / spy['ATR_14'].rolling(20).mean()
spy['Signal_ATR_Spike'] = (spy['ATR_Ratio'] > 1.5).astype(int)

# Signal 4: Volatility cluster (high vol for 3+ days)
spy['High_Vol_Days'] = (spy['Vol_5d'] > spy['Vol_20d']).rolling(3).sum()
spy['Signal_Vol_Cluster'] = (spy['High_Vol_Days'] >= 3).astype(int)

print(f"✅ Generated {sum(['Signal_' in col for col in spy.columns])} signal types")