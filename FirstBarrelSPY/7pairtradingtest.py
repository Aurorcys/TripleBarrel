#Im going to test with yf on pair trading, then i will transition to alpaca
import yfinance as yf
import numpy as np
import pandas as pd

df = yf.download(['SPY', 'QQQ'], period='10y', interval='1d')

# FIX 1: If MultiIndex, flatten it properly
if isinstance(df.columns, pd.MultiIndex):
    print("MultiIndex detected - flattening...")
    df.columns = [f'{col[1]}_{col[0]}' for col in df.columns]

spy_close = df['SPY_Close']
qqq_close = df['QQQ_Close']
spy_volume = df['SPY_Volume']
qqq_volume = df['QQQ_Volume']

df['Close_Ratio'] = qqq_close / spy_close
df['volume_ratio'] = qqq_volume / spy_volume

#z score on ratio
lookback = 20
df['ratio_z'] = (df['Close_Ratio'] - df['Close_Ratio'].rolling(lookback).mean()
                 )  / df['Close_Ratio'].rolling(lookback).std().replace(0, 0.001)

#rolling beta
#beta adjusted spread
returns_spy = spy_close.pct_change()
returns_qqq = qqq_close.pct_change()
df['beta'] = returns_qqq.rolling(60).cov(returns_spy) / returns_spy.rolling(60).var().replace(0, 0.001)
df['beta_adj_spread'] = qqq_close - (df['beta'] * spy_close)

#sector momentum differential
#XLK is a tech ETF vs SPY momentum
xlk = yf.download('XLK', period='10y', interval='1d')
if isinstance(xlk.columns, pd.MultiIndex):
    print("MultiIndex detected - flattening...")
    xlk.columns = [f'{col[1]}_{col[0]}' for col in xlk.columns]
xlk_close = xlk['XLK_Close']
df['tech_momentum'] = xlk_close.pct_change(20) - spy_close.pct_change(20)
df['sector_extreme'] = (df['tech_momentum'].rank(pct=True) > 0.9) | (df['tech_momentum'].rank(pct=True) < 0.1)

#volatility regime
vxn = yf.download('^VXN', period='10y', interval='1d')
vix = yf.download('^VIX', period='10y', interval='1d')
if isinstance(vxn.columns, pd.MultiIndex):
    print("MultiIndex detected - flattening...")
    vxn.columns = [f'{col[1]}_{col[0]}' for col in vxn.columns]
if isinstance(vix.columns, pd.MultiIndex):
    print("MultiIndex detected - flattening...")
    vix.columns = [f'{col[1]}_{col[0]}' for col in vix.columns]
print(vix.columns)
vix_close = vix['^VIX_Close']
vxn_close = vxn['^VXN_Close']
df['vol_spread'] = vxn_close - vix_close
df['vol_spread_z'] = (df['vol_spread'] - df['vol_spread'].rolling(20).mean()) / df['vol_spread'].rolling(20).std().replace(0, 0.001)

#liquidity flow
df['volume_ratio'] = qqq_volume / spy_volume
df['dollar_flow'] = (qqq_close * qqq_volume) - (spy_close * spy_volume)

#confidence score
df['confidence_ratio'] = 40 * (np.minimum(abs(df['ratio_z']), 4) / 4)

#vol con
vol_aligned = ((df['ratio_z'] > 0) & (df['vol_spread_z'] > 1)) | ((df['ratio_z'] < 0) & (df['vol_spread_z'] < -1))
df['confidence_vol'] = np.where(vol_aligned, 25 * (np.minimum(abs(df['vol_spread_z']), 3) / 3), 0)

#momentum extremity
momentum_rank = df['tech_momentum'].rolling(252).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)
df['confidence_momentum'] = np.where(((df['ratio_z'] > 0) & (momentum_rank > 0.7)) | \
                                     ((df['ratio_z'] < 0 ) & (momentum_rank < 0.3)),
                                     20 * abs(momentum_rank - 0.5) * 2, 0)

#volume con
df['confidence_volume'] = np.where(
    df['volume_ratio'] > 1.2,
    10 * np.minimum((df['volume_ratio'] - 1.2) / 1.0, 1),
    0
)

# Total confidence
df['confidence'] = df['confidence_ratio'] + df['confidence_vol'] + df['confidence_momentum'] + df['confidence_volume']

#signal time
print('signals..')

entry_thres = 2
exit_thres = 0.5

df['entry_sig'] = (abs(df['ratio_z']) > entry_thres).astype(int)
df['dir'] = np.where(df['ratio_z'] > 0, -1, 1)

df['signal_tier'] = np.where(
    df['confidence'] >= 70, 'TIER_1',
    np.where(
        df['confidence'] >= 50, 'TIER_2',
        np.where(
            df['confidence'] >= 30, 'TIER_3',
            'NO_SIGNAL'
        )
    )
)

print('SIGNAL stats')
print(f'TOTAL SIGS: {df['entry_sig'].sum()}')
print(f'TIER 1 SIGS: {(df['signal_tier'] == 'TIER_1').sum()}')
print(f'TIER 2 SIGS: {(df['signal_tier'] == 'TIER_2').sum()}')
print(f'TIER 3 SIGS: {(df['signal_tier'] == 'TIER_3').sum()}')
print(f'NO SIGNAL: {(df['signal_tier'] == 'NO_SIGNAL').sum()}')

#backtest
cash = 10000
pos = 0
shares = 0
portfolio = []
trades = []

for i in range(len(df)):
    current_price = df['SPY_Close'].iloc[i]
    current_ratio_z = df['ratio_z'].iloc[i]
    current_signal = df['entry_sig'].iloc[i]
    current_dir = df['dir'].iloc[i]
    current_confidence = df['confidence'].iloc[i]

    if pos == 0 and current_signal == 1 and current_confidence >= 30:
        risk_pct = 0.01 + (current_confidence / 100) * 0.02
        pos = 1
        shares = (cash * risk_pct) / current_price
        trades.append({
            'entry_idx': i,
            'currentclose': current_price,
            'direction': current_dir,
            'shares': shares,
            'cost': shares * current_price,
            'confidence': current_confidence
        })
        cash = cash - (shares * current_price)
    elif pos == 1:
        trade = trades[-1]
        days_held = i - trade['entry_idx']

        exit_condition = (
            abs(current_ratio_z) < exit_thres or
            days_held > 21 or
            (current_ratio_z * trade['direction'] > 3)
        )
        if exit_condition:
            cash += shares*current_price
            trades[-1]['exit_idx'] = i
            trades[-1]['exit_price'] = current_price
            trades[-1][days_held] = days_held
            pos = 0
            shares = 0
    portfolio.append(cash + shares*current_price)


print(f"\n📈 BACKTEST RESULTS:")
print(f"Initial capital: $10,000")
print(f"Final capital: ${portfolio[-1]:.2f}")
print(f"Total return: {((portfolio[-1]/10000)-1)*100:.2f}%")
print(f"Total trades: {len(trades)}")


casb = 10000
sharesb = 0
portfoliob = []
#buy and hold
for i in range(len(df)):
    if i == 0:
        sharesb = casb / df['SPY_Close'].iloc[i]
    portfoliob.append(sharesb * df['SPY_Close'].iloc[i])
    casb = 0

print(f"\nBH RESULTS:")
print(f"Initial capital: $10,000")
print(f"Final capital: ${portfoliob[-1]:.2f}")
print(f"Total return: {((portfoliob[-1]/10000)-1)*100:.2f}%")
