#I'm coding starting from scratch

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#get data
spy = yf.download('SPY', period='5y', interval='1d', progress=True)

if isinstance(spy.columns, pd.MultiIndex):
    print(f"⚠️ Detected MultiIndex columns: {spy.columns.names}")
    # Keep the original column names - just use the first level
    spy.columns = spy.columns.get_level_values(0)
    print(f"✅ Fixed columns: {spy.columns.tolist()}")
print(spy.shape)

spy['Return'] = spy['Close'].pct_change()
spy['Log_Return'] = np.log(spy['Close'] / spy['Close'].shift(1))

# =============================

#make volatility features, bband features
spy['Return'] = spy['Close'].pct_change()
spy['MBand'] = spy['Close'].rolling(window=20).mean()
spy['UBand'] = spy['MBand'] + 2 * spy['Close'].rolling(window=20).std()
spy['LBand'] = spy['MBand'] - 2 * spy['Close'].rolling(window=20).std()
print(spy.shape)

# =============================

#mean reversion features
"""
RSI
Z SCORE
VOL RATIO
VOL SPIKE
"""

#RSI, > 70 overbought, < 30 oversold
import pandas_ta
spy['RSI'] = pandas_ta.rsi(spy['Close'], length=14)
print(spy.columns)

#ZZZ, >2 overbought, < -2 oversold
rolling_mean = spy['Close'].rolling(window=20).mean()
rolling_std = spy['Close'].rolling(window=20).std()
spy['ZScore'] = (spy['Close'] - rolling_mean) / rolling_std

#volatility ratio, >1 short term > long term, potential trend
# < 1, short term < long term, potential reversal
spy['Vol_5d'] = spy['Log_Return'].rolling(5).std() * np.sqrt(252)
spy['Vol_20d'] = spy['Log_Return'].rolling(20).std() * np.sqrt(252)
spy['Vol_Ratio'] = spy['Vol_5d'] / spy['Vol_20d'].replace(0, 0.001)


# =============================
# Signal Generation

#Volatility Contraction
spy['Vol_Contract'] = (spy['Vol_Ratio'] < 0.8).astype(int)


#RSI
spy['Signal_RSI'] = ((spy['RSI'] < 30) | (spy['RSI'] > 70)).astype(int)
spy['RSI_dir'] = np.where(spy['RSI'] > 70, 1, 
                 np.where(spy['RSI'] < 30, -1, 0))
#BBand
spy['Signal_BBand'] = ((spy['Close'] <= spy['LBand']) | (spy['Close'] >= spy['UBand'])).astype(int)
spy['BBand_dir'] = np.where(spy['Close'] >= spy['UBand'], 1, 
                 np.where(spy['Close'] <= spy['LBand'], -1, 0))

#ADX to make sure we aint in high trending zone
spy.ta.adx(high=spy['High'], low=spy['Low'], close=spy['Close'], length=14, append=True)
spy['Signal_ADXWeakT'] = (spy['ADX_14'] < 25).astype(int)

# =============================
# Confidence weights
B = 0.4
R = 0.2
V = 0.15
A = 0.15
VS = 0.1

spy['Signal_Confidence'] = (spy['Vol_Contract'] * V + spy['Signal_RSI'] * R 
                            + spy['Signal_BBand'] * B + spy['Signal_ADXWeakT'] * A 
                            + spy['Vol_Contract'] * VS)

spy['Signal_Direction'] = np.where(spy['BBand_dir'] * spy['RSI_dir'] > 0, 1, -1)
print(spy.columns)

# =============================
# Backtest time

cash = 10000
shares = 0
portfolio = []
trades = []
count = 0
position = 0

risk_threshold = 0.05

for i in range(len(spy)):
    currentclose = spy['Close'].iloc[i]
    signalconf = spy['Signal_Confidence'].iloc[i]
    signaldir = spy['Signal_Direction'].iloc[i]
    midb = spy['MBand'].iloc[i]

    if shares == 0 and signalconf >= 0.6 and signaldir == 1:
        #Buy
        sharestobuy = cash*0.05 / currentclose
        shares += sharestobuy
        cash -= cash*0.05
        position = 1
    elif (shares > 0 and currentclose >= midb*0.98) or count > 3:
        #Sell
        cash += shares * currentclose
        shares = 0
        count = 0
        position = 0
    
    if position == 1:
        count += 1
    portfolio.append(cash + shares * currentclose)

casb = 10000
sharesb = 0
portfoliob = []
#buy and hold
for i in range(len(spy)):
    if i == 0:
        sharesb = casb / spy['Close'].iloc[i]
    portfoliob.append(sharesb * spy['Close'].iloc[i])
    casb = 0

plt.figure(figsize=(12,6))
plt.plot(spy.index, portfolio, label='Portfolio Value')
plt.plot(spy.index, portfoliob, label='Buy and Hold Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

print(f"Initial Portfolio Value: $10000")
print(f"Final Portfolio Value: ${portfolio[-1]:.2f}")
print(f"Final Buy and Hold Value: ${portfoliob[-1]:.2f}")

print(f"Total Return: {(portfolio[-1] - 10000) / 10000 * 100:.2f}%")
print(f"Buy and Hold Return: {(portfoliob[-1] - 10000) / 10000 * 100:.2f}%")