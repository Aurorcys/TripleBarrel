import yfinance as yf
import numpy as np
import pandas as pd


spy = yf.download('SPY', period='5y', interval='1d')
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

#neccesary features
spy['Returns'] = spy['Close'].pct_change()
spy['MA_20'] = spy['Close'].rolling(window=20).mean()
spy['LBand'] = spy['MA_20'] - 2 * spy['Close'].rolling(window=20).std()

#signal
spy['VRatio'] = spy['Returns'].rolling(5).std() / spy['Returns'].rolling(20).std()
spy['Signal'] = ((spy['Close'] <= spy['LBand']) & (spy['VRatio'] < 0.8)).astype(int)

#backtest
cash = 10000
shares = 0
portfolio = []
trades = 0

risk_threshold = 0.1

for i in range(len(spy)):
    currentclose = spy['Close'].iloc[i]
    signal = spy['Signal'].iloc[i]
    sellsig = ((spy['Close'].iloc[i] >= spy['MA_20'].iloc[i]*1.01) & (shares > 0))
    if signal and shares == 0:
        sharestobuy = cash*risk_threshold / currentclose
        shares += sharestobuy
        cash -= cash*risk_threshold
        trades += 1
    elif sellsig:
        cash += shares * currentclose
        shares = 0
        trades += 1
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

print(f'Buy and Hold Portfolio Value: {portfoliob[-1]:.2f}')
print(f'Mean Reversion Strategy Portfolio Value: {portfolio[-1]:.2f}')

print(f'Returns: {((portfolio[-1] - 10000) / 10000) * 100:.2f}%')
print(f'Buy and Hold Returns: {((portfoliob[-1] - 10000) / 10000) * 100:.2f}%')