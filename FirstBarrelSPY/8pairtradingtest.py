#Im going to test with yf on pair trading, then i will transition to alpaca
import yfinance as yf
import numpy as np
import pandas as pd

df = yf.download(['SPY', 'QQQ'], period='10y', interval='1d')

df.columns = df.columns.get_level_values(0)

df['Close_Ratio'] = df['QQQ']['Close'] / df['SPY']['Close']

#z score on ratio
lookback = 20
df['ratio_z'] = (df['Close_Ratio'] - df['Close_Ratio'].rolling(lookback).mean()
                 )  / df['Close_Ratio'].rolling(lookback).std().replace(0, 0.001)

#rolling beta
#beta adjusted spread
returns_spy = df['SPY']['Close'].pct_change()
returns_qqq = df['QQQ']['Close'].pct_change()
df['beta'] = returns_qqq.rolling(60).cov(returns_spy) / returns_spy.rolling(60).var().replace(0, 0.001)
df['beta_adj_spread'] = df['QQQ']['Close'] - (df['beta'] * df['SPY']['Close'])

