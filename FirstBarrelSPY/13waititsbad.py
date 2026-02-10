import yfinance as yf
import numpy as np
import pandas as pd 

xlk = yf.download('XLK', period='10y', interval='1d')
xlk_return = (xlk['Close'].iloc[-1] - xlk['Close'].iloc[0]) / xlk['Close'].iloc[0]
print(f'Return: {xlk_return}')

#XLK 7.26788
#damn man