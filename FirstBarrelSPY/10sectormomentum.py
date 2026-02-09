import yfinance as yf
import numpy as np
import pandas as pd 
tickers = ['XLK', 'XLV', 'XLF']


xlk = yf.download(tickers[0], period='10y', interval='1d')
if isinstance(xlk.columns, pd.MultiIndex):
    xlk.columns = xlk.columns.get_level_values(0)
print(f'Shape: {xlk.shape}')
print(f'Columns: {xlk.columns.tolist()}')

xlv = yf.download(tickers[1], period='10y', interval='1d')
if isinstance(xlv.columns, pd.MultiIndex):
    xlv.columns = xlv.columns.get_level_values(0)
print(f'Shape: {xlv.shape}')
print(f'Columns: {xlv.columns.tolist()}')

xlf = yf.download(tickers[2], period='10y', interval='1d')
if isinstance(xlf.columns, pd.MultiIndex):
    xlf.columns = xlf.columns.get_level_values(0)
print(f'Shape: {xlf.shape}')
print(f'Columns: {xlf.columns.tolist()}')

#get monthly data
monthly_pricesxlk = xlk.resample('M').last()
monthly_pricesxlv = xlv.resample('M').last()
monthly_pricesxlf = xlf.resample('M').last()

#monthly returns
monthly_returnsxlk = monthly_pricesxlk.pct_change()
monthly_returnsxlv = monthly_pricesxlv.pct_change()
monthly_returnsxlf = monthly_pricesxlf.pct_change()

print(f'Monthly data: {len(monthly_pricesxlk)} months')
print(f'Monthly data: {len(monthly_pricesxlv)} months')
print(f'Monthly data: {len(monthly_pricesxlf)} months')

#calculate momentum
def calculate_momentum(prices, lookback=12):
    return prices.pct_change(lookback)
    

momentumxlk = calculate_momentum(monthly_pricesxlk)
momentumxlv = calculate_momentum(monthly_pricesxlv)
momentumxlf = calculate_momentum(monthly_pricesxlf)
print('Momentum calculated')
print(f'Momentum mean: {momentumxlk.mean()}')
print(f'Momentum mean: {momentumxlv.mean()}')
print(f'Momentum mean: {momentumxlf.mean()}')

#simple strategy
import numpy as np
import pandas as pd

# Better structure
tickers = ['XLK', 'XLV', 'XLF']
current_sector = None  # Store index 0, 1, or 2
shares = 0
cash = 10000
trades = []
portfolio_value = []

# Assuming you have these DataFrames
# monthly_pricesxlk, monthly_pricesxlv, monthly_pricesxlf
# cmomentumxlk, cmomentumxlv, cmomentumxlf

for i in range(len(monthly_pricesxlk)):  # All should have same length
    cmomentumxlk = calculate_momentum(monthly_pricesxlk)
    cmomentumxlv = calculate_momentum(monthly_pricesxlv)
    cmomentumxlf = calculate_momentum(monthly_pricesxlf)
    # Get current momentum values
    mom_values = [
        cmomentumxlk['Close'].iloc[i] if i < len(cmomentumxlk) else -np.inf,
        cmomentumxlv['Close'].iloc[i] if i < len(cmomentumxlv) else -np.inf,
        cmomentumxlf['Close'].iloc[i] if i < len(cmomentumxlf) else -np.inf
    ]
    
    # Find best sector (handle NaN/None)
    valid_mom = [0 if pd.isna(x) else x for x in mom_values]
    best_sector = np.argmax(valid_mom)
    
    # Check if we should switch sectors
    if current_sector != best_sector:
        # SELL current position
        if shares > 0 and current_sector is not None:
            # Get correct sell price
            if tickers[current_sector] == 'XLK':
                sell_price = monthly_pricesxlk['Close'].iloc[i]
            elif tickers[current_sector] == 'XLV':
                sell_price = monthly_pricesxlv['Close'].iloc[i]
            else:  # XLF
                sell_price = monthly_pricesxlf['Close'].iloc[i]
            
            cash += shares * sell_price
            
            trades.append({
                'action': 'sell',
                'sector': tickers[current_sector],
                'price': sell_price,
                'shares': shares,
                'date': monthly_pricesxlk.index[i]  # Add date
            })
            
            shares = 0
        
        # BUY new sector
        # Get correct buy price
        if tickers[best_sector] == 'XLK':
            buy_price = monthly_pricesxlk['Close'].iloc[i]
        elif tickers[best_sector] == 'XLV':
            buy_price = monthly_pricesxlv['Close'].iloc[i]
        else:  # XLF
            buy_price = monthly_pricesxlf['Close'].iloc[i]
        
        # Calculate shares (allow fractional)
        shares = cash / buy_price
        cash = 0
        
        trades.append({
            'action': 'buy',
            'sector': tickers[best_sector],
            'price': buy_price,
            'shares': shares,
            'date': monthly_pricesxlk.index[i]
        })
        
        current_sector = best_sector
    
    # Track portfolio value
    if shares > 0 and current_sector is not None:
        if tickers[current_sector] == 'XLK':
            current_price = monthly_pricesxlk['Close'].iloc[i]
        elif tickers[current_sector] == 'XLV':
            current_price = monthly_pricesxlv['Close'].iloc[i]
        else:
            current_price = monthly_pricesxlf['Close'].iloc[i]
        
        portfolio_value.append(cash + (shares * current_price))
    else:
        portfolio_value.append(cash)

print(f'Final portfolio value: {portfolio_value[-1] if portfolio_value else cash}')
print(f'Portfolio Return: {((portfolio_value[-1] - 10000) / 10000) * 100 if portfolio_value else 0}%')


#comparison with equal weight
cashb = 10000
sharesxlk = 0
sharesxlv = 0
sharesxlf = 0
portfoliob = []
for i in range(13, len(monthly_pricesxlk)):
    if i == 13:
        sharesxlk = cashb*0.3 / monthly_pricesxlk['Close'].iloc[i]
        sharesxlv = cashb*0.3 / monthly_pricesxlv['Close'].iloc[i]
        sharesxlf = cashb*0.3 / monthly_pricesxlf['Close'].iloc[i]
    portfoliob.append(sharesxlk*monthly_pricesxlk['Close'].iloc[i] + sharesxlv*monthly_pricesxlv['Close'].iloc[i] + sharesxlf*monthly_pricesxlf['Close'].iloc[i])
print(f'Final equal-weight portfolio value: {portfoliob[-1] if portfoliob else cashb}')
print(f'Equal-weight Portfolio Return: {((portfoliob[-1] - cashb) / cashb) * 100 if portfoliob else 0}%')