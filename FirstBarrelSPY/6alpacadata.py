import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

symbol = 'SPY'

api = tradeapi.REST(
            API_KEY,
            SECRET_KEY,
            'https://paper-api.alpaca.markets'
        )

end_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)
start_date = end_date - pd.Timedelta(days=500)
print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
print(f"📅 Expected bars: ~{500 * 6.5} trading hours")
        
# Get bars (Alpaca excludes weekends/holidays automatically)
bars = api.get_bars(
    symbol,
    '1Hour',
    start=start_date.date().isoformat(),
    end=end_date.date().isoformat(),
    adjustment='raw'
    ).df
        
print(f"📊 Loaded {len(bars)} hourly bars for {symbol}")
print(f"Date range: {bars.index[0]} to {bars.index[-1]}")       


spy = bars.copy()

spy['Close'] = spy['close']

spy['Returns'] = spy['Close'].pct_change()
spy['MA_20'] = spy['Close'].rolling(20).mean()
spy['Std_20'] = spy['Close'].rolling(20).std()
spy['BB_Lower'] = spy['MA_20'] - (2 * spy['Std_20'])

# Step 3: ONE CLEAR ENTRY SIGNAL
# Buy when price hits lower BBand AND volatility is low
spy['Vol_Ratio'] = spy['Returns'].rolling(5).std() / spy['Returns'].rolling(20).std()
spy['Entry_Signal'] = ((spy['Close'] <= spy['BB_Lower']) & 
                       (spy['Vol_Ratio'] < 0.8)).astype(int)

# Step 4: CLEAR EXIT RULES
# Exit when price returns to 20-day MA OR after 5 days
spy['Position'] = 0
for i in range(20, len(spy)-5):
    if spy['Entry_Signal'].iloc[i] == 1 and spy['Position'].iloc[i-1] == 0:
        spy.loc[spy.index[i], 'Position'] = 1
        # Auto-exit after 5 days
        for j in range(1, 6):
            if i+j < len(spy):
                spy.loc[spy.index[i+j], 'Position'] = 1
                # Exit early if hits MA
                if spy['Close'].iloc[i+j] >= spy['MA_20'].iloc[i+j]:
                    break

# Step 5: CALCULATE RETURNS
spy['Strategy_Returns'] = spy['Position'] * spy['Returns']

# Calculate metrics
initial = 10000
spy['Strategy_Value'] = initial * (1 + spy['Strategy_Returns'].fillna(0)).cumprod()
spy['BuyHold_Value'] = initial * (1 + spy['Returns'].fillna(0)).cumprod()

# Key metrics
total_trades = spy['Entry_Signal'].sum()
winning_trades = len([i for i in range(len(spy)) 
                      if spy['Entry_Signal'].iloc[i] == 1 and 
                      spy['Strategy_Returns'].iloc[i+1:i+6].sum() > 0])

print(f"Total trades: {total_trades}")
print(f"Win rate: {winning_trades/total_trades:.1%}")
print(f"Strategy final: ${spy['Strategy_Value'].iloc[-1]:.2f}")
print(f"Buy & Hold final: ${spy['BuyHold_Value'].iloc[-1]:.2f}")

# Look at individual trades
for i in range(len(spy)):
    if spy['Entry_Signal'].iloc[i] == 1:
        entry_price = spy['Close'].iloc[i]
        exit_idx = min(i+5, len(spy)-1)
        exit_price = spy['Close'].iloc[exit_idx]
        pnl = (exit_price - entry_price) / entry_price * 100
        print(f"{spy.index[i].date()}: Entry ${entry_price:.2f}, Exit ${exit_price:.2f}, PnL: {pnl:.1f}%")