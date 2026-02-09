import yfinance as yf
import pandas as pd

# Test with one ETF
xlk = yf.download('XLK', period='1y', progress=False)['Close']
print(f"Daily days: {len(xlk)}")

# Resample
monthly = xlk.resample('M').last()
print(f"Monthly months: {len(monthly)}")
print(f"Expected: ~12 months (1 year)")

# Show the dates
print(f"\nMonthly dates:")
for date in monthly.index:
    print(f"  {date.date()}")
