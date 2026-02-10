import yfinance as yf
import numpy as np
import pandas as pd 

# Transaction cost and slippage parameters
TRANSACTION_COST = 0.001  # 0.1% per trade (both buy and sell)
SLIPPAGE = 0.0005  # 0.05% slippage per trade
tickers = ['XLK', 'XLV', 'XLF']

# Download data
print("📥 Downloading data...")
xlk = yf.download(tickers[0], period='10y', interval='1d', progress=False)
xlv = yf.download(tickers[1], period='10y', interval='1d', progress=False)
xlf = yf.download(tickers[2], period='10y', interval='1d', progress=False)

# Fix MultiIndex
for df in [xlk, xlv, xlf]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

# Monthly data
monthly_pricesxlk = xlk.resample('M').last()
monthly_pricesxlv = xlv.resample('M').last()
monthly_pricesxlf = xlf.resample('M').last()

# Momentum calculation
def calculate_momentum(prices, lookback=12):
    return prices.pct_change(lookback)

momentumxlk = calculate_momentum(monthly_pricesxlk)
momentumxlv = calculate_momentum(monthly_pricesxlv)
momentumxlf = calculate_momentum(monthly_pricesxlf)

print(f"✅ Data loaded: {len(monthly_pricesxlk)} months")

# ============================================================================
# STRATEGY WITH TRANSACTION COSTS AND SLIPPAGE
# ============================================================================
current_sector = None
shares = 0
cash = 10000
trades = []
portfolio_value = []
total_transaction_costs = 0
total_slippage = 0

print("\n💰 Running strategy with 0.1% transaction costs + 0.05% slippage...")

for i in range(len(monthly_pricesxlk)):
    # Recalculate momentum each time (inefficient but works)
    cmomentumxlk = calculate_momentum(monthly_pricesxlk)
    cmomentumxlv = calculate_momentum(monthly_pricesxlv)
    cmomentumxlf = calculate_momentum(monthly_pricesxlf)
    
    # Get momentum values
    mom_values = [
        cmomentumxlk['Close'].iloc[i] if i < len(cmomentumxlk) else -np.inf,
        cmomentumxlv['Close'].iloc[i] if i < len(cmomentumxlv) else -np.inf,
        cmomentumxlf['Close'].iloc[i] if i < len(cmomentumxlf) else -np.inf
    ]
    
    # Find best sector
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
            
            # Apply slippage on sale (worse execution)
            execution_price_sell = sell_price * (1 - SLIPPAGE)
            
            # Apply transaction cost on sale
            sale_value = shares * execution_price_sell
            transaction_cost_sell = sale_value * TRANSACTION_COST
            cash += sale_value - transaction_cost_sell
            total_transaction_costs += transaction_cost_sell
            
            # Track slippage impact
            slippage_cost_sell = shares * sell_price * SLIPPAGE
            total_slippage += slippage_cost_sell
            
            trades.append({
                'action': 'sell',
                'sector': tickers[current_sector],
                'quoted_price': sell_price,
                'execution_price': execution_price_sell,
                'shares': shares,
                'date': monthly_pricesxlk.index[i],
                'transaction_cost': transaction_cost_sell,
                'slippage_cost': slippage_cost_sell,
                'total_cost': transaction_cost_sell + slippage_cost_sell
            })
            
            shares = 0
        
        # BUY new sector
        if tickers[best_sector] == 'XLK':
            buy_price = monthly_pricesxlk['Close'].iloc[i]
        elif tickers[best_sector] == 'XLV':
            buy_price = monthly_pricesxlv['Close'].iloc[i]
        else:  # XLF
            buy_price = monthly_pricesxlf['Close'].iloc[i]
        
        # Apply slippage on purchase (worse execution)
        execution_price_buy = buy_price * (1 + SLIPPAGE)
        
        # Apply transaction cost on purchase
        purchase_value = cash
        transaction_cost_buy = purchase_value * TRANSACTION_COST
        net_cash_for_shares = purchase_value - transaction_cost_buy
        shares = net_cash_for_shares / execution_price_buy
        cash = 0
        total_transaction_costs += transaction_cost_buy
        
        # Track slippage impact
        slippage_cost_buy = shares * buy_price * SLIPPAGE
        total_slippage += slippage_cost_buy
        
        trades.append({
            'action': 'buy',
            'sector': tickers[best_sector],
            'quoted_price': buy_price,
            'execution_price': execution_price_buy,
            'shares': shares,
            'date': monthly_pricesxlk.index[i],
            'transaction_cost': transaction_cost_buy,
            'slippage_cost': slippage_cost_buy,
            'total_cost': transaction_cost_buy + slippage_cost_buy
        })
        
        current_sector = best_sector
    
    # Track portfolio value (using quoted prices for valuation)
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

# ============================================================================
# RESULTS WITH COSTS AND SLIPPAGE
# ============================================================================
print("\n" + "="*60)
print("📊 RESULTS WITH TRANSACTION COSTS + SLIPPAGE")
print("="*60)

final_value = portfolio_value[-1] if portfolio_value else cash
total_costs = total_transaction_costs + total_slippage

# Calculate returns
gross_return_pct = ((final_value + total_costs) / 10000 - 1) * 100
net_return_pct = (final_value / 10000 - 1) * 100
costs_reduction = gross_return_pct - net_return_pct

print(f"\n💰 FINAL PORTFOLIO VALUE: ${final_value:,.2f}")
print(f"📈 GROSS RETURN (before costs): {gross_return_pct:.1f}%")
print(f"📉 NET RETURN (after all costs): {net_return_pct:.1f}%")
print(f"\n💸 TOTAL TRANSACTION COSTS: ${total_transaction_costs:,.2f}")
print(f"📉 TOTAL SLIPPAGE COSTS: ${total_slippage:,.2f}")
print(f"💰 TOTAL IMPLEMENTATION COSTS: ${total_costs:,.2f}")
print(f"📉 COSTS REDUCED RETURNS BY: {costs_reduction:.1f}%")
print(f"🔄 TOTAL TRADES: {len(trades)}")

# ============================================================================
# EQUAL WEIGHT BENCHMARK WITH COSTS
# ============================================================================
print(f"\n" + "="*60)
print("📊 EQUAL WEIGHT BENCHMARK (with costs)")
print("="*60)

cashb = 10000
sharesxlk = 0
sharesxlv = 0
sharesxlf = 0
portfoliob = []


# Add transaction costs and slippage for benchmark
benchmark_transaction_cost = cashb * TRANSACTION_COST * 3  # Buy 3 ETFs
benchmark_slippage_cost = cashb * SLIPPAGE * 3  # Slippage on 3 purchases
cashb_net = cashb - benchmark_transaction_cost - benchmark_slippage_cost

for i in range(13, len(monthly_pricesxlk)):
    if i == 13:
        # Apply slippage to purchase prices
        price_xlk = monthly_pricesxlk['Close'].iloc[i] * (1 + SLIPPAGE)
        price_xlv = monthly_pricesxlv['Close'].iloc[i] * (1 + SLIPPAGE)
        price_xlf = monthly_pricesxlf['Close'].iloc[i] * (1 + SLIPPAGE)
        
        sharesxlk = (cashb_net / 3) / price_xlk
        sharesxlv = (cashb_net / 3) / price_xlv
        sharesxlf = (cashb_net / 3) / price_xlf
    portfoliob.append(
        sharesxlk * monthly_pricesxlk['Close'].iloc[i] + 
        sharesxlv * monthly_pricesxlv['Close'].iloc[i] + 
        sharesxlf * monthly_pricesxlf['Close'].iloc[i]
    )

benchmark_final = portfoliob[-1] if portfoliob else cashb_net
benchmark_return = (benchmark_final / 10000 - 1) * 100

print(f"\n🏆 FINAL BENCHMARK VALUE: ${benchmark_final:,.2f}")
print(f"📊 BENCHMARK RETURN (after costs): {benchmark_return:.1f}%")
print(f"🔥 STRATEGY VS BENCHMARK: {net_return_pct - benchmark_return:.1f}% alpha")

# ============================================================================
# TRADE ANALYSIS WITH SLIPPAGE
# ============================================================================
print(f"\n" + "="*60)
print("🔍 TRADE ANALYSIS WITH SLIPPAGE")
print("="*60)

if trades:
    # Count buys by sector
    sector_counts = {'XLK': 0, 'XLV': 0, 'XLF': 0}
    for trade in trades:
        if trade['action'] == 'buy':
            sector_counts[trade['sector']] += 1
    
    print(f"\n📈 SECTOR SELECTION FREQUENCY:")
    for sector, count in sector_counts.items():
        percentage = count / (len(trades)/2) * 100
        print(f"  {sector}: {count} times ({percentage:.0f}%)")
    
    # Calculate average costs
    avg_trade_cost = total_transaction_costs / (len(trades)/2) if len(trades) > 0 else 0
    avg_slippage_cost = total_slippage / (len(trades)/2) if len(trades) > 0 else 0
    avg_total_cost = total_costs / (len(trades)/2) if len(trades) > 0 else 0
    
    print(f"\n💸 AVERAGE COSTS PER ROUND TRIP:")
    print(f"  Transaction cost: ${avg_trade_cost:.2f}")
    print(f"  Slippage cost: ${avg_slippage_cost:.2f}")
    print(f"  Total cost: ${avg_total_cost:.2f}")
    
    # Calculate turnover
    total_turnover = len(trades) / 2
    print(f"\n🔄 ANNUAL TURNOVER: {total_turnover / 10:.0f}x per year")
    
    # Slippage analysis
    print(f"\n📉 SLIPPAGE IMPACT:")
    print(f"  Slippage as % of transaction costs: {(total_slippage/total_transaction_costs*100):.1f}%")
    print(f"  Slippage as % of total costs: {(total_slippage/total_costs*100):.1f}%")

# ============================================================================
# SLIPPAGE-SENSITIVE RECOMMENDATION
# ============================================================================
print(f"\n" + "="*60)
print("🎯 SLIPPAGE-SENSITIVE RECOMMENDATION")
print("="*60)

# Check if strategy still beats benchmark after all costs
if net_return_pct > benchmark_return:
    print(f"\n✅ STRATEGY STILL WORKS WITH COSTS + SLIPPAGE!")
    print(f"   Net alpha: {net_return_pct - benchmark_return:.1f}%")
    
    # Calculate if high turnover is worth it considering slippage
    alpha_per_trade = (net_return_pct - benchmark_return) / (len(trades)/2) if len(trades) > 0 else 0
    cost_per_trade = total_costs / (len(trades)/2) if len(trades) > 0 else 0
    
    print(f"\n📊 COST-BENEFIT ANALYSIS:")
    print(f"   Alpha per trade: {alpha_per_trade:.3f}%")
    print(f"   Cost per trade: ${cost_per_trade:.2f}")
    
    if alpha_per_trade > 0.15:  # Higher threshold due to slippage
        print(f"   📈 High turnover might still be justified")
    elif alpha_per_trade > 0.05:
        print(f"   ⚠️  Marginal benefit - consider reducing turnover")
        print(f"   🔄 Target: Reduce turnover by 50%")
    else:
        print(f"   ❌ Turnover too high - costs exceed benefits")
else:
    print(f"\n❌ STRATEGY FAILS WITH TRANSACTION COSTS + SLIPPAGE")
    print(f"   Benchmark beats strategy by {benchmark_return - net_return_pct:.1f}%")

# Slippage-specific suggestions
print(f"\n💡 SLIPPAGE-SPECIFIC SUGGESTIONS:")
print(f"   1. Use limit orders instead of market orders")
print(f"   2. Trade during high liquidity hours (10AM-2PM ET)")
print(f"   3. Break large orders into smaller chunks")
print(f"   4. Consider ETFs with higher AUM for better liquidity")
print(f"   5. Use volume-weighted average price (VWAP) orders")

# Revised strategy suggestions
print(f"\n📈 REVISED STRATEGY PARAMETERS:")
print(f"   Current: Monthly rebalancing, {len(trades)/2:.0f} round trips")
print(f"   Suggestion 1: Quarterly rebalancing (~{len(trades)/6:.0f} round trips)")
print(f"   Suggestion 2: 3-month momentum instead of 12-month")
print(f"   Suggestion 3: Add 1% momentum threshold to reduce whipsaws")

# Calculate break-even analysis
if len(trades) > 0:
    breakeven_turnover = (net_return_pct - benchmark_return) / 0.15  # Assuming 0.15% cost per trade
    print(f"\n💰 BREAK-EVEN ANALYSIS:")
    print(f"   To break even, need {breakeven_turnover:.1f} round trips")
    print(f"   Current: {len(trades)/2:.0f} round trips")
    
    if len(trades)/2 > breakeven_turnover:
        reduction_needed = ((len(trades)/2) - breakeven_turnover) / (len(trades)/2) * 100
        print(f"   Need to reduce turnover by {reduction_needed:.0f}%")

#print dates
for i in range(len(trades)):
    print(f"{trades[i]['date'].date()} - {trades[i]['action']} {trades[i]['shares']:.2f} {trades[i]['sector']} at ${trades[i]['execution_price']:.2f} (cost: ${trades[i]['total_cost']:.2f})")