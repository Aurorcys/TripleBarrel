import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SectorMomentum:
    def __init__(self):
        # GICS Sector ETFs (SPDR)
        self.sectors = {
            'XLK': 'Technology',
            'XLV': 'Health Care', 
            'XLF': 'Financials',
            'XLI': 'Industrials',
            'XLE': 'Energy',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
        
        # Benchmark: Equal weight all sectors
        self.benchmark_tickers = list(self.sectors.keys())
        
    def fetch_data(self, start_date='2010-01-01', end_date=None):
        """Fetch historical data for all sector ETFs"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📥 Downloading sector ETF data ({len(self.sectors)} ETFs)...")
        print(f"Period: {start_date} to {end_date}")
        
        # Download all ETFs
        data = yf.download(
            list(self.sectors.keys()),
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )['Close']
        
        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        print(f"✅ Loaded {len(data)} trading days")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
    
    def calculate_momentum(self, prices, lookback_months=12, skip_month=True):
        """
        Calculate momentum using standard academic method
        Jegadeesh & Titman (1993) style
        """
        # Resample to monthly
        monthly_prices = prices.resample('M').last()
        
        # Calculate momentum: Price(t-1) / Price(t-13) - 1
        # Skip most recent month to avoid short-term reversal
        if skip_month:
            # t-2 to t-13 (skip most recent month)
            momentum = monthly_prices.shift(2) / monthly_prices.shift(14) - 1
        else:
            # t-1 to t-12
            momentum = monthly_prices.shift(1) / monthly_prices.shift(13) - 1
        
        return momentum
    
    def select_top_sectors(self, momentum_series, n=3):
        """Select top n sectors by momentum"""
        # Drop NaN values
        valid_momentum = momentum_series.dropna()
        
        if len(valid_momentum) == 0:
            return []
        
        # Get top n sectors
        top_sectors = valid_momentum.nlargest(n).index.tolist()
        return top_sectors
    
    def backtest(self, start_date='2010-01-01', initial_capital=10000):
        """Full backtest of sector momentum strategy"""
        print("\n" + "=" * 60)
        print("🚀 SECTOR MOMENTUM STRATEGY BACKTEST")
        print("=" * 60)
        
        # Fetch data
        prices = self.fetch_data(start_date=start_date)
        
        # Resample to monthly for rebalancing
        monthly_prices = prices.resample('M').last()
        monthly_returns = monthly_prices.pct_change()
        
        # Strategy parameters
        lookback_months = 12
        top_n = 3
        skip_recent_month = True
        
        # Initialize
        portfolio_value = initial_capital
        benchmark_value = initial_capital
        
        # Track holdings and values
        portfolio_history = []
        benchmark_history = []
        trades = []
        current_holdings = set()
        
        # Need at least 13 months for momentum calculation
        start_idx = 13
        
        print(f"\n📊 STRATEGY PARAMETERS:")
        print(f"Lookback: {lookback_months} months")
        print(f"Top sectors: {top_n}")
        print(f"Skip recent month: {skip_recent_month}")
        print(f"Rebalancing: Monthly")
        print(f"Initial capital: ${initial_capital:,}")
        
        for i in range(start_idx, len(monthly_prices)):
            current_date = monthly_prices.index[i]
            prev_date = monthly_prices.index[i-1]
            
            # Calculate momentum as of previous month
            momentum_data = self.calculate_momentum(
                monthly_prices.iloc[:i],  # Data up to previous month
                lookback_months=lookback_months,
                skip_month=skip_recent_month
            )
            
            # Get momentum for previous month
            if len(momentum_data) > 0:
                last_momentum = momentum_data.iloc[-1]
            else:
                last_momentum = pd.Series()
            
            # Monthly rebalancing
            if len(last_momentum) > 0:
                # Select top sectors
                top_sectors = self.select_top_sectors(last_momentum, n=top_n)
                
                # Check if we need to rebalance
                rebalance_needed = (
                    set(top_sectors) != current_holdings or
                    current_date.month != prev_date.month
                )
                
                if rebalance_needed and len(top_sectors) > 0:
                    # Calculate new position sizes (equal weight)
                    position_size = portfolio_value / len(top_sectors)
                    
                    # Log the trade
                    trade_info = {
                        'date': current_date,
                        'sectors': top_sectors,
                        'momentum_scores': {s: last_momentum[s] for s in top_sectors},
                        'prices': {s: monthly_prices[s].iloc[i] for s in top_sectors}
                    }
                    trades.append(trade_info)
                    
                    # Update holdings
                    current_holdings = set(top_sectors)
                
                # Calculate portfolio return for this month
                if current_holdings:
                    # Equal weight among selected sectors
                    sector_return = sum(monthly_returns[s].iloc[i] for s in current_holdings) / len(current_holdings)
                else:
                    sector_return = 0
            else:
                sector_return = 0
            
            # Update portfolio value
            portfolio_value *= (1 + sector_return)
            portfolio_history.append(portfolio_value)
            
            # Benchmark: Equal weight all sectors
            benchmark_return = monthly_returns.iloc[i].mean()
            benchmark_value *= (1 + benchmark_return)
            benchmark_history.append(benchmark_value)
        
        # Convert to Series
        portfolio_series = pd.Series(portfolio_history, index=monthly_prices.index[start_idx:])
        benchmark_series = pd.Series(benchmark_history, index=monthly_prices.index[start_idx:])
        
        # Calculate metrics
        print(f"\n" + "=" * 60)
        print("📈 PERFORMANCE RESULTS")
        print("=" * 60)
        
        # Basic returns
        total_return_pct = (portfolio_series.iloc[-1] / initial_capital - 1) * 100
        benchmark_return_pct = (benchmark_series.iloc[-1] / initial_capital - 1) * 100
        
        print(f"\n💰 FINAL VALUES:")
        print(f"Momentum Strategy: ${portfolio_series.iloc[-1]:,.2f}")
        print(f"Benchmark (Equal Weight): ${benchmark_series.iloc[-1]:,.2f}")
        print(f"Buy & Hold (SPY): ${self.calculate_spy_return(start_date, initial_capital):,.2f}")
        
        print(f"\n📊 TOTAL RETURNS:")
        print(f"Momentum Strategy: {total_return_pct:.1f}%")
        print(f"Benchmark (Equal Weight): {benchmark_return_pct:.1f}%")
        print(f"Strategy vs Benchmark: {total_return_pct - benchmark_return_pct:.1f}%")
        
        # Annualized metrics
        years = (portfolio_series.index[-1] - portfolio_series.index[0]).days / 365.25
        momentum_cagr = (portfolio_series.iloc[-1] / initial_capital) ** (1/years) - 1
        benchmark_cagr = (benchmark_series.iloc[-1] / initial_capital) ** (1/years) - 1
        
        print(f"\n📈 ANNUALIZED RETURNS (CAGR):")
        print(f"Momentum Strategy: {momentum_cagr*100:.1f}%")
        print(f"Benchmark: {benchmark_cagr*100:.1f}%")
        
        # Risk metrics
        monthly_returns_portfolio = portfolio_series.pct_change().dropna()
        monthly_returns_benchmark = benchmark_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 0% risk-free)
        sharpe_portfolio = np.sqrt(12) * monthly_returns_portfolio.mean() / monthly_returns_portfolio.std()
        sharpe_benchmark = np.sqrt(12) * monthly_returns_benchmark.mean() / monthly_returns_benchmark.std()
        
        print(f"\n⚖️ RISK-ADJUSTED RETURNS:")
        print(f"Momentum Sharpe: {sharpe_portfolio:.2f}")
        print(f"Benchmark Sharpe: {sharpe_benchmark:.2f}")
        
        # Drawdowns
        def max_drawdown(series):
            running_max = series.expanding().max()
            drawdown = (series - running_max) / running_max
            return drawdown.min() * 100
        
        max_dd_portfolio = max_drawdown(portfolio_series)
        max_dd_benchmark = max_drawdown(benchmark_series)
        
        print(f"\n📉 MAX DRAWDOWNS:")
        print(f"Momentum Strategy: {max_dd_portfolio:.1f}%")
        print(f"Benchmark: {max_dd_benchmark:.1f}%")
        
        # Trade statistics
        print(f"\n🔄 TRADING STATISTICS:")
        print(f"Total trades (rebalances): {len(trades)}")
        print(f"Average hold time: {years*12/len(trades):.1f} months")
        
        # Sector frequency
        sector_counts = {}
        for trade in trades:
            for sector in trade['sectors']:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        print(f"\n🏆 TOP SECTORS BY SELECTION FREQUENCY:")
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        for sector, count in sorted_sectors[:5]:
            print(f"  {sector} ({self.sectors[sector]}): {count} times ({count/len(trades)*100:.0f}%)")
        
        # Recent trades
        print(f"\n🔍 RECENT PORTFOLIO (Last 6 months):")
        recent_trades = trades[-6:] if len(trades) >= 6 else trades
        for trade in recent_trades:
            print(f"  {trade['date'].date()}: {', '.join(trade['sectors'])}")
            for sector in trade['sectors']:
                print(f"    {sector}: Momentum = {trade['momentum_scores'][sector]*100:.1f}%")
        
        # Current recommendation
        print(f"\n🎯 CURRENT RECOMMENDATION (Next month):")
        if len(trades) > 0:
            last_trade = trades[-1]
            print(f"Hold: {', '.join(last_trade['sectors'])}")
            print("Rebalance at start of next month")
        else:
            print("Insufficient data for recommendation")
        
        return {
            'portfolio': portfolio_series,
            'benchmark': benchmark_series,
            'trades': trades,
            'metrics': {
                'cagr': momentum_cagr,
                'sharpe': sharpe_portfolio,
                'max_drawdown': max_dd_portfolio,
                'total_trades': len(trades)
            }
        }
    
    def calculate_spy_return(self, start_date, initial_capital):
        """Calculate SPY buy-and-hold return for comparison"""
        spy = yf.download('SPY', start=start_date, progress=False, auto_adjust=True)['Close']
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        
        spy_close = spy['Close'] if 'Close' in spy.columns else spy.iloc[:, 3]
        spy_return = spy_close.iloc[-1] / spy_close.iloc[0]
        return float(initial_capital * spy_return)
    
    def run_parameter_sweep(self):
        """Test different parameter combinations"""
        print("\n" + "=" * 60)
        print("🔬 PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 60)
        
        parameters = [
            {'lookback': 6, 'top_n': 3, 'skip_month': True},
            {'lookback': 12, 'top_n': 3, 'skip_month': True},
            {'lookback': 12, 'top_n': 4, 'skip_month': True},
            {'lookback': 12, 'top_n': 3, 'skip_month': False},
            {'lookback': 3, 'top_n': 3, 'skip_month': True},
        ]
        
        results = []
        
        for params in parameters:
            print(f"\nTesting: {params}")
            
            # Simplified backtest for parameter sweep
            prices = self.fetch_data(start_date='2010-01-01')
            monthly_prices = prices.resample('M').last()
            monthly_returns = monthly_prices.pct_change()
            
            start_idx = max(13, params['lookback'] + 1)
            portfolio_value = 10000
            benchmark_value = 10000
            
            current_holdings = set()
            
            for i in range(start_idx, len(monthly_prices)):
                # Calculate momentum
                if params['skip_month']:
                    momentum = monthly_prices.shift(2) / monthly_prices.shift(2 + params['lookback']) - 1
                else:
                    momentum = monthly_prices.shift(1) / monthly_prices.shift(1 + params['lookback']) - 1
                
                if len(momentum) > 0:
                    last_momentum = momentum.iloc[-1]
                    top_sectors = self.select_top_sectors(last_momentum, n=params['top_n'])
                    
                    # Monthly rebalance
                    if monthly_prices.index[i].month != monthly_prices.index[i-1].month:
                        current_holdings = set(top_sectors)
                    
                    # Calculate returns
                    if current_holdings:
                        sector_return = sum(monthly_returns[s].iloc[i] for s in current_holdings) / len(current_holdings)
                    else:
                        sector_return = 0
                else:
                    sector_return = 0
                
                portfolio_value *= (1 + sector_return)
                benchmark_value *= (1 + monthly_returns.iloc[i].mean())
            
            final_return_pct = (portfolio_value / 10000 - 1) * 100
            benchmark_return_pct = (benchmark_value / 10000 - 1) * 100
            
            print(f"  Strategy: {final_return_pct:.1f}% | Benchmark: {benchmark_return_pct:.1f}% | Alpha: {final_return_pct - benchmark_return_pct:.1f}%")
            
            results.append({
                'params': params,
                'strategy_return': final_return_pct,
                'benchmark_return': benchmark_return_pct,
                'alpha': final_return_pct - benchmark_return_pct
            })
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['alpha'])
        print(f"\n🏆 BEST PARAMETERS: {best_result['params']}")
        print(f"Alpha: {best_result['alpha']:.1f}%")
        
        return results

# ============================================================================
# RUN THE STRATEGY
# ============================================================================

if __name__ == "__main__":
    strategy = SectorMomentum()
    
    # Run full backtest
    results = strategy.backtest(start_date='2010-01-01', initial_capital=10000)
    
    # Optional: Run parameter sweep
    run_sweep = input("\nRun parameter sensitivity analysis? (y/n): ").lower()
    if run_sweep == 'y':
        strategy.run_parameter_sweep()
    
    # Optional: Plot results
    plot_option = input("\nGenerate performance chart? (y/n): ").lower()
    if plot_option == 'y':
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Portfolio value
        ax1 = axes[0]
        ax1.plot(results['portfolio'].index, results['portfolio'].values, 
                label='Sector Momentum', linewidth=2, color='green')
        ax1.plot(results['benchmark'].index, results['benchmark'].values,
                label='Equal Weight Benchmark', alpha=0.7, color='blue')
        ax1.set_title('Sector Momentum Strategy vs Benchmark')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdowns
        ax2 = axes[1]
        
        # Calculate drawdowns
        def calculate_drawdown(series):
            running_max = series.expanding().max()
            return (series - running_max) / running_max * 100
        
        momentum_dd = calculate_drawdown(results['portfolio'])
        benchmark_dd = calculate_drawdown(results['benchmark'])
        
        ax2.fill_between(momentum_dd.index, momentum_dd.values, 0, 
                        alpha=0.5, color='red', label='Momentum Drawdown')
        ax2.fill_between(benchmark_dd.index, benchmark_dd.values, 0,
                        alpha=0.3, color='blue', label='Benchmark Drawdown')
        ax2.set_title('Drawdown Comparison')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY ANALYSIS COMPLETE")
    print("=" * 60)