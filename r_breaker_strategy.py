"""
R Breaker Strategy - Streamlined Version
Combines optimization and backtesting in one file
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use grayscale
plt.style.use('grayscale')

class RBreakerStrategy:
    """R Breaker Strategy - Integrated Version"""
    
    def __init__(self, data_path: str):
        """Initialize strategy"""
        self.data_path = data_path
        self.df = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load data"""
        df = pd.read_csv(self.data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Ensure price columns are numeric types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Add date column
        df['date'] = df.index.date
        
        return df
    
    def optimize_parameters(self, 
                          a_range=(0.3, 0.6, 0.05),
                          b_range=(0.4, 0.8, 0.05),
                          c_range=(0.05, 0.3, 0.05),
                          d_range=(0.2, 0.5, 0.05)) -> Dict:
        """
        Optimize parameters
        Returns best parameters and all test results
        """
        print("Starting parameter optimization...")
        
        # Generate parameter ranges
        param_ranges = {
            'a': np.arange(*a_range),
            'b': np.arange(*b_range),
            'c': np.arange(*c_range),
            'd': np.arange(*d_range)
        }
        
        # Generate all parameter combinations
        param_combinations = list(product(
            param_ranges['a'],
            param_ranges['b'],
            param_ranges['c'],
            param_ranges['d']
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        results = []
        best_sharpe = -np.inf
        best_params = None
        
        # Test each parameter combination
        for a, b, c, d in tqdm(param_combinations, desc="Parameter optimization progress"):
            metrics = self._backtest_with_params(a, b, c, d)
            
            result = {
                'a': round(a, 2),
                'b': round(b, 2),
                'c': round(c, 2),
                'd': round(d, 2),
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades']
            }
            results.append(result)
            
            # Update best parameters
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = result
        
        print(f"\nOptimization complete! Best Sharpe ratio: {best_sharpe:.3f}")
        
        return {
            'best_params': best_params,
            'all_results': pd.DataFrame(results)
        }
    
    def _backtest_with_params(self, a: float, b: float, c: float, d: float) -> Dict:
        """Backtest with specified parameters"""
        # Group by date
        daily_groups = self.df.groupby('date')
        
        # Initialize
        capital = 100000
        position = 0
        trades = []
        equity = []
        
        # Iterate through each day
        for date, daily_data in daily_groups:
            if len(daily_data) < 2:
                continue
                
            # Get previous day's data
            prev_day_data = self._get_previous_day_data(date, daily_groups)
            if prev_day_data is None:
                continue
            
            # Calculate R Breaker prices
            r_prices = self._calculate_r_breaker_prices(
                prev_day_data['high'], 
                prev_day_data['low'], 
                prev_day_data['close'],
                a, b, c, d
            )
            
            # Record daily high and low prices
            daily_high = 0
            daily_low = float('inf')
            hit_ssetup = False
            hit_bsetup = False
            
            # Iterate through daily candles
            for idx, (timestamp, row) in enumerate(daily_data.iterrows()):
                current_price = row['close']
                
                # Update daily high and low
                daily_high = max(daily_high, row['high'])
                daily_low = min(daily_low, row['low'])
                
                # Check if observation prices are triggered
                if row['high'] > r_prices['ssetup']:
                    hit_ssetup = True
                if row['low'] < r_prices['bsetup']:
                    hit_bsetup = True
                
                # Check if it's closing time
                is_close_time = idx >= len(daily_data) - 1
                
                # Generate trading signal
                signal = self._generate_signal(
                    position, current_price, r_prices, 
                    hit_ssetup, hit_bsetup, is_close_time
                )
                
                # Execute trade
                if signal != 0:
                    if position != 0:
                        # Close position
                        pnl = self._calculate_pnl(position, entry_price, current_price)
                        capital *= (1 + pnl)
                        trades.append({
                            'datetime': timestamp,
                            'type': 'close',
                            'price': current_price,
                            'pnl': pnl
                        })
                        position = 0
                    
                    # Open new position
                    if signal != 999 and not is_close_time:
                        position = signal
                        entry_price = current_price
                        trades.append({
                            'datetime': timestamp,
                            'type': 'open',
                            'price': current_price,
                            'pnl': 0
                        })
                
                # Record equity
                current_equity = capital
                if position != 0:
                    floating_pnl = self._calculate_pnl(position, entry_price, current_price)
                    current_equity = capital * (1 + floating_pnl)
                
                equity.append(current_equity)
        
        # Calculate metrics
        return self._calculate_metrics(equity, trades)
    
    def _calculate_r_breaker_prices(self, prev_high: float, prev_low: float, 
                                   prev_close: float, a: float, b: float, 
                                   c: float, d: float) -> Dict[str, float]:
        """Calculate R Breaker six key prices"""
        ssetup = prev_high + a * (prev_close - prev_low)
        bsetup = prev_low - a * (prev_high - prev_close)
        senter = b / 2 * (prev_high + prev_low) - c * prev_low
        benter = b / 2 * (prev_high + prev_low) - c * prev_high
        sbreak = ssetup - d * (ssetup - bsetup)
        bbreak = bsetup + d * (ssetup - bsetup)
        
        return {
            'ssetup': ssetup,
            'bsetup': bsetup,
            'senter': senter,
            'benter': benter,
            'sbreak': sbreak,
            'bbreak': bbreak
        }
    
    def _get_previous_day_data(self, current_date, daily_groups) -> Dict:
        """Get previous day's OHLC data"""
        dates = sorted(daily_groups.groups.keys())
        current_idx = dates.index(current_date)
        
        if current_idx == 0:
            return None
            
        prev_date = dates[current_idx - 1]
        prev_data = daily_groups.get_group(prev_date)
        
        return {
            'open': prev_data.iloc[0]['open'],
            'high': prev_data['high'].max(),
            'low': prev_data['low'].min(),
            'close': prev_data.iloc[-1]['close']
        }
    
    def _generate_signal(self, position: int, current_price: float, 
                        r_prices: Dict[str, float], hit_ssetup: bool, 
                        hit_bsetup: bool, is_close_time: bool) -> int:
        """Generate trading signal"""
        # Force close position before market close
        if is_close_time and position != 0:
            return 999
        
        # Signal when no position
        if position == 0:
            if current_price > r_prices['bbreak']:
                return 1
            elif current_price < r_prices['sbreak']:
                return -1
        
        # When holding long position
        elif position == 1:
            if hit_ssetup and current_price < r_prices['senter']:
                return -1
        
        # When holding short position
        elif position == -1:
            if hit_bsetup and current_price > r_prices['benter']:
                return 1
        
        return 0
    
    def _calculate_pnl(self, position: int, entry_price: float, 
                      exit_price: float) -> float:
        """Calculate profit/loss rate"""
        if position == 1:
            return (exit_price - entry_price) / entry_price
        elif position == -1:
            return (entry_price - exit_price) / entry_price
        return 0
    
    def _calculate_metrics(self, equity: List[float], trades: List[Dict]) -> Dict:
        """Calculate backtest metrics"""
        if not equity:
            return {}
        
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        
        # Total return
        total_return = (equity[-1] / equity[0] - 1) if equity[0] > 0 else 0
        
        # Sharpe ratio
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = len(trades_df[(trades_df['type'] == 'close') & (trades_df['pnl'] > 0)])
            total_closed = len(trades_df[trades_df['type'] == 'close'])
            win_rate = winning_trades / total_closed if total_closed > 0 else 0
            total_trades = len(trades_df) // 2
        else:
            win_rate = 0
            total_trades = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
    
    def run_full_analysis(self, 
                         a_range=(0.3, 0.6, 0.05),
                         b_range=(0.4, 0.8, 0.05),
                         c_range=(0.05, 0.3, 0.05),
                         d_range=(0.2, 0.5, 0.05)) -> None:
        """
        Run full analysis: optimize parameters -> detailed backtest -> generate report
        """
        print("="*60)
        print("R BREAKER STRATEGY ANALYSIS")
        print("="*60)
        
        # 1. Optimize parameters
        opt_result = self.optimize_parameters(a_range, b_range, c_range, d_range)
        best_params = opt_result['best_params']
        all_results = opt_result['all_results']
        
        # 2. Detailed backtest with best parameters
        print(f"\nRunning detailed backtest with best parameters...")
        detailed_result = self._detailed_backtest(
            best_params['a'], 
            best_params['b'], 
            best_params['c'], 
            best_params['d']
        )
        
        # 3. Generate report
        self._generate_report(best_params, all_results, detailed_result)
        
        print("\nAnalysis complete!")
        print(f"Best parameters: a={best_params['a']}, b={best_params['b']}, c={best_params['c']}, d={best_params['d']}")
        print(f"Total return: {best_params['total_return']:.2%}")
        print(f"Sharpe ratio: {best_params['sharpe_ratio']:.3f}")
        
    def _detailed_backtest(self, a: float, b: float, c: float, d: float) -> Dict:
        """Detailed backtest, returns all trade records and equity curve"""
        daily_groups = self.df.groupby('date')
        
        capital = 100000
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        for date, daily_data in daily_groups:
            if len(daily_data) < 2:
                continue
                
            prev_day_data = self._get_previous_day_data(date, daily_groups)
            if prev_day_data is None:
                continue
            
            r_prices = self._calculate_r_breaker_prices(
                prev_day_data['high'], 
                prev_day_data['low'], 
                prev_day_data['close'],
                a, b, c, d
            )
            
            daily_high = 0
            daily_low = float('inf')
            hit_ssetup = False
            hit_bsetup = False
            
            for idx, (timestamp, row) in enumerate(daily_data.iterrows()):
                current_price = row['close']
                
                daily_high = max(daily_high, row['high'])
                daily_low = min(daily_low, row['low'])
                
                if row['high'] > r_prices['ssetup']:
                    hit_ssetup = True
                if row['low'] < r_prices['bsetup']:
                    hit_bsetup = True
                
                is_close_time = idx >= len(daily_data) - 1
                
                signal = self._generate_signal(
                    position, current_price, r_prices, 
                    hit_ssetup, hit_bsetup, is_close_time
                )
                
                if signal != 0:
                    if position != 0:
                        pnl = self._calculate_pnl(position, entry_price, current_price)
                        capital *= (1 + pnl)
                        
                        trades.append({
                            'datetime': timestamp,
                            'type': 'close',
                            'position': position,
                            'price': current_price,
                            'pnl': pnl,
                            'capital': capital
                        })
                        
                        position = 0
                    
                    if signal != 999 and not is_close_time:
                        position = signal
                        entry_price = current_price
                        
                        trades.append({
                            'datetime': timestamp,
                            'type': 'open',
                            'position': position,
                            'price': current_price,
                            'pnl': 0,
                            'capital': capital
                        })
                
                current_equity = capital
                if position != 0:
                    floating_pnl = self._calculate_pnl(position, entry_price, current_price)
                    current_equity = capital * (1 + floating_pnl)
                
                equity_curve.append({
                    'datetime': timestamp,
                    'equity': current_equity,
                    'position': position
                })
        
        return {
            'trades': pd.DataFrame(trades),
            'equity_curve': pd.DataFrame(equity_curve)
        }
    
    def _generate_report(self, best_params: Dict, all_results: pd.DataFrame, 
                        detailed_result: Dict) -> None:
        """Generate comprehensive report"""
        
        # Create multi-page PDF
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages('r_breaker_report.pdf') as pdf:
            # Page 1: Main analysis
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('parameter-optimized-r-breaker-strategy', fontsize=16, fontweight='bold')
            # Add signature
            fig.text(0.5, 0.93, 'developed by Zhan Chen', ha='center', fontsize=12, style='italic')
            
            # 1. Best parameters
            ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2)
            ax1.text(0.5, 0.5, 
                    f"Best Parameters: a={best_params['a']}, b={best_params['b']}, "
                    f"c={best_params['c']}, d={best_params['d']}\n"
                    f"Return: {best_params['total_return']:.2%} | "
                    f"Sharpe: {best_params['sharpe_ratio']:.3f} | "
                    f"Max DD: {best_params['max_drawdown']:.2%} | "
                    f"Win Rate: {best_params['win_rate']:.2%}",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax1.axis('off')
            
            # 2. Equity curve
            ax2 = plt.subplot2grid((5, 2), (1, 0), colspan=2, rowspan=2)
            equity_curve = detailed_result['equity_curve']
            ax2.plot(equity_curve['equity'] / 100000, linewidth=1.5)
            ax2.set_ylabel('Capital (x100k)')
            ax2.set_title('Equity Curve')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('')
            
            # 3. Parameter distribution heatmap
            ax3 = plt.subplot2grid((5, 2), (3, 0))
            pivot = all_results.pivot_table(
                values='sharpe_ratio', 
                index='a', 
                columns='d', 
                aggfunc='mean'
            )
            im = ax3.imshow(pivot, cmap='gray', aspect='auto')
            ax3.set_xlabel('d')
            ax3.set_ylabel('a')
            ax3.set_title('Sharpe Ratio Heatmap (a vs d)')
            
            # 4. Return distribution
            ax4 = plt.subplot2grid((5, 2), (3, 1))
            ax4.hist(all_results['total_return'], bins=30, color='gray', edgecolor='black')
            ax4.axvline(x=best_params['total_return'], color='black', linestyle='--')
            ax4.set_xlabel('Total Return')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Return Distribution')
            
            # 5. Trading statistics
            ax5 = plt.subplot2grid((5, 2), (4, 0), colspan=2)
            trades_df = detailed_result['trades']
            if not trades_df.empty:
                stats_text = f"Total Trades: {len(trades_df)//2}\n"
                stats_text += f"Avg Trade Return: {trades_df[trades_df['type']=='close']['pnl'].mean():.3%}\n"
                stats_text += f"Best Trade: {trades_df[trades_df['type']=='close']['pnl'].max():.3%}\n"
                stats_text += f"Worst Trade: {trades_df[trades_df['type']=='close']['pnl'].min():.3%}"
            else:
                stats_text = "No trades executed"
            
            ax5.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10)
            ax5.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Trade records table (possibly multiple pages)
            if not trades_df.empty:
                # Prepare trade records data
                trades_display = trades_df[['datetime', 'type', 'position', 'price', 'pnl']].copy()
                trades_display['datetime'] = pd.to_datetime(trades_display['datetime']).dt.strftime('%Y-%m-%d %H:%M')
                trades_display['price'] = trades_display['price'].round(2)
                trades_display['pnl'] = (trades_display['pnl'] * 100).round(2)
                trades_display.loc[trades_display['type'] == 'open', 'pnl'] = '-'
                trades_display['position'] = trades_display['position'].map({1: 'Long', -1: 'Short', 0: '-'})
                
                # Trades per page
                trades_per_page = 40
                total_trades = len(trades_display)
                num_pages = (total_trades + trades_per_page - 1) // trades_per_page
                
                for page in range(num_pages):
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    
                    # Title
                    if num_pages > 1:
                        fig.suptitle(f'TRADE RECORDS (Page {page + 1} of {num_pages})', 
                                   fontsize=14, fontweight='bold')
                    else:
                        fig.suptitle('TRADE RECORDS', fontsize=14, fontweight='bold')
                    
                    # Get trades for current page
                    start_idx = page * trades_per_page
                    end_idx = min(start_idx + trades_per_page, total_trades)
                    page_trades = trades_display.iloc[start_idx:end_idx]
                    
                    # Create table
                    table_data = []
                    headers = ['DateTime', 'Type', 'Position', 'Price', 'PnL(%)']
                    
                    for _, row in page_trades.iterrows():
                        table_data.append([
                            row['datetime'],
                            row['type'].upper(),
                            row['position'],
                            f"${row['price']}",
                            f"{row['pnl']}%" if row['pnl'] != '-' else '-'
                        ])
                    
                    # Draw table
                    table = ax.table(cellText=table_data,
                                   colLabels=headers,
                                   cellLoc='center',
                                   loc='center',
                                   colWidths=[0.25, 0.12, 0.12, 0.12, 0.12])
                    
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.5)
                    
                    # Set table style
                    for i in range(len(headers)):
                        table[(0, i)].set_facecolor('#E0E0E0')
                        table[(0, i)].set_text_props(weight='bold')
                    
                    # Mark colors based on profit/loss
                    for i, row in enumerate(page_trades.itertuples(), 1):
                        if row.type == 'close':
                            if row.pnl != '-' and float(str(row.pnl).replace('%', '')) > 0:
                                table[(i, 4)].set_facecolor('#F0F0F0')
                            elif row.pnl != '-' and float(str(row.pnl).replace('%', '')) < 0:
                                table[(i, 4)].set_facecolor('#D0D0D0')
                    
                    ax.axis('off')
                    pdf.savefig(fig)
                    plt.close()
        
        print("\nFiles generated:")
        print("- r_breaker_report.pdf (Comprehensive report containing strategy analysis and trade records)")


def main():
    """Main function"""
    # Check if data file is provided
    if len(sys.argv) < 2:
        print("Error: No data file specified!")
        print("\nUsage: python r_breaker_strategy.py <data_file.csv>")
        print("\nExamples:")
        print("  python r_breaker_strategy.py TSLA_60d_5m.csv")
        print("  python r_breaker_strategy.py AAPL_60d_5m.csv")
        print("  python r_breaker_strategy.py MSFT_60d_5m.csv")
        print("\nData file must contain: datetime, open, high, low, close, volume columns")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        print(f"\nPlease make sure the file exists in the current directory.")
        sys.exit(1)
    
    print(f"Using data file: {data_file}")
    print("="*60)
    
    # Create strategy instance
    strategy = RBreakerStrategy(data_file)
    
    # Run full analysis
    strategy.run_full_analysis(
        a_range=(0.3, 0.6, 0.05),
        b_range=(0.4, 0.8, 0.05),
        c_range=(0.05, 0.3, 0.05),
        d_range=(0.2, 0.5, 0.05)
    )


if __name__ == "__main__":
    main()
