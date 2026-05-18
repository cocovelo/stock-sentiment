"""
Sentiment Comparison Analysis: FinBERT vs MarketAux

This script compares z-score normalized FinBERT sentiment scores and MarketAux 
sentiment scores against z-score normalized price changes across all indexes 
and time points to determine which sentiment source has stronger correlations.

Usage:
    conda activate ml
    python sentiment_comparison_analysis.py
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore


class SentimentComparisonAnalyzer:
    """Analyzes and compares FinBERT vs MarketAux sentiment correlations"""
    
    def __init__(self, credentials_path: str):
        """Initialize Firebase connection"""
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        
    def z_score_normalize(self, values: List[float]) -> Tuple[np.ndarray, float, float]:
        """
        Z-score normalize a list of values
        
        Returns:
            (normalized_values, mean, std_dev)
        """
        if len(values) == 0:
            return np.array([]), 0.0, 0.0
            
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)  # Sample standard deviation
        
        if std == 0:
            return np.zeros_like(arr), mean, 0.0
            
        normalized = (arr - mean) / std
        return normalized, mean, std
    
    def extract_stock_data(self, doc_data: Dict) -> List[Dict]:
        """
        Extract stock-level data from a Firestore document
        
        Returns list of dicts with:
        - symbol
        - finbert_sentiment (continuous)
        - marketaux_sentiment (average of article scores)
        - price_change_percent
        - index_name
        """
        results = []
        
        # The stock data is in index_top_stocks, not market_indexes
        index_top_stocks = doc_data.get('index_top_stocks', {})
        
        if not index_top_stocks:
            return results
        
        for index_symbol, index_data in index_top_stocks.items():
            index_name = index_data.get('index_name', index_symbol)
            stocks_data = index_data.get('stocks_data', [])
            
            for stock in stocks_data:
                symbol = stock.get('symbol')
                price_change_percent = stock.get('price_change_percent')
                finbert_sentiment = stock.get('sentiment_score_continuous')
                articles = stock.get('articles', [])
                
                # Skip if missing critical data
                if price_change_percent is None or finbert_sentiment is None:
                    continue
                
                # Calculate average MarketAux sentiment from articles
                marketaux_scores = []
                for article in articles:
                    score = article.get('marketaux_score')
                    if score is not None:
                        marketaux_scores.append(score)
                
                # Skip if no MarketAux data
                if not marketaux_scores:
                    continue
                    
                marketaux_sentiment = np.mean(marketaux_scores)
                
                results.append({
                    'symbol': symbol,
                    'index_name': index_name,
                    'index_symbol': index_symbol,
                    'finbert_sentiment': finbert_sentiment,
                    'marketaux_sentiment': marketaux_sentiment,
                    'price_change_percent': price_change_percent,
                    'num_articles': len(articles),
                    'num_marketaux_scores': len(marketaux_scores)
                })
        
        return results
    
    def fetch_all_timepoint_data(self) -> pd.DataFrame:
        """
        Fetch all sentiment_index_correlation data from Firestore
        
        Returns DataFrame with all stock-level observations
        """
        # Use the correct Firestore path structure
        collection_path = 'artifacts/client-project-viewer/public/data/sentiment_index_correlation'
        print(f"Fetching from collection: {collection_path}")
        
        collection_ref = self.db.collection(collection_path)
        docs = collection_ref.stream()
        
        all_data = []
        timepoint_count = 0
        
        for doc in docs:
            timepoint_count += 1
            doc_data = doc.to_dict()
            timepoint_id = doc.id
            
            if timepoint_count == 1:
                print(f"First document ID: {timepoint_id}")
                print(f"Document keys: {list(doc_data.keys())}")
                
                # Check for index_top_stocks
                if 'index_top_stocks' in doc_data:
                    index_top_stocks = doc_data['index_top_stocks']
                    print(f"✓ Found 'index_top_stocks' with {len(index_top_stocks)} indexes")
                    
                    if index_top_stocks:
                        first_index = list(index_top_stocks.keys())[0]
                        print(f"  First index: {first_index}")
                        first_index_data = index_top_stocks[first_index]
                        print(f"  Index data keys: {list(first_index_data.keys())}")
                        
                        stocks = first_index_data.get('stocks_data', [])
                        if stocks:
                            print(f"  First stock keys: {list(stocks[0].keys())}")
                            print(f"  Number of articles: {len(stocks[0].get('articles', []))}")
                        else:
                            print(f"  ⚠️ No stocks_data found in this index")
                else:
                    print("  ⚠️ No 'index_top_stocks' field in this document")
                    print("  Available fields suggest this is index-level data only, not stock-level")
            
            stock_data = self.extract_stock_data(doc_data)
            
            # Add timepoint ID to each record
            for record in stock_data:
                record['timepoint_id'] = timepoint_id
                all_data.append(record)
        
        print(f"Fetched {timepoint_count} timepoints")
        print(f"Extracted {len(all_data)} stock-level observations")
        
        return pd.DataFrame(all_data)
    
    def calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Calculate z-score normalized correlations for both sentiment sources
        
        Returns dict with correlation statistics
        """
        # Extract values
        finbert = df['finbert_sentiment'].values
        marketaux = df['marketaux_sentiment'].values
        price_changes = df['price_change_percent'].values
        
        # Z-score normalize
        finbert_z, finbert_mean, finbert_std = self.z_score_normalize(finbert)
        marketaux_z, marketaux_mean, marketaux_std = self.z_score_normalize(marketaux)
        price_z, price_mean, price_std = self.z_score_normalize(price_changes)
        
        # Calculate correlations (Pearson)
        finbert_corr, finbert_pval = stats.pearsonr(finbert_z, price_z)
        marketaux_corr, marketaux_pval = stats.pearsonr(marketaux_z, price_z)
        
        # Calculate R-squared
        finbert_r2 = finbert_corr ** 2
        marketaux_r2 = marketaux_corr ** 2
        
        # Spearman correlation (rank-based, robust to outliers)
        finbert_spearman, finbert_spearman_pval = stats.spearmanr(finbert, price_changes)
        marketaux_spearman, marketaux_spearman_pval = stats.spearmanr(marketaux, price_changes)
        
        return {
            'n_observations': len(df),
            'finbert': {
                'pearson_r': finbert_corr,
                'pearson_pval': finbert_pval,
                'r_squared': finbert_r2,
                'spearman_r': finbert_spearman,
                'spearman_pval': finbert_spearman_pval,
                'mean': finbert_mean,
                'std': finbert_std
            },
            'marketaux': {
                'pearson_r': marketaux_corr,
                'pearson_pval': marketaux_pval,
                'r_squared': marketaux_r2,
                'spearman_r': marketaux_spearman,
                'spearman_pval': marketaux_spearman_pval,
                'mean': marketaux_mean,
                'std': marketaux_std
            },
            'price_change': {
                'mean': price_mean,
                'std': price_std
            },
            'normalized_data': {
                'finbert_z': finbert_z,
                'marketaux_z': marketaux_z,
                'price_z': price_z
            }
        }
    
    def calculate_by_index_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations separately for each index"""
        results = []
        
        for index_name in df['index_name'].unique():
            index_df = df[df['index_name'] == index_name]
            
            if len(index_df) < 3:  # Need at least 3 points for correlation
                continue
            
            corr_stats = self.calculate_correlations(index_df)
            
            results.append({
                'index_name': index_name,
                'n_observations': corr_stats['n_observations'],
                'finbert_pearson_r': corr_stats['finbert']['pearson_r'],
                'finbert_r_squared': corr_stats['finbert']['r_squared'],
                'finbert_pval': corr_stats['finbert']['pearson_pval'],
                'marketaux_pearson_r': corr_stats['marketaux']['pearson_r'],
                'marketaux_r_squared': corr_stats['marketaux']['r_squared'],
                'marketaux_pval': corr_stats['marketaux']['pearson_pval'],
                'finbert_stronger': abs(corr_stats['finbert']['pearson_r']) > 
                                   abs(corr_stats['marketaux']['pearson_r'])
            })
        
        return pd.DataFrame(results)
    
    def create_comparison_plots(self, overall_stats: Dict, by_index_stats: pd.DataFrame, 
                                output_dir: str = 'output'):
        """Create visualization plots comparing both sentiment sources"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Overall Scatter Plot Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        finbert_z = overall_stats['normalized_data']['finbert_z']
        marketaux_z = overall_stats['normalized_data']['marketaux_z']
        price_z = overall_stats['normalized_data']['price_z']
        
        # FinBERT scatter
        axes[0].scatter(finbert_z, price_z, alpha=0.5, s=30, color='blue')
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        axes[0].set_xlabel('FinBERT Sentiment (z-score)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price Change % (z-score)', fontsize=12, fontweight='bold')
        axes[0].set_title(
            f"FinBERT vs Price Change\n"
            f"r = {overall_stats['finbert']['pearson_r']:.3f}, "
            f"R² = {overall_stats['finbert']['r_squared']:.3f}, "
            f"p = {overall_stats['finbert']['pearson_pval']:.4f}",
            fontsize=13, fontweight='bold'
        )
        axes[0].grid(True, alpha=0.3)
        
        # MarketAux scatter
        axes[1].scatter(marketaux_z, price_z, alpha=0.5, s=30, color='green')
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        axes[1].set_xlabel('MarketAux Sentiment (z-score)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Price Change % (z-score)', fontsize=12, fontweight='bold')
        axes[1].set_title(
            f"MarketAux vs Price Change\n"
            f"r = {overall_stats['marketaux']['pearson_r']:.3f}, "
            f"R² = {overall_stats['marketaux']['r_squared']:.3f}, "
            f"p = {overall_stats['marketaux']['pearson_pval']:.4f}",
            fontsize=13, fontweight='bold'
        )
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/overall_comparison_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. By-Index Comparison Bar Chart
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        indices = by_index_stats['index_name'].values
        x_pos = np.arange(len(indices))
        width = 0.35
        
        # Pearson correlation comparison
        axes[0].bar(x_pos - width/2, by_index_stats['finbert_pearson_r'].values, 
                   width, label='FinBERT', color='blue', alpha=0.7)
        axes[0].bar(x_pos + width/2, by_index_stats['marketaux_pearson_r'].values, 
                   width, label='MarketAux', color='green', alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0].set_xlabel('Index', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Pearson Correlation (r)', fontsize=12, fontweight='bold')
        axes[0].set_title('Correlation Comparison by Index', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(indices, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # R-squared comparison
        axes[1].bar(x_pos - width/2, by_index_stats['finbert_r_squared'].values, 
                   width, label='FinBERT', color='blue', alpha=0.7)
        axes[1].bar(x_pos + width/2, by_index_stats['marketaux_r_squared'].values, 
                   width, label='MarketAux', color='green', alpha=0.7)
        axes[1].set_xlabel('Index', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('R-squared', fontsize=12, fontweight='bold')
        axes[1].set_title('R² Comparison by Index', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(indices, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/by_index_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribution Comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].hist(finbert_z, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('z-score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('FinBERT Sentiment Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(marketaux_z, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('z-score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('MarketAux Sentiment Distribution', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(price_z, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('z-score', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title('Price Change % Distribution', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to '{output_dir}/' directory")
    
    def generate_report(self, df: pd.DataFrame, overall_stats: Dict, 
                       by_index_stats: pd.DataFrame, output_dir: str = 'output'):
        """Generate a detailed text report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = f'{output_dir}/sentiment_comparison_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SENTIMENT COMPARISON ANALYSIS: FinBERT vs MarketAux\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Observations: {overall_stats['n_observations']}\n")
            f.write(f"Unique Stocks: {df['symbol'].nunique()}\n")
            f.write(f"Unique Indexes: {df['index_name'].nunique()}\n")
            f.write(f"Unique Timepoints: {df['timepoint_id'].nunique()}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("OVERALL CORRELATION RESULTS (Z-Score Normalized)\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("FinBERT Sentiment:\n")
            f.write(f"  Pearson Correlation (r): {overall_stats['finbert']['pearson_r']:.4f}\n")
            f.write(f"  R-squared: {overall_stats['finbert']['r_squared']:.4f}\n")
            f.write(f"  P-value: {overall_stats['finbert']['pearson_pval']:.6f}\n")
            f.write(f"  Spearman Correlation: {overall_stats['finbert']['spearman_r']:.4f}\n")
            f.write(f"  Original Mean: {overall_stats['finbert']['mean']:.4f}\n")
            f.write(f"  Original Std Dev: {overall_stats['finbert']['std']:.4f}\n\n")
            
            f.write("MarketAux Sentiment:\n")
            f.write(f"  Pearson Correlation (r): {overall_stats['marketaux']['pearson_r']:.4f}\n")
            f.write(f"  R-squared: {overall_stats['marketaux']['r_squared']:.4f}\n")
            f.write(f"  P-value: {overall_stats['marketaux']['pearson_pval']:.6f}\n")
            f.write(f"  Spearman Correlation: {overall_stats['marketaux']['spearman_r']:.4f}\n")
            f.write(f"  Original Mean: {overall_stats['marketaux']['mean']:.4f}\n")
            f.write(f"  Original Std Dev: {overall_stats['marketaux']['std']:.4f}\n\n")
            
            f.write("Price Change %:\n")
            f.write(f"  Original Mean: {overall_stats['price_change']['mean']:.4f}%\n")
            f.write(f"  Original Std Dev: {overall_stats['price_change']['std']:.4f}%\n\n")
            
            # Determine winner
            finbert_r = abs(overall_stats['finbert']['pearson_r'])
            marketaux_r = abs(overall_stats['marketaux']['pearson_r'])
            
            f.write("-" * 80 + "\n")
            f.write("OVERALL WINNER\n")
            f.write("-" * 80 + "\n")
            if finbert_r > marketaux_r:
                diff = finbert_r - marketaux_r
                pct_stronger = (diff / marketaux_r * 100) if marketaux_r > 0 else float('inf')
                f.write(f"✓ FinBERT has STRONGER correlation: |r| = {finbert_r:.4f}\n")
                f.write(f"  MarketAux: |r| = {marketaux_r:.4f}\n")
                f.write(f"  Difference: {diff:.4f} ({pct_stronger:.1f}% stronger)\n\n")
            else:
                diff = marketaux_r - finbert_r
                pct_stronger = (diff / finbert_r * 100) if finbert_r > 0 else float('inf')
                f.write(f"✓ MarketAux has STRONGER correlation: |r| = {marketaux_r:.4f}\n")
                f.write(f"  FinBERT: |r| = {finbert_r:.4f}\n")
                f.write(f"  Difference: {diff:.4f} ({pct_stronger:.1f}% stronger)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("BY-INDEX BREAKDOWN\n")
            f.write("-" * 80 + "\n\n")
            
            by_index_sorted = by_index_stats.sort_values('n_observations', ascending=False)
            
            for _, row in by_index_sorted.iterrows():
                f.write(f"\n{row['index_name']}:\n")
                f.write(f"  Observations: {row['n_observations']}\n")
                f.write(f"  FinBERT   - r: {row['finbert_pearson_r']:7.4f}, R²: {row['finbert_r_squared']:.4f}, p: {row['finbert_pval']:.4f}\n")
                f.write(f"  MarketAux - r: {row['marketaux_pearson_r']:7.4f}, R²: {row['marketaux_r_squared']:.4f}, p: {row['marketaux_pval']:.4f}\n")
                f.write(f"  Winner: {'FinBERT' if row['finbert_stronger'] else 'MarketAux'}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            finbert_wins = by_index_stats['finbert_stronger'].sum()
            marketaux_wins = len(by_index_stats) - finbert_wins
            
            f.write(f"Indexes where FinBERT wins: {finbert_wins}/{len(by_index_stats)}\n")
            f.write(f"Indexes where MarketAux wins: {marketaux_wins}/{len(by_index_stats)}\n\n")
            
            f.write(f"Average |r| for FinBERT: {by_index_stats['finbert_pearson_r'].abs().mean():.4f}\n")
            f.write(f"Average |r| for MarketAux: {by_index_stats['marketaux_pearson_r'].abs().mean():.4f}\n\n")
            
            f.write(f"Average R² for FinBERT: {by_index_stats['finbert_r_squared'].mean():.4f}\n")
            f.write(f"Average R² for MarketAux: {by_index_stats['marketaux_r_squared'].mean():.4f}\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"\nDetailed report saved to: {report_path}")
    
    def run_complete_analysis(self, output_dir: str = 'output'):
        """Run the complete analysis pipeline"""
        print("=" * 80)
        print("SENTIMENT COMPARISON ANALYSIS: FinBERT vs MarketAux")
        print("=" * 80)
        print("\nStep 1: Fetching data from Firestore...")
        
        df = self.fetch_all_timepoint_data()
        
        if df.empty:
            print("ERROR: No data found!")
            return None, None, None
        
        print(f"\nStep 2: Calculating overall correlations...")
        overall_stats = self.calculate_correlations(df)
        
        print(f"\nStep 3: Calculating by-index correlations...")
        by_index_stats = self.calculate_by_index_correlations(df)
        
        print(f"\nStep 4: Creating visualizations...")
        self.create_comparison_plots(overall_stats, by_index_stats, output_dir)
        
        print(f"\nStep 5: Generating detailed report...")
        self.generate_report(df, overall_stats, by_index_stats, output_dir)
        
        # Print quick summary to console
        print("\n" + "=" * 80)
        print("QUICK SUMMARY")
        print("=" * 80)
        print(f"\nTotal Observations: {overall_stats['n_observations']}")
        print(f"\nFinBERT Correlation: r = {overall_stats['finbert']['pearson_r']:.4f}, R² = {overall_stats['finbert']['r_squared']:.4f}")
        print(f"MarketAux Correlation: r = {overall_stats['marketaux']['pearson_r']:.4f}, R² = {overall_stats['marketaux']['r_squared']:.4f}")
        
        finbert_r = abs(overall_stats['finbert']['pearson_r'])
        marketaux_r = abs(overall_stats['marketaux']['pearson_r'])
        
        if finbert_r > marketaux_r:
            print(f"\n✓ FinBERT has STRONGER overall correlation")
        else:
            print(f"\n✓ MarketAux has STRONGER overall correlation")
        
        print("\n" + "=" * 80)
        print(f"\nAll results saved to '{output_dir}/' directory")
        print("=" * 80)
        
        return df, overall_stats, by_index_stats


def main():
    """Main execution function"""
    # Path to Firebase credentials
    credentials_path = 'client-project-viewer-firebase-adminsdk-fbsvc-79862bf9e6.json'
    
    # Create output directory
    output_dir = 'sentiment_analysis_output'
    
    # Initialize analyzer
    print("\nInitializing Firebase connection...")
    analyzer = SentimentComparisonAnalyzer(credentials_path)
    
    # Run complete analysis
    result = analyzer.run_complete_analysis(output_dir)
    
    if result[0] is None:
        print("\nAnalysis could not be completed - no data available.")
        return
    
    df, overall_stats, by_index_stats = result
    
    # Optionally save raw data to CSV
    csv_path = f'{output_dir}/raw_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to: {csv_path}")


if __name__ == "__main__":
    main()
