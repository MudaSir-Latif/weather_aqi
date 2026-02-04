"""
Exploratory Data Analysis for AQI Prediction System
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import OpenMeteoFetcher
from datetime import datetime, timedelta


def analyze_data(df: pd.DataFrame, output_dir: Path = None):
    """Perform exploratory data analysis"""
    
    if output_dir is None:
        output_dir = Path('data/analysis')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting EDA...")
    logger.info(f"Output directory: {output_dir}")
    
    # Basic info
    logger.info("\n=== Dataset Info ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
    logger.info(f"Number of days: {(df['time'].max() - df['time'].min()).days}")
    
    # Missing values
    logger.info("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    logger.info(f"\n{missing_df[missing_df['Missing Count'] > 0]}")
    
    # AQI statistics
    if 'aqi' in df.columns:
        logger.info("\n=== AQI Statistics ===")
        logger.info(f"\n{df['aqi'].describe()}")
        
        # AQI distribution plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(df['aqi'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('AQI')
        plt.ylabel('Frequency')
        plt.title('AQI Distribution')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        df['aqi'].plot(kind='box')
        plt.ylabel('AQI')
        plt.title('AQI Box Plot')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'aqi_distribution.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'aqi_distribution.png'}")
        plt.close()
        
        # AQI category distribution
        if 'aqi_category' in df.columns:
            logger.info("\n=== AQI Category Distribution ===")
            category_counts = df['aqi_category'].value_counts()
            logger.info(f"\n{category_counts}")
            
            plt.figure(figsize=(10, 6))
            category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.xlabel('AQI Category')
            plt.ylabel('Count')
            plt.title('Distribution of AQI Categories')
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(output_dir / 'aqi_categories.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_dir / 'aqi_categories.png'}")
            plt.close()
    
    # Pollutant statistics
    pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    if available_pollutants:
        logger.info("\n=== Pollutant Statistics ===")
        logger.info(f"\n{df[available_pollutants].describe()}")
        
        # Pollutant correlation matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = df[available_pollutants + ['aqi']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix: Pollutants and AQI')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'correlation_matrix.png'}")
        plt.close()
        
        # Time series of pollutants
        fig, axes = plt.subplots(len(available_pollutants), 1, 
                                 figsize=(14, 3 * len(available_pollutants)))
        
        if len(available_pollutants) == 1:
            axes = [axes]
        
        for idx, pollutant in enumerate(available_pollutants):
            axes[idx].plot(df['time'], df[pollutant], linewidth=0.5)
            axes[idx].set_ylabel(pollutant.upper())
            axes[idx].set_title(f'{pollutant.upper()} Over Time')
            axes[idx].grid(alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        plt.savefig(output_dir / 'pollutant_timeseries.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'pollutant_timeseries.png'}")
        plt.close()
    
    # Temporal patterns
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Hourly pattern
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        hourly_mean = df.groupby('hour')['aqi'].mean()
        plt.plot(hourly_mean.index, hourly_mean.values, marker='o')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average AQI')
        plt.title('AQI by Hour')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 2)
        daily_mean = df.groupby('day_of_week')['aqi'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(range(7), daily_mean.values, color='lightcoral', edgecolor='black')
        plt.xlabel('Day of Week')
        plt.ylabel('Average AQI')
        plt.title('AQI by Day of Week')
        plt.xticks(range(7), days)
        plt.grid(alpha=0.3, axis='y')
        
        plt.subplot(1, 3, 3)
        monthly_mean = df.groupby('month')['aqi'].mean()
        plt.bar(monthly_mean.index, monthly_mean.values, color='lightgreen', edgecolor='black')
        plt.xlabel('Month')
        plt.ylabel('Average AQI')
        plt.title('AQI by Month')
        plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'temporal_patterns.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_dir / 'temporal_patterns.png'}")
        plt.close()
    
    logger.info("\n=== EDA Complete ===")


def main():
    """Main EDA function"""
    parser = argparse.ArgumentParser(description='Perform EDA on AQI data')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input CSV file (if not provided, fetches fresh data)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days of data to analyze (if no input file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/analysis',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    # Load or fetch data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        df['time'] = pd.to_datetime(df['time'])
    else:
        logger.info(f"Fetching {args.days} days of data")
        fetcher = OpenMeteoFetcher()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        df = fetcher.fetch_air_quality_history(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            logger.error("No data fetched")
            sys.exit(1)
    
    # Perform analysis
    output_dir = Path(args.output_dir)
    analyze_data(df, output_dir)


if __name__ == "__main__":
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()
