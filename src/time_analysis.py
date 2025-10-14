"""
Time Trends Analysis Module for Tech News Trends

Analyzes temporal patterns in news data using pandas datetime operations.
Creates visualizations showing:
- Daily article volume trends
- Weekly patterns and seasonality
- Publication timing analysis
- Time-based comparisons across providers
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explore import load_all_news_data


def prepare_datetime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date strings to proper datetime objects and extract time components.

    Args:
        df: DataFrame with news articles

    Returns:
        DataFrame with additional datetime columns
    """
    print("\n[DATETIME] Preparing datetime data...")

    # Convert collection_date to datetime
    df['collection_datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')

    # Extract time components for analysis
    df['collection_date_only'] = df['collection_datetime'].dt.date
    df['collection_hour'] = df['collection_datetime'].dt.hour
    df['collection_day_of_week'] = df['collection_datetime'].dt.day_name()
    df['collection_week'] = df['collection_datetime'].dt.to_period('W')

    # Handle published_at if available (from some providers)
    if 'published_at' in df.columns and df['published_at'].notna().any():
        df['published_datetime'] = pd.to_datetime(df['published_at'], errors='coerce')
        df['published_hour'] = df['published_datetime'].dt.hour
        df['published_day_of_week'] = df['published_datetime'].dt.day_name()

    print(f"   Processed datetime for {len(df)} articles")
    print(f"   Date range: {df['collection_date_only'].min()} to {df['collection_date_only'].max()}")

    return df


def analyze_daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze daily article volume trends.

    Args:
        df: DataFrame with datetime columns

    Returns:
        DataFrame with daily aggregated data
    """
    print("\n[DAILY] Analyzing daily trends...")

    # Group by date and count articles
    daily_stats = df.groupby('collection_date_only').agg({
        'title': 'count',
        'data_provider': lambda x: x.value_counts().to_dict(),
        'collection_hour': 'mean'
    }).reset_index()

    daily_stats.columns = ['date', 'article_count', 'provider_breakdown', 'avg_collection_hour']

    # Calculate rolling averages
    daily_stats = daily_stats.sort_values('date')
    daily_stats['rolling_3day'] = daily_stats['article_count'].rolling(window=3, center=True).mean()
    daily_stats['rolling_7day'] = daily_stats['article_count'].rolling(window=7, center=True).mean()

    # Calculate day-over-day changes
    daily_stats['daily_change'] = daily_stats['article_count'].pct_change() * 100

    print(f"   Daily average: {daily_stats['article_count'].mean():.1f} articles")
    print(f"   Daily std dev: {daily_stats['article_count'].std():.1f} articles")
    print(f"   Max daily: {daily_stats['article_count'].max()} articles")
    print(f"   Min daily: {daily_stats['article_count'].min()} articles")

    return daily_stats


def analyze_weekly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze weekly patterns and day-of-week trends.

    Args:
        df: DataFrame with datetime columns

    Returns:
        DataFrame with weekly pattern analysis
    """
    print("\n[WEEKLY] Analyzing weekly patterns...")

    # Day of week analysis
    dow_stats = df.groupby('collection_day_of_week').agg({
        'title': 'count',
        'collection_hour': 'mean'
    }).reset_index()
    dow_stats.columns = ['day_of_week', 'article_count', 'avg_hour']

    # Reorder by actual day sequence
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats['day_num'] = dow_stats['day_of_week'].map({day: i for i, day in enumerate(day_order)})
    dow_stats = dow_stats.sort_values('day_num')

    # Weekly aggregation
    weekly_stats = df.groupby('collection_week').agg({
        'title': 'count',
        'data_provider': lambda x: x.value_counts().to_dict()
    }).reset_index()
    weekly_stats.columns = ['week', 'article_count', 'provider_breakdown']

    print(f"   Most active day: {dow_stats.loc[dow_stats['article_count'].idxmax(), 'day_of_week']}")
    print(f"   Least active day: {dow_stats.loc[dow_stats['article_count'].idxmin(), 'day_of_week']}")
    print(f"   Weekly average: {weekly_stats['article_count'].mean():.1f} articles")

    return dow_stats, weekly_stats


def create_daily_trend_chart(daily_stats: pd.DataFrame) -> None:
    """
    Create enhanced daily trend visualization.

    Args:
        daily_stats: Daily aggregated statistics
    """
    print("\n[CHART] Creating daily trend visualization...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main trend chart
    ax1.plot(daily_stats['date'], daily_stats['article_count'],
             marker='o', linewidth=2, markersize=5, color='#2E86AB', label='Daily Count')

    # Add rolling averages
    ax1.plot(daily_stats['date'], daily_stats['rolling_3day'],
             linewidth=2, color='#F18F01', alpha=0.8, label='3-day Moving Average')
    ax1.plot(daily_stats['date'], daily_stats['rolling_7day'],
             linewidth=2, color='#A23B72', alpha=0.8, label='7-day Moving Average')

    ax1.set_title('Daily Article Collection Trends', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Articles Collected', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Daily change chart
    colors = ['green' if x >= 0 else 'red' for x in daily_stats['daily_change']]
    ax2.bar(daily_stats['date'], daily_stats['daily_change'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Day-over-Day Change (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Change (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def create_weekly_pattern_chart(dow_stats: pd.DataFrame) -> None:
    """
    Create weekly pattern visualization.

    Args:
        dow_stats: Day of week statistics
    """
    print("\n[CHART] Creating weekly pattern visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Articles by day of week
    bars = ax1.bar(dow_stats['day_of_week'], dow_stats['article_count'],
                   color='#2E86AB', alpha=0.8)
    ax1.set_title('Articles by Day of Week', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Day of Week', fontsize=12)
    ax1.set_ylabel('Total Articles', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')

    # Average collection hour by day
    line = ax2.plot(dow_stats['day_of_week'], dow_stats['avg_hour'],
                    marker='o', linewidth=3, markersize=8, color='#F18F01')
    ax2.set_title('Average Collection Hour by Day', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('Hour (24h format)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Format y-axis to show hours
    ax2.set_ylim(0, 24)
    ax2.set_yticks(range(0, 25, 4))

    plt.tight_layout()
    plt.show()


def create_provider_timeline_comparison(df: pd.DataFrame) -> None:
    """
    Compare collection patterns between providers over time.

    Args:
        df: DataFrame with datetime columns
    """
    print("\n[CHART] Creating provider timeline comparison...")

    # Group by date and provider
    provider_daily = df.groupby(['collection_date_only', 'data_provider']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))

    # Plot each provider
    for provider in provider_daily.columns:
        plt.plot(provider_daily.index, provider_daily[provider],
                marker='o', linewidth=2, label=f'{provider.title()} ({provider_daily[provider].sum()} total)')

    plt.title('Article Collection by Provider Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Articles Collected', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print provider comparison stats
    print(f"   Provider totals: {dict(provider_daily.sum())}")
    print(f"   Days with both providers: {(provider_daily > 0).all(axis=1).sum()}")


def run_time_trends_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete time trends analysis.

    Args:
        df: DataFrame with news articles

    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("TIME TRENDS ANALYSIS")
    print("="*60)

    # Prepare datetime data
    df = prepare_datetime_data(df)

    # Analyze patterns
    daily_stats = analyze_daily_trends(df)
    dow_stats, weekly_stats = analyze_weekly_patterns(df)

    # Create visualizations
    create_daily_trend_chart(daily_stats)
    create_weekly_pattern_chart(dow_stats)
    create_provider_timeline_comparison(df)

    # Compile results
    results = {
        'daily_stats': daily_stats,
        'weekly_patterns': dow_stats,
        'weekly_totals': weekly_stats,
        'total_days': len(daily_stats),
        'avg_daily': daily_stats['article_count'].mean(),
        'date_range': (df['collection_date_only'].min(), df['collection_date_only'].max())
    }

    print("\n" + "="*50)
    print("[SUCCESS] Time Trends Analysis Complete!")
    print("   Key findings:")
    print(f"   - Analyzed {len(df)} articles over {results['total_days']} days")
    print(f"   - Average {results['avg_daily']:.1f} articles per day")
    print(f"   - Created 3 time-based visualizations")
    print("   - Identified daily and weekly patterns")
    print("="*50)

    return results


def main():
    """
    Main function to run time trends analysis.
    """
    print("Starting Time Trends Analysis")
    print("="*50)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data to analyze. Please check your data files.")
        return

    # Run analysis
    results = run_time_trends_analysis(df)

    return df, results


if __name__ == "__main__":
    df, results = main()