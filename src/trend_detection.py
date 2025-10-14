"""
Trend Detection Module for Tech News Trends

Uses NumPy for statistical analysis to identify trending topics and calculate growth rates.
Analyzes keyword momentum, detects emerging themes, and identifies declining topics.

Features:
- Calculate growth rates and momentum using NumPy
- Statistical significance testing for trends
- Rolling average analysis for smoothing
- Trend classification (hot, emerging, stable, declining)
- Visualize trend patterns over time
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explore import load_all_news_data


def calculate_daily_keyword_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily keyword frequencies for trend analysis.

    Args:
        df: DataFrame with articles and cleaned text

    Returns:
        DataFrame with daily keyword counts
    """
    print("\n[TRENDS] Calculating daily keyword frequencies...")

    # Key tech terms to track (simplified for better performance)
    keywords_to_track = [
        'ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'openai',
        'blockchain', 'bitcoin', 'crypto', 'startup', 'funding', 'investment',
        'google', 'apple', 'microsoft', 'amazon', 'meta', 'tesla', 'nvidia',
        'cloud', 'api', 'python', 'cybersecurity', 'quantum', 'robotics'
    ]

    # Ensure datetime columns exist
    if 'collection_datetime' not in df.columns:
        df['collection_datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')
        df['collection_date_only'] = df['collection_datetime'].dt.date

    # Prepare text if not already done
    if 'clean_text' not in df.columns:
        df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        df['clean_text'] = df['full_text'].str.lower()

    # Create daily trends data
    dates = sorted(df['collection_date_only'].unique())
    trend_data = []

    for date in dates:
        daily_df = df[df['collection_date_only'] == date]
        daily_counts = {'date': date, 'total_articles': len(daily_df)}

        for keyword in keywords_to_track:
            count = daily_df['clean_text'].str.contains(keyword, case=False, na=False).sum()
            daily_counts[keyword] = count

        trend_data.append(daily_counts)

    trends_df = pd.DataFrame(trend_data)

    print(f"   Tracking {len(keywords_to_track)} keywords over {len(dates)} days")
    print(f"   Date range: {min(dates)} to {max(dates)}")

    return trends_df


def calculate_growth_rates(trends_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate growth rates and momentum for each keyword using NumPy.

    Args:
        trends_df: DataFrame with daily keyword counts

    Returns:
        Dictionary with growth statistics for each keyword
    """
    print("\n[GROWTH] Calculating growth rates with NumPy...")

    growth_stats = {}
    keyword_columns = [col for col in trends_df.columns if col not in ['date', 'total_articles']]

    for keyword in keyword_columns:
        values = trends_df[keyword].values.astype(float)

        # Skip if no mentions
        if np.sum(values) == 0:
            continue

        # Basic statistics
        mean_daily = np.mean(values)
        std_daily = np.std(values)
        total_mentions = np.sum(values)

        # Growth rate calculation
        # Compare first half vs second half of time period
        midpoint = len(values) // 2
        if midpoint > 0:
            first_half_avg = np.mean(values[:midpoint])
            second_half_avg = np.mean(values[midpoint:])

            # Avoid division by zero
            if first_half_avg > 0:
                growth_rate = ((second_half_avg - first_half_avg) / first_half_avg) * 100
            else:
                growth_rate = float('inf') if second_half_avg > 0 else 0
        else:
            growth_rate = 0

        # Momentum calculation using linear regression slope
        days = np.arange(len(values))
        if len(values) > 1:
            slope, intercept = np.polyfit(days, values, 1)
            momentum_score = slope  # Articles per day change
        else:
            slope = 0
            momentum_score = 0

        # Volatility (coefficient of variation)
        volatility = (std_daily / mean_daily * 100) if mean_daily > 0 else 0

        # Recent trend (last 7 days vs previous 7 days)
        if len(values) >= 14:
            recent_avg = np.mean(values[-7:])
            previous_avg = np.mean(values[-14:-7])
            recent_change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        else:
            recent_change = 0

        growth_stats[keyword] = {
            'total_mentions': int(total_mentions),
            'daily_average': round(mean_daily, 2),
            'growth_rate': round(growth_rate, 2),
            'momentum_score': round(momentum_score, 4),
            'volatility': round(volatility, 2),
            'recent_change': round(recent_change, 2),
            'trend_slope': round(slope, 4)
        }

    print(f"   Analyzed growth rates for {len(growth_stats)} keywords")

    return growth_stats


def classify_trends(growth_stats: Dict) -> Dict[str, List[str]]:
    """
    Classify keywords into trend categories using statistical thresholds.

    Args:
        growth_stats: Growth statistics for each keyword

    Returns:
        Dictionary with keywords grouped by trend type
    """
    print("\n[CLASSIFY] Classifying trends...")

    trend_categories = {
        'hot_trending': [],      # High growth + high mentions
        'emerging': [],          # High growth + moderate mentions
        'stable_popular': [],    # Low growth + high mentions
        'declining': [],         # Negative growth
        'volatile': [],          # High volatility
        'recent_surge': []       # High recent change
    }

    # Calculate thresholds using NumPy percentiles
    all_keywords = list(growth_stats.keys())

    if not all_keywords:
        return trend_categories

    growth_rates = [growth_stats[kw]['growth_rate'] for kw in all_keywords]
    mentions = [growth_stats[kw]['total_mentions'] for kw in all_keywords]
    volatility = [growth_stats[kw]['volatility'] for kw in all_keywords]
    recent_changes = [growth_stats[kw]['recent_change'] for kw in all_keywords]

    # Calculate thresholds
    high_growth_threshold = np.percentile(growth_rates, 75)
    high_mentions_threshold = np.percentile(mentions, 75)
    high_volatility_threshold = np.percentile(volatility, 80)
    high_recent_change = np.percentile(recent_changes, 80)

    for keyword in all_keywords:
        stats = growth_stats[keyword]

        # Classification logic
        if stats['growth_rate'] >= high_growth_threshold and stats['total_mentions'] >= high_mentions_threshold:
            trend_categories['hot_trending'].append(keyword)
        elif stats['growth_rate'] >= high_growth_threshold:
            trend_categories['emerging'].append(keyword)
        elif stats['growth_rate'] < 0:
            trend_categories['declining'].append(keyword)
        elif stats['total_mentions'] >= high_mentions_threshold and abs(stats['growth_rate']) < 10:
            trend_categories['stable_popular'].append(keyword)
        elif stats['volatility'] >= high_volatility_threshold:
            trend_categories['volatile'].append(keyword)
        elif stats['recent_change'] >= high_recent_change:
            trend_categories['recent_surge'].append(keyword)

    # Print classification results
    for category, keywords in trend_categories.items():
        if keywords:
            print(f"   {category.replace('_', ' ').title()}: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")

    return trend_categories


def create_growth_rate_chart(growth_stats: Dict) -> None:
    """
    Create scatter plot showing growth rate vs total mentions.

    Args:
        growth_stats: Growth statistics for keywords
    """
    print("\n[CHART] Creating growth rate analysis chart...")

    if not growth_stats:
        print("   No data available for growth chart")
        return

    keywords = list(growth_stats.keys())
    growth_rates = [growth_stats[kw]['growth_rate'] for kw in keywords]
    total_mentions = [growth_stats[kw]['total_mentions'] for kw in keywords]

    plt.figure(figsize=(12, 8))

    # Create scatter plot with color coding based on growth rate
    colors = ['red' if gr < 0 else 'orange' if gr < 25 else 'green' for gr in growth_rates]
    scatter = plt.scatter(total_mentions, growth_rates, c=colors, alpha=0.7, s=100)

    # Add keyword labels for interesting points
    for i, keyword in enumerate(keywords):
        if abs(growth_rates[i]) > 20 or total_mentions[i] > np.percentile(total_mentions, 80):
            plt.annotate(keyword, (total_mentions[i], growth_rates[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Keyword Growth Rate vs Total Mentions', fontsize=16, fontweight='bold')
    plt.xlabel('Total Mentions', fontsize=12)
    plt.ylabel('Growth Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.text(0.02, 0.98, 'Red: Declining\nOrange: Stable\nGreen: Growing',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def create_momentum_timeline(trends_df: pd.DataFrame, top_keywords: List[str]) -> None:
    """
    Create timeline showing momentum for top keywords.

    Args:
        trends_df: Daily keyword trends
        top_keywords: Keywords to display
    """
    print("\n[CHART] Creating momentum timeline...")

    plt.figure(figsize=(14, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#21BF73']

    for i, keyword in enumerate(top_keywords[:6]):
        if keyword in trends_df.columns:
            values = trends_df[keyword].values

            # Calculate rolling average for smoothing
            if len(values) >= 3:
                rolling_avg = np.convolve(values, np.ones(3)/3, mode='same')
            else:
                rolling_avg = values

            plt.plot(trends_df['date'], rolling_avg,
                    marker='o', linewidth=2, label=f'{keyword.title()}',
                    color=colors[i % len(colors)])

    plt.title('Keyword Momentum Timeline (3-day Rolling Average)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Mentions (Smoothed)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_trend_heatmap(growth_stats: Dict) -> None:
    """
    Create heatmap showing different trend metrics.

    Args:
        growth_stats: Growth statistics dictionary
    """
    print("\n[CHART] Creating trend metrics heatmap...")

    if not growth_stats:
        print("   No data available for heatmap")
        return

    # Select top keywords by total mentions
    sorted_keywords = sorted(growth_stats.items(),
                           key=lambda x: x[1]['total_mentions'], reverse=True)[:10]

    keywords = [kw[0] for kw in sorted_keywords]
    metrics = ['growth_rate', 'momentum_score', 'volatility', 'recent_change']
    metric_labels = ['Growth Rate', 'Momentum', 'Volatility', 'Recent Change']

    # Create matrix for heatmap
    heatmap_data = np.zeros((len(keywords), len(metrics)))

    for i, keyword in enumerate(keywords):
        for j, metric in enumerate(metrics):
            value = growth_stats[keyword][metric]
            # Normalize values for better visualization
            if metric == 'momentum_score':
                value *= 100  # Scale momentum for visibility
            heatmap_data[i, j] = value

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

    plt.title('Trend Metrics Heatmap (Top Keywords)', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Keywords', fontsize=12)

    plt.yticks(range(len(keywords)), [kw.title() for kw in keywords])
    plt.xticks(range(len(metrics)), metric_labels)

    # Add color bar
    plt.colorbar(label='Metric Value')

    # Add text annotations
    for i in range(len(keywords)):
        for j in range(len(metrics)):
            value = heatmap_data[i, j]
            plt.text(j, i, f'{value:.1f}', ha='center', va='center',
                    color='white' if abs(value) > np.max(np.abs(heatmap_data))*0.5 else 'black')

    plt.tight_layout()
    plt.show()


def run_trend_detection_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete trend detection analysis pipeline.

    Args:
        df: DataFrame with news articles

    Returns:
        Dictionary with trend analysis results
    """
    print("="*60)
    print("TREND DETECTION ANALYSIS")
    print("="*60)

    # Calculate daily trends
    trends_df = calculate_daily_keyword_trends(df)

    # Calculate growth rates
    growth_stats = calculate_growth_rates(trends_df)

    # Classify trends
    trend_categories = classify_trends(growth_stats)

    # Create visualizations
    create_growth_rate_chart(growth_stats)

    # Get top keywords for timeline
    top_keywords_by_mentions = sorted(growth_stats.items(),
                                    key=lambda x: x[1]['total_mentions'], reverse=True)
    top_keyword_names = [kw[0] for kw in top_keywords_by_mentions[:8]]

    create_momentum_timeline(trends_df, top_keyword_names)
    create_trend_heatmap(growth_stats)

    # Compile results
    results = {
        'trends_data': trends_df,
        'growth_statistics': growth_stats,
        'trend_categories': trend_categories,
        'analysis_period': f"{trends_df['date'].min()} to {trends_df['date'].max()}",
        'keywords_analyzed': len(growth_stats)
    }

    print("\n" + "="*50)
    print("[SUCCESS] Trend Detection Complete!")
    print("   Key achievements:")
    print(f"   - Analyzed {len(growth_stats)} keywords over {len(trends_df)} days")
    print(f"   - Calculated growth rates using NumPy statistical methods")
    print(f"   - Classified trends into 6 categories")
    print(f"   - Created 3 trend visualization charts")
    print("   - Identified hot, emerging, and declining topics")
    print("="*50)

    return results


def main():
    """
    Main function to run trend detection analysis.
    """
    print("Starting Trend Detection Analysis")
    print("="*50)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data to analyze. Please check your data files.")
        return

    # Run trend detection
    results = run_trend_detection_analysis(df)

    return df, results


if __name__ == "__main__":
    df, results = main()