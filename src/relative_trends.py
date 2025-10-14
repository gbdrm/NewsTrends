"""
Relative Trend Analysis Module for Tech News Trends

Analyzes keyword trends as percentages of daily articles to account for
changing data volume when new sources are added. Provides more accurate
trend detection by normalizing for article count changes.

Features:
- Calculate keyword frequency as percentage of daily articles
- Growth rate analysis using relative frequencies
- Compare pre/post data source addition periods
- Momentum analysis independent of volume changes
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explore import load_all_news_data


def calculate_relative_keyword_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily keyword frequencies as percentages of total daily articles.

    Args:
        df: DataFrame with articles and cleaned text

    Returns:
        DataFrame with relative keyword percentages
    """
    print("\n[RELATIVE] Calculating relative keyword frequencies...")

    # Key tech terms to track
    keywords_to_track = [
        'ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'openai',
        'blockchain', 'bitcoin', 'crypto', 'startup', 'funding', 'investment',
        'google', 'apple', 'microsoft', 'amazon', 'meta', 'tesla', 'nvidia',
        'cloud', 'api', 'cybersecurity', 'quantum', 'robotics'
    ]

    # Ensure datetime columns exist
    if 'collection_datetime' not in df.columns:
        df['collection_datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')
        df['collection_date_only'] = df['collection_datetime'].dt.date

    # Prepare text if not already done
    if 'clean_text' not in df.columns:
        df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
        df['clean_text'] = df['full_text'].str.lower()

    # Create relative trends data
    dates = sorted(df['collection_date_only'].unique())
    relative_data = []

    for date in dates:
        daily_df = df[df['collection_date_only'] == date]
        total_articles = len(daily_df)

        daily_entry = {
            'date': date,
            'total_articles': total_articles,
            'data_sources': daily_df['data_provider'].nunique()
        }

        # Calculate percentages for each keyword
        for keyword in keywords_to_track:
            count = daily_df['clean_text'].str.contains(keyword, case=False, na=False).sum()
            percentage = (count / total_articles * 100) if total_articles > 0 else 0

            daily_entry[f'{keyword}_count'] = count
            daily_entry[f'{keyword}_pct'] = round(percentage, 2)

        relative_data.append(daily_entry)

    trends_df = pd.DataFrame(relative_data)

    print(f"   Calculated relative frequencies for {len(keywords_to_track)} keywords")
    print(f"   Date range: {min(dates)} to {max(dates)}")
    print(f"   Articles per day range: {trends_df['total_articles'].min()}-{trends_df['total_articles'].max()}")

    return trends_df


def analyze_source_transition_impact(trends_df: pd.DataFrame) -> Dict:
    """
    Analyze the impact of adding NewsAPI source around Sep 28.

    Args:
        trends_df: DataFrame with relative keyword trends

    Returns:
        Dictionary with transition analysis
    """
    print("\n[TRANSITION] Analyzing source addition impact...")

    # Find the transition point (when daily articles significantly increased)
    trends_df_sorted = trends_df.sort_values('date')

    # Look for significant jump in article count
    trends_df_sorted['article_change'] = trends_df_sorted['total_articles'].pct_change()

    # Find the biggest jump (likely when NewsAPI was added)
    max_jump_idx = trends_df_sorted['article_change'].idxmax()

    if pd.notna(max_jump_idx):
        transition_date = trends_df_sorted.loc[max_jump_idx, 'date']

        # Split periods
        pre_transition = trends_df_sorted[trends_df_sorted['date'] < transition_date]
        post_transition = trends_df_sorted[trends_df_sorted['date'] >= transition_date]

        print(f"   Detected transition around: {transition_date}")
        print(f"   Pre-transition period: {len(pre_transition)} days, avg {pre_transition['total_articles'].mean():.1f} articles/day")
        print(f"   Post-transition period: {len(post_transition)} days, avg {post_transition['total_articles'].mean():.1f} articles/day")
    else:
        # Fallback: use date-based split around Sep 28
        transition_date = pd.to_datetime('2025-09-28').date()
        pre_transition = trends_df_sorted[trends_df_sorted['date'] < transition_date]
        post_transition = trends_df_sorted[trends_df_sorted['date'] >= transition_date]

    return {
        'transition_date': transition_date,
        'pre_transition_data': pre_transition,
        'post_transition_data': post_transition,
        'pre_avg_articles': pre_transition['total_articles'].mean(),
        'post_avg_articles': post_transition['total_articles'].mean(),
        'volume_increase': post_transition['total_articles'].mean() / pre_transition['total_articles'].mean()
    }


def calculate_relative_growth_rates(trends_df: pd.DataFrame, transition_info: Dict) -> Dict:
    """
    Calculate growth rates using relative percentages instead of raw counts.

    Args:
        trends_df: DataFrame with relative trends
        transition_info: Information about data source transition

    Returns:
        Dictionary with relative growth statistics
    """
    print("\n[GROWTH] Calculating percentage-based growth rates...")

    relative_growth_stats = {}

    # Get percentage columns
    pct_columns = [col for col in trends_df.columns if col.endswith('_pct')]
    keywords = [col.replace('_pct', '') for col in pct_columns]

    for keyword in keywords:
        pct_col = f'{keyword}_pct'
        values = trends_df[pct_col].values

        # Skip if no meaningful data
        if np.sum(values) == 0:
            continue

        # Basic statistics
        mean_pct = np.mean(values)
        std_pct = np.std(values)
        max_pct = np.max(values)

        # Calculate growth comparing periods
        pre_data = transition_info['pre_transition_data']
        post_data = transition_info['post_transition_data']

        if len(pre_data) > 0 and len(post_data) > 0 and pct_col in pre_data.columns:
            pre_avg_pct = pre_data[pct_col].mean()
            post_avg_pct = post_data[pct_col].mean()

            if pre_avg_pct > 0:
                relative_growth = ((post_avg_pct - pre_avg_pct) / pre_avg_pct) * 100
            else:
                relative_growth = float('inf') if post_avg_pct > 0 else 0
        else:
            relative_growth = 0

        # Momentum using linear regression on percentages
        days = np.arange(len(values))
        if len(values) > 1:
            slope, intercept = np.polyfit(days, values, 1)
            momentum_score = slope  # Percentage points per day
        else:
            slope = 0
            momentum_score = 0

        # Volatility of percentages
        volatility = (std_pct / mean_pct * 100) if mean_pct > 0 else 0

        relative_growth_stats[keyword] = {
            'avg_percentage': round(mean_pct, 3),
            'max_percentage': round(max_pct, 3),
            'relative_growth': round(relative_growth, 2),
            'momentum_pct_per_day': round(momentum_score, 4),
            'volatility': round(volatility, 2),
            'pre_period_avg': round(pre_data[pct_col].mean(), 3) if pct_col in pre_data.columns else 0,
            'post_period_avg': round(post_data[pct_col].mean(), 3) if pct_col in post_data.columns else 0
        }

    print(f"   Analyzed relative growth for {len(relative_growth_stats)} keywords")

    return relative_growth_stats


def create_volume_vs_percentage_chart(trends_df: pd.DataFrame) -> None:
    """
    Show how raw counts vs percentages tell different stories.

    Args:
        trends_df: DataFrame with both counts and percentages
    """
    print("\n[CHART] Creating volume vs percentage comparison...")

    # Select a popular keyword for demonstration
    keyword = 'ai'
    count_col = f'{keyword}_count'
    pct_col = f'{keyword}_pct'

    if count_col not in trends_df.columns:
        print(f"   Keyword '{keyword}' not found in data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Raw counts (potentially misleading due to volume change)
    ax1.plot(trends_df['date'], trends_df[count_col],
             marker='o', linewidth=2, color='#A23B72', label='Raw Count')
    ax1.set_title(f'{keyword.upper()} Raw Count Over Time (Misleading due to volume changes)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Daily Mentions Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add total article volume as background
    ax1_twin = ax1.twinx()
    ax1_twin.bar(trends_df['date'], trends_df['total_articles'],
                alpha=0.3, color='gray', label='Total Articles')
    ax1_twin.set_ylabel('Total Articles per Day', fontsize=12)

    # Percentage (true trend)
    ax2.plot(trends_df['date'], trends_df[pct_col],
             marker='o', linewidth=2, color='#2E86AB', label='Percentage')
    ax2.set_title(f'{keyword.upper()} Percentage of Daily Articles (True Trend)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Percentage of Daily Articles (%)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def create_relative_growth_chart(relative_growth_stats: Dict) -> None:
    """
    Create chart showing relative growth rates.

    Args:
        relative_growth_stats: Dictionary with relative growth data
    """
    print("\n[CHART] Creating relative growth analysis...")

    if not relative_growth_stats:
        print("   No data available for relative growth chart")
        return

    # Prepare data
    keywords = list(relative_growth_stats.keys())
    growth_rates = [relative_growth_stats[kw]['relative_growth'] for kw in keywords]
    avg_percentages = [relative_growth_stats[kw]['avg_percentage'] for kw in keywords]

    # Filter out infinite values for visualization
    finite_data = [(kw, gr, ap) for kw, gr, ap in zip(keywords, growth_rates, avg_percentages)
                   if not np.isinf(gr) and ap > 0.1]  # Only show keywords with >0.1% avg

    if not finite_data:
        print("   No meaningful data for visualization")
        return

    keywords_filtered, growth_filtered, avg_filtered = zip(*finite_data)

    plt.figure(figsize=(12, 8))

    # Color code based on growth
    colors = ['red' if gr < -20 else 'orange' if gr < 20 else 'green' for gr in growth_filtered]

    scatter = plt.scatter(avg_filtered, growth_filtered, c=colors, s=100, alpha=0.7)

    # Add labels for interesting points
    for i, keyword in enumerate(keywords_filtered):
        if abs(growth_filtered[i]) > 30 or avg_filtered[i] > 2:
            plt.annotate(keyword, (avg_filtered[i], growth_filtered[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Relative Keyword Growth (Pre vs Post Source Addition)', fontsize=16, fontweight='bold')
    plt.xlabel('Average Percentage of Daily Articles (%)', fontsize=12)
    plt.ylabel('Relative Growth Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.text(0.02, 0.98, 'Red: Declining >20%\nOrange: Stable ±20%\nGreen: Growing >20%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def run_relative_trend_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete relative trend analysis accounting for volume changes.

    Args:
        df: DataFrame with news articles

    Returns:
        Dictionary with relative analysis results
    """
    print("="*60)
    print("RELATIVE TREND ANALYSIS")
    print("="*60)

    # Calculate relative trends
    trends_df = calculate_relative_keyword_trends(df)

    # Analyze source transition impact
    transition_info = analyze_source_transition_impact(trends_df)

    # Calculate relative growth rates
    relative_growth_stats = calculate_relative_growth_rates(trends_df, transition_info)

    # Create visualizations
    create_volume_vs_percentage_chart(trends_df)
    create_relative_growth_chart(relative_growth_stats)

    # Print key insights
    print(f"\n[INSIGHTS] Key Relative Trend Findings:")
    print(f"   Volume increased {transition_info['volume_increase']:.1f}x after {transition_info['transition_date']}")

    # Top growing (relative)
    growing = [(k, v['relative_growth']) for k, v in relative_growth_stats.items()
               if not np.isinf(v['relative_growth']) and v['relative_growth'] > 20]
    growing.sort(key=lambda x: x[1], reverse=True)

    if growing:
        print("   Top growing topics (relative):")
        for keyword, growth in growing[:5]:
            print(f"     {keyword}: +{growth:.1f}% relative growth")

    # Top declining (relative)
    declining = [(k, v['relative_growth']) for k, v in relative_growth_stats.items()
                 if v['relative_growth'] < -20]
    declining.sort(key=lambda x: x[1])

    if declining:
        print("   Declining topics (relative):")
        for keyword, growth in declining[:3]:
            print(f"     {keyword}: {growth:.1f}% relative decline")

    results = {
        'relative_trends': trends_df,
        'transition_analysis': transition_info,
        'relative_growth_stats': relative_growth_stats,
        'growing_topics': growing,
        'declining_topics': declining
    }

    print("\n" + "="*50)
    print("[SUCCESS] Relative Trend Analysis Complete!")
    print("   Key achievements:")
    print(f"   - Normalized trends for volume changes ({transition_info['volume_increase']:.1f}x increase)")
    print(f"   - Calculated percentage-based growth rates")
    print(f"   - Identified {len(growing)} growing and {len(declining)} declining topics")
    print(f"   - Created volume vs percentage comparison charts")
    print("="*50)

    return results


def main():
    """
    Main function to run relative trend analysis.
    """
    print("Starting Relative Trend Analysis")
    print("="*50)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data to analyze. Please check your data files.")
        return

    # Run relative analysis
    results = run_relative_trend_analysis(df)

    return df, results


if __name__ == "__main__":
    df, results = main()