"""
Source Analysis Module

This module analyzes different news sources to understand their:
- Publication patterns and frequency
- Content quality metrics
- Timing and consistency
- Comparative characteristics

Part of the NewsTrends project - demonstrating Pandas, NumPy, and Matplotlib skills
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

# Import our config and data loading utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.explore import load_all_news_data


def prepare_source_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for source analysis by cleaning and enriching.

    Args:
        df: Raw DataFrame with news articles

    Returns:
        DataFrame with prepared source data
    """
    print("\n=== Preparing Source Data ===")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Parse published_at dates (handle different date formats)
    if 'publishedAt' in df.columns:
        df['published_at'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    elif 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

    # Extract source name (different structure for different APIs)
    if 'source.name' in df.columns:
        df['source_name'] = df['source.name']
    elif 'source' in df.columns and df['source'].dtype == 'object':
        # Handle case where source might be a string already
        df['source_name'] = df['source'].apply(
            lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
        )
    else:
        df['source_name'] = 'Unknown'

    # Calculate content quality metrics
    df['title_length'] = df['title'].astype(str).str.len()
    df['description_length'] = df['description'].fillna('').astype(str).str.len()

    # Content completeness flags
    df['has_description'] = df['description'].notna() & (df['description_length'] > 0)
    df['has_url'] = df['url'].notna()
    df['has_image'] = df['urlToImage'].notna() if 'urlToImage' in df.columns else False

    # Time-based features
    if 'published_at' in df.columns:
        df['publish_hour'] = df['published_at'].dt.hour
        df['publish_day_of_week'] = df['published_at'].dt.dayofweek
        df['publish_date'] = df['published_at'].dt.date

    print(f"Prepared data with {len(df)} articles from {df['source_name'].nunique()} unique sources")

    return df


def analyze_source_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of articles across different sources.

    Args:
        df: DataFrame with prepared source data

    Returns:
        DataFrame with source distribution statistics
    """
    print("\n=== Source Distribution Analysis ===")

    # Count articles per source
    source_counts = df['source_name'].value_counts()

    # Calculate percentages
    source_percentages = (source_counts / len(df) * 100).round(2)

    # Create summary DataFrame
    source_summary = pd.DataFrame({
        'article_count': source_counts,
        'percentage': source_percentages,
        'provider': df.groupby('source_name')['data_provider'].first()
    })

    # Add ranking
    source_summary['rank'] = range(1, len(source_summary) + 1)

    print(f"\nTop 10 Sources by Article Count:")
    print(source_summary.head(10).to_string())

    print(f"\n[STATS] Source Distribution:")
    print(f"   Total unique sources: {len(source_summary)}")
    print(f"   Average articles per source: {source_summary['article_count'].mean():.1f}")
    print(f"   Median articles per source: {source_summary['article_count'].median():.0f}")
    print(f"   Top source coverage: {source_summary['percentage'].iloc[0]:.1f}%")

    return source_summary


def analyze_content_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze content quality metrics by source.

    Args:
        df: DataFrame with prepared source data

    Returns:
        DataFrame with quality metrics per source
    """
    print("\n=== Content Quality Analysis ===")

    # Group by source and calculate quality metrics
    quality_metrics = df.groupby('source_name').agg({
        'title_length': ['mean', 'std', 'min', 'max'],
        'description_length': ['mean', 'std', 'min', 'max'],
        'has_description': 'mean',  # Percentage with description
        'has_url': 'mean',  # Percentage with URL
        'has_image': 'mean',  # Percentage with image
        'title': 'count'  # Article count for filtering
    }).round(2)

    # Flatten column names
    quality_metrics.columns = ['_'.join(col).strip() for col in quality_metrics.columns.values]

    # Rename for clarity
    quality_metrics = quality_metrics.rename(columns={
        'title_count': 'article_count',
        'has_description_mean': 'description_rate',
        'has_url_mean': 'url_rate',
        'has_image_mean': 'image_rate'
    })

    # Calculate composite quality score (0-100)
    quality_metrics['quality_score'] = (
        (quality_metrics['description_rate'] * 40) +  # 40% weight
        (quality_metrics['url_rate'] * 20) +  # 20% weight
        (quality_metrics['image_rate'] * 20) +  # 20% weight
        (np.clip(quality_metrics['title_length_mean'] / 100, 0, 1) * 10) +  # 10% weight
        (np.clip(quality_metrics['description_length_mean'] / 200, 0, 1) * 10)  # 10% weight
    ).round(1)

    # Filter sources with at least 5 articles for meaningful stats
    quality_metrics = quality_metrics[quality_metrics['article_count'] >= 5].copy()

    # Sort by quality score
    quality_metrics = quality_metrics.sort_values('quality_score', ascending=False)

    print(f"\nTop 10 Sources by Quality Score:")
    print(quality_metrics[['article_count', 'quality_score', 'description_rate',
                           'title_length_mean', 'description_length_mean']].head(10).to_string())

    print(f"\n[STATS] Content Quality:")
    print(f"   Average quality score: {quality_metrics['quality_score'].mean():.1f}/100")
    print(f"   Sources with 100% descriptions: {(quality_metrics['description_rate'] == 1.0).sum()}")
    print(f"   Sources with 100% images: {(quality_metrics['image_rate'] == 1.0).sum()}")

    return quality_metrics


def analyze_publication_timing(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze publication timing patterns by source.

    Args:
        df: DataFrame with prepared source data

    Returns:
        Dictionary with timing analysis DataFrames
    """
    print("\n=== Publication Timing Analysis ===")

    # Filter to articles with valid timestamps
    df_timed = df[df['published_at'].notna()].copy()

    if len(df_timed) == 0:
        print("No articles with valid timestamps found")
        return {}

    # Hourly distribution by source
    hourly_patterns = df_timed.groupby(['source_name', 'publish_hour']).size().unstack(fill_value=0)

    # Daily distribution by source
    daily_patterns = df_timed.groupby(['source_name', 'publish_day_of_week']).size().unstack(fill_value=0)

    # Calculate publication frequency (articles per day)
    date_range = (df_timed['published_at'].max() - df_timed['published_at'].min()).days + 1
    pub_frequency = df_timed.groupby('source_name').agg({
        'published_at': 'count'
    }).rename(columns={'published_at': 'total_articles'})
    pub_frequency['days_active'] = date_range
    pub_frequency['articles_per_day'] = (pub_frequency['total_articles'] / pub_frequency['days_active']).round(2)

    # Find peak publication hours for each source
    peak_hours = hourly_patterns.idxmax(axis=1)

    # Calculate publication consistency (std dev of daily counts)
    daily_counts = df_timed.groupby(['source_name', 'publish_date']).size().reset_index(name='count')
    consistency = daily_counts.groupby('source_name')['count'].agg(['mean', 'std']).round(2)
    consistency['consistency_score'] = (100 - np.clip(consistency['std'] / consistency['mean'] * 100, 0, 100)).round(1)

    print(f"\n[STATS] Publication Timing:")
    print(f"   Date range analyzed: {date_range} days")
    print(f"   Average articles per day: {pub_frequency['articles_per_day'].mean():.2f}")

    print(f"\nTop 5 Most Frequent Publishers:")
    print(pub_frequency.nlargest(5, 'articles_per_day')[['total_articles', 'articles_per_day']].to_string())

    return {
        'hourly_patterns': hourly_patterns,
        'daily_patterns': daily_patterns,
        'frequency': pub_frequency,
        'peak_hours': peak_hours,
        'consistency': consistency
    }


def compare_sources(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Create comprehensive comparison of top N sources.

    Args:
        df: DataFrame with prepared source data
        top_n: Number of top sources to compare

    Returns:
        DataFrame with comprehensive source comparison
    """
    print(f"\n=== Comparing Top {top_n} Sources ===")

    # Get top sources by article count
    top_sources = df['source_name'].value_counts().head(top_n).index
    df_top = df[df['source_name'].isin(top_sources)].copy()

    # Aggregate multiple metrics
    comparison = df_top.groupby('source_name').agg({
        'title': 'count',
        'title_length': 'mean',
        'description_length': 'mean',
        'has_description': 'mean',
        'has_image': 'mean',
        'data_provider': 'first'
    }).round(2)

    comparison.columns = ['article_count', 'avg_title_length', 'avg_desc_length',
                          'description_rate', 'image_rate', 'provider']

    # Add percentage of total coverage
    comparison['coverage_pct'] = (comparison['article_count'] / len(df) * 100).round(2)

    # Sort by article count
    comparison = comparison.sort_values('article_count', ascending=False)

    print(f"\nComprehensive Source Comparison:")
    print(comparison.to_string())

    return comparison


def visualize_source_distribution(source_summary: pd.DataFrame, top_n: int = 15):
    """
    Create visualization of source distribution.

    Args:
        source_summary: DataFrame with source distribution data
        top_n: Number of top sources to display
    """
    print(f"\n=== Creating Source Distribution Visualization ===")

    # Get top N sources
    top_sources = source_summary.head(top_n)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_sources)))
    bars = ax1.barh(range(len(top_sources)), top_sources['article_count'], color=colors)
    ax1.set_yticks(range(len(top_sources)))
    ax1.set_yticklabels(top_sources.index)
    ax1.set_xlabel('Number of Articles', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} News Sources by Article Count', fontsize=13, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, top_sources['article_count'])):
        ax1.text(count + max(top_sources['article_count']) * 0.01, i,
                f'{count} ({top_sources["percentage"].iloc[i]}%)',
                va='center', fontsize=9)

    # Right: Pie chart showing concentration
    # Top 5, next 10, and rest
    top5_sum = source_summary.head(5)['article_count'].sum()
    next10_sum = source_summary.iloc[5:15]['article_count'].sum()
    rest_sum = source_summary.iloc[15:]['article_count'].sum()

    sizes = [top5_sum, next10_sum, rest_sum]
    labels = [f'Top 5 Sources\n({top5_sum} articles)',
              f'Next 10 Sources\n({next10_sum} articles)',
              f'Remaining Sources\n({rest_sum} articles)']
    colors_pie = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 10})
    ax2.set_title('Source Concentration Distribution', fontsize=13, fontweight='bold', pad=15)

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    plt.tight_layout()

    # Save figure
    output_path = 'visuals/Source_Distribution.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.close()


def visualize_quality_comparison(quality_metrics: pd.DataFrame, top_n: int = 12):
    """
    Create visualization comparing content quality across sources.

    Args:
        quality_metrics: DataFrame with quality metrics
        top_n: Number of sources to display
    """
    print(f"\n=== Creating Quality Comparison Visualization ===")

    # Get top N by quality score
    top_quality = quality_metrics.nlargest(top_n, 'quality_score')

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Quality Score comparison
    colors_score = plt.cm.RdYlGn(top_quality['quality_score'] / 100)
    bars1 = ax1.barh(range(len(top_quality)), top_quality['quality_score'], color=colors_score)
    ax1.set_yticks(range(len(top_quality)))
    ax1.set_yticklabels(top_quality.index, fontsize=9)
    ax1.set_xlabel('Quality Score (0-100)', fontsize=10, fontweight='bold')
    ax1.set_title('Content Quality Score by Source', fontsize=12, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 105)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars1, top_quality['quality_score'])):
        ax1.text(score + 1, i, f'{score:.1f}', va='center', fontsize=8, fontweight='bold')

    # 2. Content completeness rates
    x = np.arange(len(top_quality))
    width = 0.25

    bars_desc = ax2.bar(x - width, top_quality['description_rate'] * 100, width,
                        label='Has Description', color='#3498db', alpha=0.8)
    bars_url = ax2.bar(x, top_quality['url_rate'] * 100, width,
                       label='Has URL', color='#2ecc71', alpha=0.8)
    bars_img = ax2.bar(x + width, top_quality['image_rate'] * 100, width,
                       label='Has Image', color='#e74c3c', alpha=0.8)

    ax2.set_ylabel('Completeness Rate (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Content Completeness Rates', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_quality.index, rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 110)

    # 3. Title vs Description Length
    scatter = ax3.scatter(top_quality['title_length_mean'],
                         top_quality['description_length_mean'],
                         s=top_quality['article_count'] * 3,
                         c=top_quality['quality_score'],
                         cmap='viridis',
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=1)

    # Add source labels
    for idx, row in top_quality.iterrows():
        ax3.annotate(idx[:15],
                    (row['title_length_mean'], row['description_length_mean']),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

    ax3.set_xlabel('Average Title Length (chars)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Average Description Length (chars)', fontsize=10, fontweight='bold')
    ax3.set_title('Title vs Description Length\n(bubble size = article count)',
                  fontsize=12, fontweight='bold', pad=10)
    ax3.grid(alpha=0.3, linestyle='--')

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Quality Score', fontsize=9, fontweight='bold')

    # 4. Article count vs Quality
    colors_scatter = plt.cm.viridis(top_quality['quality_score'] / 100)
    ax4.scatter(top_quality['article_count'], top_quality['quality_score'],
               s=200, c=colors_scatter, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add source labels
    for idx, row in top_quality.iterrows():
        ax4.annotate(idx[:15],
                    (row['article_count'], row['quality_score']),
                    fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')

    ax4.set_xlabel('Number of Articles', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Quality Score', fontsize=10, fontweight='bold')
    ax4.set_title('Article Volume vs Content Quality', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save figure
    output_path = 'visuals/Source_Quality_Comparison.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.close()


def visualize_publication_timing(timing_data: Dict, top_n: int = 10):
    """
    Create visualization of publication timing patterns.

    Args:
        timing_data: Dictionary with timing analysis data
        top_n: Number of sources to display
    """
    print(f"\n=== Creating Publication Timing Visualization ===")

    if not timing_data or 'hourly_patterns' not in timing_data:
        print("No timing data available for visualization")
        return

    hourly = timing_data['hourly_patterns']
    frequency = timing_data['frequency']

    # Get top sources by frequency
    top_sources = frequency.nlargest(top_n, 'articles_per_day').index
    hourly_top = hourly.loc[hourly.index.isin(top_sources)]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Heatmap of hourly patterns
    # Normalize each row to show relative patterns
    hourly_normalized = hourly_top.div(hourly_top.sum(axis=1), axis=0) * 100

    im = ax1.imshow(hourly_normalized.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24), fontsize=9)
    ax1.set_yticks(range(len(hourly_normalized)))
    ax1.set_yticklabels(hourly_normalized.index, fontsize=9)
    ax1.set_xlabel('Hour of Day (UTC)', fontsize=10, fontweight='bold')
    ax1.set_title(f'Publication Timing Heatmap - Top {top_n} Sources\n(% of daily articles)',
                  fontsize=12, fontweight='bold', pad=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('% of Daily Articles', fontsize=9, fontweight='bold')

    # 2. Articles per day comparison
    top_freq = frequency.nlargest(top_n, 'articles_per_day')
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_freq)))

    bars = ax2.barh(range(len(top_freq)), top_freq['articles_per_day'], color=colors)
    ax2.set_yticks(range(len(top_freq)))
    ax2.set_yticklabels(top_freq.index, fontsize=9)
    ax2.set_xlabel('Articles per Day', fontsize=10, fontweight='bold')
    ax2.set_title(f'Publication Frequency - Top {top_n} Sources',
                  fontsize=12, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_freq['articles_per_day'])):
        ax2.text(val + max(top_freq['articles_per_day']) * 0.01, i,
                f'{val:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = 'visuals/Source_Publication_Timing.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.close()


def main():
    """
    Main function to run complete source analysis.
    """
    print("="*70)
    print(" "*15 + "SOURCE ANALYSIS - NEWS TRENDS")
    print("="*70)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data available for analysis")
        return

    # Prepare data
    df = prepare_source_data(df)

    # Run analyses
    print("\n" + "="*70)
    print("STEP 1: SOURCE DISTRIBUTION")
    print("="*70)
    source_summary = analyze_source_distribution(df)

    print("\n" + "="*70)
    print("STEP 2: CONTENT QUALITY")
    print("="*70)
    quality_metrics = analyze_content_quality(df)

    print("\n" + "="*70)
    print("STEP 3: PUBLICATION TIMING")
    print("="*70)
    timing_data = analyze_publication_timing(df)

    print("\n" + "="*70)
    print("STEP 4: SOURCE COMPARISON")
    print("="*70)
    comparison = compare_sources(df, top_n=10)

    # Create visualizations
    print("\n" + "="*70)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*70)

    visualize_source_distribution(source_summary, top_n=15)
    visualize_quality_comparison(quality_metrics, top_n=12)
    visualize_publication_timing(timing_data, top_n=10)

    print("\n" + "="*70)
    print("[SUCCESS] Source Analysis Complete!")
    print("="*70)
    print("\nKey Findings:")
    print(f"   - Analyzed {len(df)} articles from {df['source_name'].nunique()} unique sources")
    print(f"   - Top source coverage: {source_summary['percentage'].iloc[0]:.1f}%")
    print(f"   - Average quality score: {quality_metrics['quality_score'].mean():.1f}/100")
    print(f"   - Visualizations saved to visuals/ directory")
    print("\nSkills Demonstrated:")
    print("   * Pandas: Groupby, aggregations, data cleaning")
    print("   * NumPy: Statistical calculations, normalization")
    print("   * Matplotlib: Multi-panel layouts, heatmaps, scatter plots")
    print("="*70)

    return {
        'data': df,
        'source_summary': source_summary,
        'quality_metrics': quality_metrics,
        'timing_data': timing_data,
        'comparison': comparison
    }


if __name__ == "__main__":
    results = main()
