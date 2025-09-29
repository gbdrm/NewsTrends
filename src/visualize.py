import os
import sys
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explore import load_all_news_data


def setup_matplotlib():
    """Configure matplotlib for consistent chart styling."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def create_provider_distribution_chart(df: pd.DataFrame) -> None:
    """
    Create bar chart showing article count by data provider.

    Args:
        df: DataFrame with news articles
    """
    print("\n[CHART] Creating Provider Distribution Bar Chart...")

    # Count articles by provider
    provider_counts = df['data_provider'].value_counts()

    # Create bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(provider_counts.index, provider_counts.values,
                   color=['#2E86AB', '#A23B72', '#F18F01'])

    # Customize chart
    plt.title('News Articles by Data Provider', fontsize=16, fontweight='bold')
    plt.xlabel('Data Provider', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Add percentage labels
    total = provider_counts.sum()
    for i, (provider, count) in enumerate(provider_counts.items()):
        percentage = (count / total) * 100
        plt.text(i, count/2, f'{percentage:.1f}%',
                ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"   Provider distribution: {dict(provider_counts)}")


def create_text_length_histograms(df: pd.DataFrame) -> None:
    """
    Create histograms showing distribution of title and description lengths.

    Args:
        df: DataFrame with news articles
    """
    print("\n[CHART] Creating Text Length Distribution Histograms...")

    # Calculate text lengths if not already done
    if 'title_length' not in df.columns:
        df['title_length'] = df['title'].astype(str).str.len()
    if 'description_length' not in df.columns:
        df['description_length'] = df['description'].astype(str).str.len()

    # Create subplot for both histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Title length histogram
    ax1.hist(df['title_length'], bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_title('Title Length Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Characters', fontsize=12)
    ax1.set_ylabel('Number of Articles', fontsize=12)
    ax1.axvline(df['title_length'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["title_length"].mean():.1f}')
    ax1.legend()

    # Description length histogram
    ax2.hist(df['description_length'], bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_title('Description Length Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Characters', fontsize=12)
    ax2.set_ylabel('Number of Articles', fontsize=12)
    ax2.axvline(df['description_length'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["description_length"].mean():.1f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"   Title length - Mean: {df['title_length'].mean():.1f}, "
          f"Range: {df['title_length'].min()}-{df['title_length'].max()}")
    print(f"   Description length - Mean: {df['description_length'].mean():.1f}, "
          f"Range: {df['description_length'].min()}-{df['description_length'].max()}")


def create_collection_timeline(df: pd.DataFrame) -> None:
    """
    Create line plot showing article collection over time.

    Args:
        df: DataFrame with news articles
    """
    print("\n[CHART] Creating Collection Timeline...")

    # Convert collection_date to datetime for proper sorting
    df['collection_datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')

    # Count articles by date
    daily_counts = df.groupby(df['collection_datetime'].dt.date).size().reset_index()
    daily_counts.columns = ['date', 'article_count']

    # Create line plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_counts['date'], daily_counts['article_count'],
             marker='o', linewidth=2, markersize=6, color='#F18F01')

    # Customize chart
    plt.title('Daily Article Collection Timeline', fontsize=16, fontweight='bold')
    plt.xlabel('Collection Date', fontsize=12)
    plt.ylabel('Articles Collected', fontsize=12)
    plt.xticks(rotation=45)

    # Add trend line using numpy
    x_numeric = np.arange(len(daily_counts))
    z = np.polyfit(x_numeric, daily_counts['article_count'], 1)
    p = np.poly1d(z)
    plt.plot(daily_counts['date'], p(x_numeric), "--", alpha=0.8, color='red',
             label=f'Trend (slope: {z[0]:.2f})')

    # Add average line
    avg_articles = daily_counts['article_count'].mean()
    plt.axhline(y=avg_articles, color='green', linestyle=':', alpha=0.7,
                label=f'Average: {avg_articles:.1f}')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print timeline summary
    total_days = len(daily_counts)
    total_articles = daily_counts['article_count'].sum()
    print(f"   Collection period: {total_days} days")
    print(f"   Total articles: {total_articles}")
    print(f"   Average per day: {avg_articles:.1f}")
    print(f"   Trend: {z[0]:.2f} articles/day change")


def generate_basic_visualizations(df: pd.DataFrame) -> None:
    """
    Generate all basic visualizations for the news data.

    Args:
        df: DataFrame with news articles
    """
    print("="*60)
    print("GENERATING BASIC VISUALIZATIONS")
    print("="*60)

    # Setup matplotlib styling
    setup_matplotlib()

    # Create all charts
    create_provider_distribution_chart(df)
    create_text_length_histograms(df)
    create_collection_timeline(df)

    print("\n" + "="*50)
    print("[SUCCESS] Basic Visualizations Complete!")
    print("   Created 3 charts:")
    print("   - Provider distribution (bar chart)")
    print("   - Text length distributions (histograms)")
    print("   - Collection timeline (line plot)")
    print("="*50)


def main():
    """
    Main function to load data and generate basic visualizations.
    """
    print("Starting Basic Visualization Generation")
    print("="*50)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data to visualize. Please check your data files.")
        return

    # Generate visualizations
    generate_basic_visualizations(df)

    return df


if __name__ == "__main__":
    df = main()