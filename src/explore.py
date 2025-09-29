"""
Data Exploration Module for Tech News Trends

This module provides comprehensive data exploration capabilities for analyzing
collected news articles using pandas, numpy, and matplotlib.

Step 1: Basic Data Exploration
- Load all collected JSON files into a unified DataFrame
- Perform initial data exploration and quality assessment
- Display key statistics and data structure insights
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
# matplotlib import will be added later when needed

# Import our config for directory paths
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


def load_json_files_from_directory(directory: str, provider: str = None) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a specific directory.

    Args:
        directory: Path to directory containing JSON files
        provider: Optional provider name to add to each record

    Returns:
        List of dictionaries containing article data
    """
    articles_data = []

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return articles_data

    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {directory}")

    for filename in sorted(json_files):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract articles from the data structure
            if 'articles' in data:
                articles = data['articles']
                # Add metadata to each article
                for article in articles:
                    article['file_source'] = filename
                    article['data_provider'] = provider or data.get('provider', 'unknown')
                    article['collection_date'] = filename.split('_')[0]  # Extract timestamp

                articles_data.extend(articles)
                print(f"  Loaded {len(articles)} articles from {filename}")
            else:
                print(f"  Warning: No 'articles' key found in {filename}")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    return articles_data


def load_all_news_data() -> pd.DataFrame:
    """
    Load all news data from both provider directories and create unified DataFrame.

    Returns:
        pandas DataFrame with all articles and metadata
    """
    print("=== Loading All News Data ===")

    all_articles = []

    # Load from GNews directory
    gnews_articles = load_json_files_from_directory(config.GNEWS_RAW_DIR, 'gnews')
    all_articles.extend(gnews_articles)

    # Load from NewsAPI directory
    newsapi_articles = load_json_files_from_directory(config.NEWSAPI_RAW_DIR, 'newsapi')
    all_articles.extend(newsapi_articles)

    print(f"\nTotal articles loaded: {len(all_articles)}")

    if not all_articles:
        print("No articles found! Check your data directories.")
        return pd.DataFrame()

    # Convert to DataFrame using json_normalize for flat structure
    df = pd.json_normalize(all_articles)

    print(f"DataFrame created with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def basic_data_exploration(df: pd.DataFrame) -> None:
    """
    Perform basic data exploration on the news DataFrame.

    Args:
        df: pandas DataFrame containing news articles
    """
    print("\n" + "="*60)
    print("BASIC DATA EXPLORATION")
    print("="*60)

    # Dataset overview
    print(f"\n[DATASET] Overview:")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Data types and basic info
    print(f"\n[DATA INFO]:")
    print(df.info(memory_usage='deep'))

    # Basic statistics for numeric columns
    print(f"\n[NUMERIC] Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("   No numeric columns found")

    # Text columns analysis
    print(f"\n[TEXT] Columns Analysis:")
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols[:10]:  # Show first 10 text columns
        if col in df.columns:
            non_null_count = df[col].count()
            unique_count = df[col].nunique()
            print(f"   {col}: {non_null_count} non-null, {unique_count} unique")

    # Missing data analysis
    print(f"\n[MISSING] Data Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)

    # Show only columns with missing data
    missing_cols = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_cols) > 0:
        print(missing_cols.head(10))
    else:
        print("   No missing data found!")

    # Sample data preview
    print(f"\n[SAMPLE] Data (First 3 Rows):")
    print(df.head(3).to_string())

    # Provider distribution
    if 'data_provider' in df.columns:
        print(f"\n[PROVIDERS] Distribution:")
        provider_counts = df['data_provider'].value_counts()
        print(provider_counts)

    # Collection dates
    if 'collection_date' in df.columns:
        print(f"\n[DATES] Collection Range:")
        dates = df['collection_date'].unique()
        print(f"   From: {min(dates)} to {max(dates)}")
        print(f"   Total collection days: {len(dates)}")


def analyze_text_lengths(df: pd.DataFrame) -> None:
    """
    Analyze text length distributions for title and description fields.

    Args:
        df: pandas DataFrame containing news articles
    """
    print("\n" + "="*60)
    print("TEXT LENGTH ANALYSIS")
    print("="*60)

    # Calculate text lengths
    if 'title' in df.columns:
        df['title_length'] = df['title'].astype(str).str.len()
        title_stats = df['title_length'].describe()
        print(f"\n[LENGTH] Title Statistics:")
        print(title_stats)

    if 'description' in df.columns:
        df['description_length'] = df['description'].astype(str).str.len()
        desc_stats = df['description_length'].describe()
        print(f"\n[LENGTH] Description Statistics:")
        print(desc_stats)

    # Find articles with unusually short or long content
    if 'title_length' in df.columns:
        short_titles = df[df['title_length'] < 20]
        long_titles = df[df['title_length'] > 100]
        print(f"\n[OUTLIERS] Content Length:")
        print(f"   Short titles (<20 chars): {len(short_titles)}")
        print(f"   Long titles (>100 chars): {len(long_titles)}")


def main():
    """
    Main function to run the basic data exploration.
    """
    print("Starting Basic Data Exploration")
    print("="*50)

    # Load all data
    df = load_all_news_data()

    if df.empty:
        print("No data to explore. Please check your data files.")
        return

    # Perform basic exploration
    basic_data_exploration(df)

    # Analyze text lengths
    analyze_text_lengths(df)

    print("\n" + "="*50)
    print("[SUCCESS] Complete: Basic Data Exploration")
    print("   Key learnings:")
    print("   - Loaded data using pandas and json_normalize")
    print("   - Analyzed data structure and quality")
    print("   - Calculated basic statistics")
    print("   - Identified text length patterns")
    print("="*50)

    return df


if __name__ == "__main__":
    df = main()