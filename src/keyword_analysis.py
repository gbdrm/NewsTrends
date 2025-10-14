"""
Keyword Analysis Module for Tech News Trends

Analyzes keyword frequencies and trends in news content using pandas text operations.
Tracks popular tech terms over time and identifies trending topics.

Features:
- Extract keywords from titles and descriptions
- Count keyword frequencies across the dataset
- Track keyword trends over time
- Visualize keyword popularity and growth
"""

import os
import sys
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explore import load_all_news_data


# Define tech keywords to track
TECH_KEYWORDS = {
    'AI & ML': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
                'neural network', 'chatgpt', 'openai', 'llm', 'gpt', 'claude', 'gemini'],
    'Blockchain & Crypto': ['blockchain', 'bitcoin', 'cryptocurrency', 'crypto', 'ethereum',
                           'nft', 'defi', 'web3', 'metaverse', 'solana'],
    'Business': ['startup', 'funding', 'investment', 'ipo', 'acquisition', 'merger',
                 'venture capital', 'vc', 'valuation', 'unicorn'],
    'Big Tech': ['google', 'apple', 'microsoft', 'amazon', 'meta', 'facebook', 'netflix',
                 'tesla', 'nvidia', 'openai', 'anthropic'],
    'Development': ['python', 'javascript', 'react', 'node', 'api', 'cloud', 'aws',
                   'docker', 'kubernetes', 'github', 'open source'],
    'Emerging Tech': ['quantum', 'robotics', '5g', 'iot', 'ar', 'vr', 'autonomous',
                     'drone', 'biotech', 'fintech', 'cybersecurity']
}


def prepare_text_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare text data for keyword analysis.

    Args:
        df: DataFrame with news articles

    Returns:
        DataFrame with cleaned text columns
    """
    print("\n[TEXT] Preparing text for keyword analysis...")

    # Combine title and description for comprehensive analysis
    df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['full_text'] = df['full_text'].str.lower()

    # Clean text: remove special characters, keep alphanumeric and spaces
    df['clean_text'] = df['full_text'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"   Prepared text for {len(df)} articles")
    print(f"   Average text length: {df['clean_text'].str.len().mean():.0f} characters")

    return df


def extract_keyword_frequencies(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Extract keyword frequencies from the text data.

    Args:
        df: DataFrame with cleaned text

    Returns:
        Dictionary with keyword categories and their frequencies
    """
    print("\n[KEYWORDS] Extracting keyword frequencies...")

    keyword_frequencies = {}
    total_matches = 0

    for category, keywords in TECH_KEYWORDS.items():
        category_counts = {}

        for keyword in keywords:
            # Count occurrences using pandas string contains
            count = df['clean_text'].str.contains(keyword, case=False, na=False).sum()
            if count > 0:
                category_counts[keyword] = count
                total_matches += count

        if category_counts:  # Only include categories with matches
            keyword_frequencies[category] = category_counts

    print(f"   Found {total_matches} total keyword matches")
    print(f"   Categories with matches: {len(keyword_frequencies)}")

    return keyword_frequencies


def analyze_keyword_trends_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how keyword frequencies change over time.

    Args:
        df: DataFrame with articles and datetime info

    Returns:
        DataFrame with keyword trends over time
    """
    print("\n[TRENDS] Analyzing keyword trends over time...")

    # Ensure datetime columns exist
    if 'collection_datetime' not in df.columns:
        df['collection_datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')
        df['collection_date_only'] = df['collection_datetime'].dt.date

    # Track top keywords over time
    trend_data = []

    # Get top keywords across all categories
    all_keyword_counts = {}
    for category, keywords in TECH_KEYWORDS.items():
        for keyword in keywords:
            count = df['clean_text'].str.contains(keyword, case=False, na=False).sum()
            if count > 0:
                all_keyword_counts[keyword] = count

    # Select top 10 keywords for trend analysis
    top_keywords = sorted(all_keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_keyword_names = [kw[0] for kw in top_keywords]

    # Analyze trends by date
    daily_trends = {}
    for date in sorted(df['collection_date_only'].unique()):
        day_df = df[df['collection_date_only'] == date]
        daily_counts = {}

        for keyword in top_keyword_names:
            count = day_df['clean_text'].str.contains(keyword, case=False, na=False).sum()
            daily_counts[keyword] = count

        daily_trends[date] = daily_counts

    # Convert to DataFrame for easy manipulation
    trends_df = pd.DataFrame(daily_trends).T.fillna(0)
    trends_df.index.name = 'date'
    trends_df = trends_df.reset_index()

    print(f"   Tracking {len(top_keyword_names)} keywords over {len(trends_df)} days")
    print(f"   Top keywords: {', '.join(top_keyword_names[:5])}")

    return trends_df


def create_keyword_frequency_chart(keyword_frequencies: Dict[str, Dict[str, int]]) -> None:
    """
    Create bar chart showing keyword frequencies by category.

    Args:
        keyword_frequencies: Dictionary with keyword counts by category
    """
    print("\n[CHART] Creating keyword frequency visualization...")

    # Prepare data for visualization
    categories = []
    totals = []
    top_keywords = []

    for category, keywords in keyword_frequencies.items():
        total_count = sum(keywords.values())
        categories.append(category)
        totals.append(total_count)

        # Get top keyword in category
        top_keyword = max(keywords.items(), key=lambda x: x[1])
        top_keywords.append(f"{top_keyword[0]} ({top_keyword[1]})")

    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(categories, totals, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#21BF73'])

    plt.title('Tech Keyword Frequencies by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Total Mentions', fontsize=12)
    plt.ylabel('Category', fontsize=12)

    # Add value labels and top keywords
    for i, (bar, total, top_kw) in enumerate(zip(bars, totals, top_keywords)):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{total}\n(Top: {top_kw})', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()


def create_keyword_trends_timeline(trends_df: pd.DataFrame) -> None:
    """
    Create timeline showing how top keywords trend over time.

    Args:
        trends_df: DataFrame with keyword counts by date
    """
    print("\n[CHART] Creating keyword trends timeline...")

    # Get top 6 keywords by total mentions
    keyword_totals = trends_df.iloc[:, 1:].sum().sort_values(ascending=False)
    top_6_keywords = keyword_totals.head(6).index.tolist()

    plt.figure(figsize=(14, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#21BF73']

    for i, keyword in enumerate(top_6_keywords):
        plt.plot(trends_df['date'], trends_df[keyword],
                marker='o', linewidth=2, label=f'{keyword.title()} (Total: {keyword_totals[keyword]})',
                color=colors[i % len(colors)])

    plt.title('Top Tech Keywords Timeline', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Mentions', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_keyword_heatmap(trends_df: pd.DataFrame) -> None:
    """
    Create heatmap showing keyword intensity over time.

    Args:
        trends_df: DataFrame with keyword counts by date
    """
    print("\n[CHART] Creating keyword trends heatmap...")

    # Get top 8 keywords for heatmap
    keyword_totals = trends_df.iloc[:, 1:].sum().sort_values(ascending=False)
    top_8_keywords = keyword_totals.head(8).index.tolist()

    # Prepare data for heatmap
    heatmap_data = trends_df[top_8_keywords].T

    plt.figure(figsize=(14, 8))
    plt.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')

    plt.title('Keyword Intensity Heatmap Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Keywords', fontsize=12)

    # Set ticks and labels
    plt.yticks(range(len(top_8_keywords)), [kw.title() for kw in top_8_keywords])
    plt.xticks(range(0, len(trends_df), max(1, len(trends_df)//10)),
              [str(date)[-5:] for date in trends_df['date'][::max(1, len(trends_df)//10)]], rotation=45)

    plt.colorbar(label='Daily Mentions')
    plt.tight_layout()
    plt.show()


def generate_keyword_insights(keyword_frequencies: Dict, trends_df: pd.DataFrame) -> Dict:
    """
    Generate insights from keyword analysis.

    Args:
        keyword_frequencies: Keyword frequency data
        trends_df: Trends over time data

    Returns:
        Dictionary with key insights
    """
    print("\n[INSIGHTS] Generating keyword insights...")

    # Overall statistics
    total_articles = len(trends_df)
    total_keyword_mentions = sum(sum(keywords.values()) for keywords in keyword_frequencies.values())

    # Most popular category
    category_totals = {cat: sum(keywords.values()) for cat, keywords in keyword_frequencies.items()}
    top_category = max(category_totals.items(), key=lambda x: x[1])

    # Most mentioned keyword overall
    all_keywords = {}
    for keywords in keyword_frequencies.values():
        all_keywords.update(keywords)
    top_keyword = max(all_keywords.items(), key=lambda x: x[1])

    # Trend analysis (if we have enough data)
    trending_up = []
    if len(trends_df) > 7:  # Need at least a week of data
        # Compare first vs last week averages
        first_week = trends_df.head(7).iloc[:, 1:].mean()
        last_week = trends_df.tail(7).iloc[:, 1:].mean()

        for keyword in first_week.index:
            if last_week[keyword] > first_week[keyword] * 1.2:  # 20% increase
                trending_up.append(keyword)

    insights = {
        'total_mentions': total_keyword_mentions,
        'articles_analyzed': total_articles,
        'top_category': top_category,
        'top_keyword': top_keyword,
        'trending_up': trending_up,
        'categories_found': len(keyword_frequencies)
    }

    print(f"   Total keyword mentions: {total_keyword_mentions}")
    print(f"   Top category: {top_category[0]} ({top_category[1]} mentions)")
    print(f"   Most mentioned: {top_keyword[0]} ({top_keyword[1]} times)")
    if trending_up:
        print(f"   Trending up: {', '.join(trending_up)}")

    return insights


def run_keyword_analysis(df: pd.DataFrame) -> Dict:
    """
    Run complete keyword analysis pipeline.

    Args:
        df: DataFrame with news articles

    Returns:
        Dictionary with analysis results
    """
    print("="*60)
    print("KEYWORD ANALYSIS")
    print("="*60)

    # Prepare text data
    df = prepare_text_for_analysis(df)

    # Extract keyword frequencies
    keyword_frequencies = extract_keyword_frequencies(df)

    # Analyze trends over time
    trends_df = analyze_keyword_trends_over_time(df)

    # Create visualizations
    create_keyword_frequency_chart(keyword_frequencies)
    create_keyword_trends_timeline(trends_df)
    create_keyword_heatmap(trends_df)

    # Generate insights
    insights = generate_keyword_insights(keyword_frequencies, trends_df)

    print("\n" + "="*50)
    print("[SUCCESS] Keyword Analysis Complete!")
    print("   Key achievements:")
    print(f"   - Analyzed {insights['articles_analyzed']} articles for tech keywords")
    print(f"   - Found {insights['total_mentions']} keyword mentions across {insights['categories_found']} categories")
    print(f"   - Created 3 keyword visualizations")
    print(f"   - Tracked trends for top keywords over time")
    print("="*50)

    return {
        'keyword_frequencies': keyword_frequencies,
        'trends_data': trends_df,
        'insights': insights
    }


def main():
    """
    Main function to run keyword analysis.
    """
    print("Starting Keyword Analysis")
    print("="*50)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data to analyze. Please check your data files.")
        return

    # Run analysis
    results = run_keyword_analysis(df)

    return df, results


if __name__ == "__main__":
    df, results = main()