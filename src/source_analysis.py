"""
Analyzing different news sources to see which ones publish the most
and which have better content quality.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.explore import load_all_news_data


def prepare_data(df):
    """Get the data ready for analysis"""
    print("\nPreparing source data...")

    df = df.copy()

    # Get source names - different APIs structure this differently
    if 'source.name' in df.columns:
        df['source_name'] = df['source.name']
    elif 'source' in df.columns:
        # Some sources are dicts, some are strings
        df['source_name'] = df['source'].apply(
            lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
        )
    else:
        df['source_name'] = 'Unknown'

    # Basic content metrics
    df['title_length'] = df['title'].astype(str).str.len()
    df['desc_length'] = df['description'].fillna('').astype(str).str.len()
    df['has_description'] = df['description'].notna() & (df['desc_length'] > 0)
    df['has_image'] = df['urlToImage'].notna() if 'urlToImage' in df.columns else False

    print(f"Found {df['source_name'].nunique()} unique sources")
    return df


def analyze_sources(df):
    """Count articles per source and calculate quality metrics"""
    print("\nAnalyzing sources...")

    # Count articles per source
    source_counts = df['source_name'].value_counts()

    # Calculate quality metrics for each source
    source_stats = df.groupby('source_name').agg({
        'title': 'count',
        'title_length': 'mean',
        'desc_length': 'mean',
        'has_description': 'mean',
        'has_image': 'mean'
    }).round(2)

    source_stats.columns = ['articles', 'avg_title_len', 'avg_desc_len',
                            'desc_rate', 'image_rate']

    # Simple quality score - just based on completeness
    # Higher is better
    source_stats['quality'] = (
        source_stats['desc_rate'] * 50 +  # having a description is most important
        source_stats['image_rate'] * 30 +  # images are nice to have
        (source_stats['avg_title_len'] / 100).clip(0, 1) * 10 +  # decent title length
        (source_stats['avg_desc_len'] / 200).clip(0, 1) * 10  # decent desc length
    ).round(1)

    source_stats = source_stats.sort_values('articles', ascending=False)

    print(f"\nTop 10 sources by volume:")
    print(source_stats.head(10)[['articles', 'quality']])

    return source_stats


def plot_source_distribution(source_stats, top_n=15):
    """Make charts showing source distribution"""
    print(f"\nCreating source distribution charts (top {top_n})...")

    top_sources = source_stats.head(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of article counts
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_sources)))
    ax1.barh(range(len(top_sources)), top_sources['articles'], color=colors)
    ax1.set_yticks(range(len(top_sources)))
    ax1.set_yticklabels([s[:30] for s in top_sources.index], fontsize=9)
    ax1.set_xlabel('Number of Articles')
    ax1.set_title(f'Top {top_n} News Sources', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Pie chart showing concentration
    top5 = source_stats.head(5)['articles'].sum()
    next10 = source_stats.iloc[5:15]['articles'].sum()
    rest = source_stats.iloc[15:]['articles'].sum()

    sizes = [top5, next10, rest]
    labels = [f'Top 5\n{top5} articles',
              f'Next 10\n{next10} articles',
              f'Others\n{rest} articles']
    colors_pie = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Source Concentration', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = 'visuals/Source_Distribution.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_quality_comparison(source_stats, top_n=12):
    """Compare source quality metrics"""
    print(f"\nCreating quality comparison charts (top {top_n})...")

    # Filter to sources with at least 5 articles
    source_stats_filtered = source_stats[source_stats['articles'] >= 5].copy()
    top_quality = source_stats_filtered.nlargest(top_n, 'quality')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Quality scores
    colors_q = plt.cm.RdYlGn(top_quality['quality'] / 100)
    ax1.barh(range(len(top_quality)), top_quality['quality'], color=colors_q)
    ax1.set_yticks(range(len(top_quality)))
    ax1.set_yticklabels([s[:25] for s in top_quality.index], fontsize=8)
    ax1.set_xlabel('Quality Score (0-100)')
    ax1.set_title('Source Quality Scores', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Completeness rates
    x = np.arange(len(top_quality))
    width = 0.35
    ax2.bar(x - width/2, top_quality['desc_rate'] * 100, width,
            label='Has Desc', color='#3498db')
    ax2.bar(x + width/2, top_quality['image_rate'] * 100, width,
            label='Has Image', color='#e74c3c')
    ax2.set_ylabel('Completeness Rate (%)')
    ax2.set_title('Content Completeness', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[:15] for s in top_quality.index], rotation=45, ha='right', fontsize=7)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Title vs description length scatter
    scatter = ax3.scatter(top_quality['avg_title_len'], top_quality['avg_desc_len'],
                         s=top_quality['articles'] * 3, c=top_quality['quality'],
                         cmap='viridis', alpha=0.6, edgecolors='black')
    ax3.set_xlabel('Avg Title Length')
    ax3.set_ylabel('Avg Description Length')
    ax3.set_title('Content Length (size = # articles)', fontweight='bold')
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Quality')

    # Volume vs quality
    ax4.scatter(source_stats_filtered['articles'], source_stats_filtered['quality'],
               alpha=0.5, color='#9b59b6')
    ax4.set_xlabel('Number of Articles')
    ax4.set_ylabel('Quality Score')
    ax4.set_title('Volume vs Quality', fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    output_path = 'visuals/Source_Quality_Comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Run the source analysis"""
    print("="*60)
    print("SOURCE ANALYSIS")
    print("="*60)

    # Load data
    df = load_all_news_data()
    if df.empty:
        print("No data found")
        return

    # Prepare and analyze
    df = prepare_data(df)
    source_stats = analyze_sources(df)

    # Create visualizations
    plot_source_distribution(source_stats, top_n=15)
    plot_quality_comparison(source_stats, top_n=12)

    print("\n" + "="*60)
    print("Done! Check the visuals/ folder for charts")
    print("="*60)

    return source_stats


if __name__ == "__main__":
    results = main()
