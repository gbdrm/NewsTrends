"""
Visual Reports Module for Tech News Trends

Creates comprehensive, publication-ready reports combining all analyses:
- Executive summary dashboard
- Trend analysis overview
- Source analysis summary
- Time-based patterns
- Multi-page report layouts

Part of the NewsTrends project - demonstrating advanced Matplotlib visualization skills
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explore import load_all_news_data
from src.config import GNEWS_RAW_DIR, NEWSAPI_RAW_DIR


def set_report_style():
    """Set consistent styling for all report visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#dee2e6',
        'axes.labelcolor': '#212529',
        'text.color': '#212529',
        'xtick.color': '#495057',
        'ytick.color': '#495057',
        'grid.color': '#dee2e6',
        'grid.alpha': 0.5,
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 14
    })


def prepare_data_for_reports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and enrich data for report generation.

    Args:
        df: Raw news DataFrame

    Returns:
        Enriched DataFrame with all needed columns
    """
    print("\n[PREP] Preparing data for reports...")

    df = df.copy()

    # Parse dates
    df['collection_datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df['collection_date_only'] = df['collection_datetime'].dt.date
    df['collection_hour'] = df['collection_datetime'].dt.hour
    df['collection_day_of_week'] = df['collection_datetime'].dt.day_name()

    # Handle published dates
    if 'publishedAt' in df.columns:
        df['published_at'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    elif 'published_at' not in df.columns:
        df['published_at'] = df['collection_datetime']

    # Extract source name
    if 'source.name' in df.columns:
        df['source_name'] = df['source.name']
    elif 'source' in df.columns:
        df['source_name'] = df['source'].apply(
            lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
        )
    else:
        df['source_name'] = 'Unknown'

    # Text processing
    df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['clean_text'] = df['full_text'].str.lower()

    # Content metrics
    df['title_length'] = df['title'].astype(str).str.len()
    df['description_length'] = df['description'].fillna('').astype(str).str.len()
    df['has_description'] = df['description'].notna() & (df['description_length'] > 0)
    df['has_image'] = df['urlToImage'].notna() if 'urlToImage' in df.columns else False

    print(f"   Prepared {len(df)} articles")
    print(f"   Date range: {df['collection_date_only'].min()} to {df['collection_date_only'].max()}")
    print(f"   Providers: {df['data_provider'].unique().tolist()}")

    return df


def calculate_key_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate key metrics for executive summary.

    Args:
        df: Prepared DataFrame

    Returns:
        Dictionary with key metrics
    """
    print("\n[METRICS] Calculating key metrics...")

    # Define keywords to track
    keywords = {
        'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'openai'],
        'Blockchain': ['blockchain', 'bitcoin', 'crypto', 'ethereum'],
        'Startup': ['startup', 'funding', 'investment', 'ipo'],
        'Big Tech': ['google', 'apple', 'microsoft', 'amazon', 'meta', 'tesla']
    }

    # Calculate keyword mentions
    keyword_stats = {}
    for category, terms in keywords.items():
        pattern = '|'.join(terms)
        count = df['clean_text'].str.contains(pattern, case=False, na=False).sum()
        keyword_stats[category] = count

    # Time metrics
    date_range_days = (df['collection_date_only'].max() - df['collection_date_only'].min()).days + 1

    metrics = {
        'total_articles': len(df),
        'unique_sources': df['source_name'].nunique(),
        'date_range_days': date_range_days,
        'articles_per_day': len(df) / date_range_days if date_range_days > 0 else 0,
        'providers': df['data_provider'].value_counts().to_dict(),
        'keyword_mentions': keyword_stats,
        'avg_title_length': df['title_length'].mean(),
        'avg_desc_length': df['description_length'].mean(),
        'description_rate': df['has_description'].mean() * 100,
        'image_rate': df['has_image'].mean() * 100,
        'top_sources': df['source_name'].value_counts().head(5).to_dict(),
        'start_date': df['collection_date_only'].min(),
        'end_date': df['collection_date_only'].max()
    }

    print(f"   Calculated {len(metrics)} key metrics")
    return metrics


def create_executive_summary_report(df: pd.DataFrame, metrics: Dict):
    """
    Create executive summary dashboard with key insights.

    Args:
        df: Prepared DataFrame
        metrics: Dictionary with key metrics
    """
    print("\n[REPORT] Creating Executive Summary Dashboard...")

    set_report_style()

    # Create figure with custom grid
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Add main title
    fig.suptitle('Tech News Trends - Executive Summary Report',
                 fontsize=18, fontweight='bold', y=0.98)

    # Add subtitle with date range
    fig.text(0.5, 0.94,
             f"Analysis Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['date_range_days']} days)",
             ha='center', fontsize=11, style='italic', color='#495057')

    # 1. Key Metrics Panel (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    metrics_text = f"""
    KEY METRICS

    Total Articles: {metrics['total_articles']:,}
    Unique Sources: {metrics['unique_sources']}
    Articles/Day: {metrics['articles_per_day']:.1f}

    Avg Title Length: {metrics['avg_title_length']:.0f} chars
    Avg Desc Length: {metrics['avg_desc_length']:.0f} chars

    Has Description: {metrics['description_rate']:.1f}%
    Has Image: {metrics['image_rate']:.1f}%
    """

    ax1.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#e9ecef', alpha=0.8))

    # 2. Provider Distribution (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    providers = metrics['providers']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    wedges, texts, autotexts = ax2.pie(providers.values(), labels=providers.keys(),
                                         autopct='%1.1f%%', colors=colors[:len(providers)],
                                         startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Data Provider Distribution', fontweight='bold', pad=10)

    # 3. Top Keywords (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    keywords = metrics['keyword_mentions']
    keyword_names = list(keywords.keys())
    keyword_counts = list(keywords.values())
    colors_bar = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']

    bars = ax3.barh(keyword_names, keyword_counts, color=colors_bar)
    ax3.set_xlabel('Number of Mentions', fontweight='bold')
    ax3.set_title('Top Keyword Categories', fontweight='bold', pad=10)
    ax3.invert_yaxis()

    # Add value labels
    for bar, count in zip(bars, keyword_counts):
        ax3.text(count + max(keyword_counts) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontweight='bold', fontsize=9)

    # 4. Daily Article Volume (middle, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    daily_counts = df.groupby('collection_date_only').size()
    dates = pd.to_datetime(daily_counts.index)

    ax4.plot(dates, daily_counts.values, marker='o', linestyle='-',
             linewidth=2, markersize=4, color='#3498db', alpha=0.7)

    # Add rolling average
    rolling_avg = pd.Series(daily_counts.values).rolling(window=7, center=True).mean()
    ax4.plot(dates, rolling_avg, linestyle='--', linewidth=2.5,
             color='#e74c3c', label='7-day Average')

    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylabel('Number of Articles', fontweight='bold')
    ax4.set_title('Daily Article Volume Over Time', fontweight='bold', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 5. Top Sources (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    top_sources = metrics['top_sources']
    source_names = [s[:20] + '...' if len(s) > 20 else s for s in top_sources.keys()]
    source_counts = list(top_sources.values())

    bars = ax5.barh(range(len(source_names)), source_counts, color='#2ecc71', alpha=0.8)
    ax5.set_yticks(range(len(source_names)))
    ax5.set_yticklabels(source_names, fontsize=8)
    ax5.set_xlabel('Articles', fontweight='bold')
    ax5.set_title('Top 5 News Sources', fontweight='bold', pad=10)
    ax5.invert_yaxis()

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, source_counts)):
        ax5.text(count + max(source_counts) * 0.02, i,
                f'{count}', va='center', fontsize=8, fontweight='bold')

    # 6. Day of Week Pattern (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['collection_day_of_week'].value_counts()
    day_counts = day_counts.reindex(day_order, fill_value=0)

    colors_days = plt.cm.viridis(np.linspace(0.2, 0.8, len(day_counts)))
    bars = ax6.bar(range(len(day_counts)), day_counts.values, color=colors_days)
    ax6.set_xticks(range(len(day_counts)))
    ax6.set_xticklabels([d[:3] for d in day_order], fontsize=8)
    ax6.set_ylabel('Articles', fontweight='bold')
    ax6.set_title('Articles by Day of Week', fontweight='bold', pad=10)
    ax6.grid(axis='y', alpha=0.3)

    # 7. Hourly Collection Pattern (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    hourly_counts = df['collection_hour'].value_counts().sort_index()

    ax7.bar(hourly_counts.index, hourly_counts.values, color='#9b59b6', alpha=0.8)
    ax7.set_xlabel('Hour of Day (UTC)', fontweight='bold')
    ax7.set_ylabel('Articles', fontweight='bold')
    ax7.set_title('Article Collection by Hour', fontweight='bold', pad=10)
    ax7.set_xticks(range(0, 24, 3))
    ax7.grid(axis='y', alpha=0.3)

    # 8. Content Length Distribution (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])

    # Create scatter of title vs description length
    sample_size = min(500, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    scatter = ax8.scatter(sample_df['title_length'], sample_df['description_length'],
                         alpha=0.4, s=20, c=sample_df.index, cmap='viridis')

    # Add median lines
    ax8.axvline(df['title_length'].median(), color='red', linestyle='--',
                alpha=0.7, linewidth=2, label=f"Median Title: {df['title_length'].median():.0f}")
    ax8.axhline(df['description_length'].median(), color='blue', linestyle='--',
                alpha=0.7, linewidth=2, label=f"Median Desc: {df['description_length'].median():.0f}")

    ax8.set_xlabel('Title Length (chars)', fontweight='bold')
    ax8.set_ylabel('Description Length (chars)', fontweight='bold')
    ax8.set_title('Content Length Distribution', fontweight='bold', pad=10)
    ax8.legend(fontsize=7, loc='upper right')
    ax8.grid(True, alpha=0.3)

    # Add footer
    fig.text(0.5, 0.02,
             'Generated by NewsTrends Analysis System | Powered by Pandas, NumPy, Matplotlib',
             ha='center', fontsize=8, style='italic', color='#6c757d')

    # Save
    output_path = 'visuals/Executive_Summary_Report.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {output_path}")

    plt.close()


def create_trend_analysis_report(df: pd.DataFrame):
    """
    Create comprehensive trend analysis report.

    Args:
        df: Prepared DataFrame
    """
    print("\n[REPORT] Creating Trend Analysis Report...")

    set_report_style()

    # Define keywords to track
    trending_keywords = {
        'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt'],
        'Crypto': ['blockchain', 'bitcoin', 'crypto', 'ethereum'],
        'Startup': ['startup', 'funding', 'investment'],
        'Cloud': ['cloud', 'aws', 'azure'],
        'Security': ['cybersecurity', 'security', 'privacy']
    }

    # Calculate daily trends
    daily_trends = []
    dates = sorted(df['collection_date_only'].unique())

    for date in dates:
        daily_df = df[df['collection_date_only'] == date]
        trend_row = {'date': date, 'total': len(daily_df)}

        for category, keywords in trending_keywords.items():
            pattern = '|'.join(keywords)
            count = daily_df['clean_text'].str.contains(pattern, case=False, na=False).sum()
            trend_row[category] = count

        daily_trends.append(trend_row)

    trends_df = pd.DataFrame(daily_trends)
    trends_df['date'] = pd.to_datetime(trends_df['date'])

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Title
    fig.suptitle('Tech News Trends - Keyword Analysis Report',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. All trends over time (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, category in enumerate(trending_keywords.keys()):
        ax1.plot(trends_df['date'], trends_df[category],
                marker='o', linestyle='-', linewidth=2, markersize=4,
                label=category, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Number of Mentions', fontweight='bold')
    ax1.set_title('Keyword Trends Over Time', fontweight='bold', pad=15, fontsize=14)
    ax1.legend(loc='upper left', ncol=5, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2-6. Individual trend details (middle and bottom rows)
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for idx, (category, keywords) in enumerate(list(trending_keywords.items())[:4]):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        # Plot trend
        values = trends_df[category].values
        ax.plot(trends_df['date'], values, marker='o', linestyle='-',
               linewidth=2, markersize=5, color=colors[idx], alpha=0.8)

        # Add rolling average
        if len(values) >= 3:
            rolling = pd.Series(values).rolling(window=3, center=True).mean()
            ax.plot(trends_df['date'], rolling, linestyle='--', linewidth=2,
                   color='gray', alpha=0.6, label='3-day avg')

        # Calculate statistics
        total_mentions = int(values.sum())
        avg_daily = values.mean()
        max_daily = int(values.max())

        # Add growth indicator
        if len(values) >= 2:
            recent_avg = values[-7:].mean() if len(values) >= 7 else values[-len(values)//2:].mean()
            earlier_avg = values[:7].mean() if len(values) >= 7 else values[:len(values)//2].mean()
            growth = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0

            growth_text = f"{'↑' if growth > 0 else '↓'} {abs(growth):.1f}%"
            growth_color = '#2ecc71' if growth > 0 else '#e74c3c'
        else:
            growth_text = "N/A"
            growth_color = '#95a5a6'

        # Add stats text box
        stats_text = f"Total: {total_mentions}\nAvg/day: {avg_daily:.1f}\nPeak: {max_daily}\nTrend: "
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.78, growth_text, transform=ax.transAxes,
               fontsize=10, fontweight='bold', verticalalignment='top',
               color=growth_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Date', fontweight='bold', fontsize=9)
        ax.set_ylabel('Mentions', fontweight='bold', fontsize=9)
        ax.set_title(f'{category} Trend', fontweight='bold', pad=10, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)

        if len(values) >= 3:
            ax.legend(fontsize=7, loc='upper right')

    # Footer
    fig.text(0.5, 0.02,
             f'Analysis of {len(df)} articles across {len(dates)} days | Keywords tracked: {sum(len(k) for k in trending_keywords.values())}',
             ha='center', fontsize=8, style='italic', color='#6c757d')

    # Save
    output_path = 'visuals/Trend_Analysis_Report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {output_path}")

    plt.close()


def create_source_overview_report(df: pd.DataFrame):
    """
    Create source analysis overview report.

    Args:
        df: Prepared DataFrame
    """
    print("\n[REPORT] Creating Source Overview Report...")

    set_report_style()

    # Calculate source metrics
    source_stats = df.groupby('source_name').agg({
        'title': 'count',
        'title_length': 'mean',
        'description_length': 'mean',
        'has_description': 'mean',
        'has_image': 'mean',
        'data_provider': 'first'
    }).round(2)

    source_stats.columns = ['articles', 'avg_title_len', 'avg_desc_len',
                            'desc_rate', 'image_rate', 'provider']

    # Calculate quality score
    source_stats['quality_score'] = (
        source_stats['desc_rate'] * 40 +
        source_stats['image_rate'] * 20 +
        np.clip(source_stats['avg_title_len'] / 100, 0, 1) * 20 +
        np.clip(source_stats['avg_desc_len'] / 200, 0, 1) * 20
    ).round(1)

    source_stats = source_stats.sort_values('articles', ascending=False)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Tech News Trends - Source Analysis Report',
                 fontsize=18, fontweight='bold', y=0.96)

    # 1. Top sources by volume (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    top_10 = source_stats.head(10)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))

    bars = ax1.barh(range(len(top_10)), top_10['articles'], color=colors)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([s[:25] for s in top_10.index], fontsize=8)
    ax1.set_xlabel('Number of Articles', fontweight='bold')
    ax1.set_title('Top 10 Sources by Volume', fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Add labels
    for i, count in enumerate(top_10['articles']):
        ax1.text(count, i, f' {count}', va='center', fontsize=8)

    # 2. Source quality scores (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    top_quality = source_stats.nlargest(10, 'quality_score')
    colors_q = plt.cm.RdYlGn(top_quality['quality_score'] / 100)

    bars = ax2.barh(range(len(top_quality)), top_quality['quality_score'], color=colors_q)
    ax2.set_yticks(range(len(top_quality)))
    ax2.set_yticklabels([s[:25] for s in top_quality.index], fontsize=8)
    ax2.set_xlabel('Quality Score (0-100)', fontweight='bold')
    ax2.set_title('Top 10 Sources by Quality', fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 105)

    # Add labels
    for i, score in enumerate(top_quality['quality_score']):
        ax2.text(score + 1, i, f'{score:.0f}', va='center', fontsize=8, fontweight='bold')

    # 3. Provider comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    provider_stats = df.groupby('data_provider').agg({
        'title': 'count',
        'source_name': 'nunique'
    })
    provider_stats.columns = ['Articles', 'Sources']

    x = np.arange(len(provider_stats))
    width = 0.35

    bars1 = ax3.bar(x - width/2, provider_stats['Articles'], width,
                    label='Articles', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, provider_stats['Sources'], width,
                    label='Sources', color='#e74c3c', alpha=0.8)

    ax3.set_xlabel('Provider', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Provider Comparison', fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(provider_stats.index)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

    # 4. Content completeness heatmap (bottom left)
    ax4 = fig.add_subplot(gs[1, :2])

    top_sources_comp = source_stats.head(15)
    completeness_data = top_sources_comp[['desc_rate', 'image_rate']].T * 100

    im = ax4.imshow(completeness_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax4.set_xticks(range(len(top_sources_comp)))
    ax4.set_xticklabels([s[:20] for s in top_sources_comp.index],
                        rotation=45, ha='right', fontsize=8)
    ax4.set_yticks(range(2))
    ax4.set_yticklabels(['Description %', 'Image %'], fontsize=9)
    ax4.set_title('Content Completeness - Top 15 Sources', fontweight='bold', pad=10)

    # Add text annotations
    for i in range(2):
        for j in range(len(top_sources_comp)):
            text = ax4.text(j, i, f'{completeness_data.iloc[i, j]:.0f}%',
                          ha='center', va='center', color='black', fontsize=7)

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Completeness %', fontweight='bold')

    # 5. Content length comparison (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])

    top_15 = source_stats.head(15)
    scatter = ax5.scatter(top_15['avg_title_len'], top_15['avg_desc_len'],
                         s=top_15['articles'] * 5,
                         c=top_15['quality_score'], cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=1)

    ax5.set_xlabel('Avg Title Length', fontweight='bold')
    ax5.set_ylabel('Avg Description Length', fontweight='bold')
    ax5.set_title('Content Length Analysis\n(size = volume)', fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Quality Score', fontweight='bold', fontsize=8)

    # Footer
    fig.text(0.5, 0.02,
             f'Analyzed {len(df)} articles from {len(source_stats)} sources | Average quality score: {source_stats["quality_score"].mean():.1f}/100',
             ha='center', fontsize=8, style='italic', color='#6c757d')

    # Save
    output_path = 'visuals/Source_Overview_Report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {output_path}")

    plt.close()


def main():
    """
    Main function to generate all visual reports.
    """
    print("="*70)
    print(" "*15 + "VISUAL REPORTS GENERATOR")
    print("="*70)

    # Load data
    df = load_all_news_data()

    if df.empty:
        print("No data available for reports")
        return

    # Prepare data
    df = prepare_data_for_reports(df)

    # Calculate metrics
    metrics = calculate_key_metrics(df)

    # Generate reports
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE REPORTS")
    print("="*70)

    create_executive_summary_report(df, metrics)
    create_trend_analysis_report(df)
    create_source_overview_report(df)

    print("\n" + "="*70)
    print("[SUCCESS] All Visual Reports Generated!")
    print("="*70)
    print("\nReports Created:")
    print("   1. Executive_Summary_Report.png - Key metrics dashboard")
    print("   2. Trend_Analysis_Report.png - Keyword trends over time")
    print("   3. Source_Overview_Report.png - Source analysis summary")
    print("\nAll reports saved to: visuals/")
    print("\nSkills Demonstrated:")
    print("   * Matplotlib: GridSpec, multi-panel layouts, custom styling")
    print("   * Pandas: Complex aggregations, time series analysis")
    print("   * NumPy: Statistical calculations, data normalization")
    print("   * Data Visualization: Professional report design")
    print("="*70)


if __name__ == "__main__":
    main()
