"""
Creates a few summary report visualizations combining different analyses.
Basically just putting together nice-looking dashboards from the data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from src.explore import load_all_news_data


def prepare_data(df):
    """Get data ready with all the columns we need"""
    df = df.copy()

    # Parse dates
    df['date'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df['date_only'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()

    # Get source names
    if 'source.name' in df.columns:
        df['source_name'] = df['source.name']
    elif 'source' in df.columns:
        df['source_name'] = df['source'].apply(
            lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
        )
    else:
        df['source_name'] = 'Unknown'

    # Text for searching
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['text'] = df['text'].str.lower()

    # Content stats
    df['title_length'] = df['title'].astype(str).str.len()
    df['desc_length'] = df['description'].fillna('').astype(str).str.len()
    df['has_desc'] = df['description'].notna() & (df['desc_length'] > 0)
    df['has_image'] = df['urlToImage'].notna() if 'urlToImage' in df.columns else False

    return df


def create_executive_summary(df):
    """Main dashboard showing overview stats"""
    print("\nCreating executive summary dashboard...")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Tech News Trends - Summary Report', fontsize=16, fontweight='bold')

    # Calculate some basic stats
    total_articles = len(df)
    num_sources = df['source_name'].nunique()
    date_range = (df['date_only'].max() - df['date_only'].min()).days + 1
    articles_per_day = total_articles / date_range if date_range > 0 else 0

    # 1. Key stats text box (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    stats_text = f"""
KEY STATS

Total Articles: {total_articles:,}
Unique Sources: {num_sources}
Days of Data: {date_range}
Avg Per Day: {articles_per_day:.1f}

Has Description: {df['has_desc'].mean()*100:.1f}%
Has Image: {df['has_image'].mean()*100:.1f}%
    """
    ax1.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # 2. Provider split (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    provider_counts = df['data_provider'].value_counts()
    ax2.pie(provider_counts.values, labels=provider_counts.index,
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Data Providers', fontweight='bold')

    # 3. Top keywords (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    keywords = {
        'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt'],
        'Crypto': ['bitcoin', 'crypto', 'blockchain'],
        'Startup': ['startup', 'funding'],
        'Tech Giants': ['google', 'apple', 'microsoft', 'amazon']
    }
    keyword_counts = {}
    for name, terms in keywords.items():
        pattern = '|'.join(terms)
        keyword_counts[name] = df['text'].str.contains(pattern, case=False).sum()

    ax3.barh(list(keyword_counts.keys()), list(keyword_counts.values()))
    ax3.set_xlabel('Mentions')
    ax3.set_title('Top Keywords', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # 4. Daily volume (middle row, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    daily = df.groupby('date_only').size()
    dates = pd.to_datetime(daily.index)
    ax4.plot(dates, daily.values, marker='o', linewidth=2, markersize=3)

    # Add 7-day average if we have enough data
    if len(daily) >= 7:
        rolling = pd.Series(daily.values).rolling(window=7, center=True).mean()
        ax4.plot(dates, rolling, '--', linewidth=2, label='7-day avg')
        ax4.legend()

    ax4.set_xlabel('Date')
    ax4.set_ylabel('Articles')
    ax4.set_title('Daily Article Volume', fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Top sources (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    top_sources = df['source_name'].value_counts().head(8)
    ax5.barh(range(len(top_sources)), top_sources.values)
    ax5.set_yticks(range(len(top_sources)))
    ax5.set_yticklabels([s[:20] for s in top_sources.index], fontsize=8)
    ax5.set_xlabel('Articles')
    ax5.set_title('Top Sources', fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

    # 6. Day of week (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
    ax6.bar(range(len(day_counts)), day_counts.values)
    ax6.set_xticks(range(len(day_counts)))
    ax6.set_xticklabels([d[:3] for d in day_order], fontsize=8)
    ax6.set_ylabel('Articles')
    ax6.set_title('By Day of Week', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    # 7. Hourly pattern (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    hourly = df['hour'].value_counts().sort_index()
    ax7.bar(hourly.index, hourly.values)
    ax7.set_xlabel('Hour (UTC)')
    ax7.set_ylabel('Articles')
    ax7.set_title('By Hour of Day', fontweight='bold')
    ax7.set_xticks(range(0, 24, 3))
    ax7.grid(axis='y', alpha=0.3)

    # 8. Content length scatter (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    sample = df.sample(n=min(300, len(df)))
    ax8.scatter(sample['title_length'], sample['desc_length'], alpha=0.4, s=10)
    ax8.axvline(df['title_length'].median(), color='r', linestyle='--', alpha=0.6)
    ax8.axhline(df['desc_length'].median(), color='r', linestyle='--', alpha=0.6)
    ax8.set_xlabel('Title Length')
    ax8.set_ylabel('Description Length')
    ax8.set_title('Content Length', fontweight='bold')
    ax8.grid(alpha=0.3)

    plt.tight_layout()

    output = 'visuals/Executive_Summary_Report.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def create_trend_report(df):
    """Show keyword trends over time"""
    print("\nCreating trend analysis report...")

    keywords = {
        'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt'],
        'Crypto': ['blockchain', 'bitcoin', 'crypto'],
        'Startup': ['startup', 'funding', 'investment'],
        'Cloud': ['cloud', 'aws'],
        'Security': ['cybersecurity', 'security']
    }

    # Count keywords by day
    dates = sorted(df['date_only'].unique())
    trends = []
    for date in dates:
        daily = df[df['date_only'] == date]
        row = {'date': date}
        for name, terms in keywords.items():
            pattern = '|'.join(terms)
            row[name] = daily['text'].str.contains(pattern, case=False).sum()
        trends.append(row)

    trend_df = pd.DataFrame(trends)
    trend_df['date'] = pd.to_datetime(trend_df['date'])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Keyword Trends Over Time', fontsize=16, fontweight='bold')

    # All trends together
    ax1 = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, name in enumerate(keywords.keys()):
        ax1.plot(trend_df['date'], trend_df[name], marker='o',
                label=name, linewidth=2, markersize=4, color=colors[i])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mentions')
    ax1.set_title('All Keywords', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Total mentions bar chart
    ax2 = axes[1]
    totals = {name: trend_df[name].sum() for name in keywords.keys()}
    ax2.barh(list(totals.keys()), list(totals.values()), color=colors)
    ax2.set_xlabel('Total Mentions')
    ax2.set_title('Overall Totals', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output = 'visuals/Trend_Analysis_Report.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    """Generate all reports"""
    print("="*60)
    print("VISUAL REPORTS")
    print("="*60)

    df = load_all_news_data()
    if df.empty:
        print("No data found")
        return

    df = prepare_data(df)

    create_executive_summary(df)
    create_trend_report(df)

    print("\n" + "="*60)
    print("Done! Check visuals/ folder")
    print("="*60)


if __name__ == "__main__":
    main()
