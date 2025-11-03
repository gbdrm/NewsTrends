"""
Finding trending keywords - which ones are growing vs declining
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.explore import load_all_news_data


# Main keywords to track
KEYWORDS = [
    'ai', 'chatgpt', 'blockchain', 'bitcoin', 'startup',
    'funding', 'google', 'apple', 'cybersecurity', 'cloud'
]


def prepare_data(df):
    """Get text and dates ready"""
    df['text'] = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.lower()
    df['datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')
    df['date'] = df['datetime'].dt.date
    return df


def count_daily_keywords(df):
    """Count each keyword per day"""
    dates = sorted(df['date'].unique())
    results = []

    for date in dates:
        daily = df[df['date'] == date]
        row = {'date': date, 'total': len(daily)}

        for keyword in KEYWORDS:
            count = daily['text'].str.contains(keyword, case=False).sum()
            row[keyword] = count

        results.append(row)

    return pd.DataFrame(results)


def calculate_growth(trends_df):
    """Calculate growth rates for each keyword"""
    # Compare last 7 days to previous 7 days
    if len(trends_df) < 14:
        print("Not enough data for growth calculation")
        return {}

    growth_stats = {}

    for keyword in KEYWORDS:
        recent = trends_df[keyword].iloc[-7:].mean()
        previous = trends_df[keyword].iloc[-14:-7].mean()

        if previous > 0:
            growth_pct = ((recent - previous) / previous) * 100
        else:
            growth_pct = 0

        total_mentions = trends_df[keyword].sum()

        growth_stats[keyword] = {
            'total': int(total_mentions),
            'recent_avg': recent,
            'previous_avg': previous,
            'growth_pct': growth_pct
        }

    return growth_stats


def plot_trends(trends_df, growth_stats):
    """Make visualizations of keyword trends"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Line chart of trends over time
    dates = pd.to_datetime(trends_df['date'])
    colors = plt.cm.tab10(range(len(KEYWORDS)))

    for i, keyword in enumerate(KEYWORDS):
        ax1.plot(dates, trends_df[keyword], marker='o', markersize=3,
                label=keyword, linewidth=1.5, color=colors[i], alpha=0.7)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mentions')
    ax1.set_title('Keyword Trends Over Time', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Growth rate bar chart
    if growth_stats:
        keywords_sorted = sorted(growth_stats.items(),
                                key=lambda x: x[1]['growth_pct'],
                                reverse=True)
        keywords = [k for k, v in keywords_sorted]
        growth_rates = [v['growth_pct'] for k, v in keywords_sorted]

        colors_growth = ['green' if g > 0 else 'red' for g in growth_rates]
        ax2.barh(keywords, growth_rates, color=colors_growth, alpha=0.7)
        ax2.set_xlabel('Growth Rate (%)')
        ax2.set_title('Keyword Growth (Last 7 days vs Previous 7)', fontweight='bold')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, (k, rate) in enumerate(zip(keywords, growth_rates)):
            label = f'{rate:+.1f}%'
            x_pos = rate + (5 if rate > 0 else -5)
            ax2.text(x_pos, i, label, va='center', fontsize=8)

    plt.tight_layout()

    output = 'visuals/Keyword_Momentum.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def plot_scatter(growth_stats):
    """Scatter plot of total mentions vs growth"""
    if not growth_stats:
        return

    keywords = list(growth_stats.keys())
    totals = [growth_stats[k]['total'] for k in keywords]
    growth = [growth_stats[k]['growth_pct'] for k in keywords]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if g > 0 else 'red' for g in growth]
    ax.scatter(totals, growth, s=200, alpha=0.6, c=colors, edgecolors='black')

    # Label each point
    for i, keyword in enumerate(keywords):
        ax.annotate(keyword, (totals[i], growth[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Total Mentions')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_title('Keyword Popularity vs Growth', fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output = 'visuals/Growth_vs_Mentions.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*50)
    print("TREND DETECTION")
    print("="*50)

    df = load_all_news_data()
    if df.empty:
        print("No data")
        return

    df = prepare_data(df)
    trends_df = count_daily_keywords(df)

    print(f"\nTracking {len(KEYWORDS)} keywords over {len(trends_df)} days")

    growth_stats = calculate_growth(trends_df)

    if growth_stats:
        print("\nGrowth rates (recent 7 days vs previous 7 days):")
        for keyword, stats in sorted(growth_stats.items(),
                                     key=lambda x: x[1]['growth_pct'],
                                     reverse=True):
            sign = '+' if stats['growth_pct'] > 0 else ''
            print(f"  {keyword}: {sign}{stats['growth_pct']:.1f}% "
                  f"(total: {stats['total']} mentions)")

    plot_trends(trends_df, growth_stats)
    plot_scatter(growth_stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
