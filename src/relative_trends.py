"""
Analyzing keyword trends as percentages instead of raw counts
This helps when the number of articles per day changes
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.explore import load_all_news_data


KEYWORDS = ['ai', 'chatgpt', 'bitcoin', 'startup', 'cloud', 'cybersecurity']


def prepare_data(df):
    """Get text and dates ready"""
    df['text'] = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.lower()
    df['datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')
    df['date'] = df['datetime'].dt.date
    return df


def calc_relative_trends(df):
    """Calculate keywords as % of daily articles"""
    dates = sorted(df['date'].unique())
    results = []

    for date in dates:
        daily = df[df['date'] == date]
        total_articles = len(daily)

        row = {'date': date, 'total_articles': total_articles}

        for keyword in KEYWORDS:
            count = daily['text'].str.contains(keyword, case=False).sum()
            # Calculate as percentage
            pct = (count / total_articles * 100) if total_articles > 0 else 0
            row[f'{keyword}_pct'] = pct
            row[f'{keyword}_count'] = count

        results.append(row)

    return pd.DataFrame(results)


def plot_relative_trends(trends_df):
    """Plot percentage trends"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    dates = pd.to_datetime(trends_df['date'])
    colors = plt.cm.tab10(range(len(KEYWORDS)))

    # Percentage trends
    for i, keyword in enumerate(KEYWORDS):
        ax1.plot(dates, trends_df[f'{keyword}_pct'], marker='o', markersize=3,
                label=keyword, linewidth=1.5, color=colors[i], alpha=0.7)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('% of Daily Articles')
    ax1.set_title('Keyword Trends (as % of articles)', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Daily article volume for reference
    ax2.plot(dates, trends_df['total_articles'], marker='o',
            linewidth=2, markersize=3, color='gray')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Articles')
    ax2.set_title('Daily Article Volume (for reference)', fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    output = 'visuals/Relative_Trends.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*50)
    print("RELATIVE TREND ANALYSIS")
    print("="*50)

    df = load_all_news_data()
    if df.empty:
        print("No data")
        return

    df = prepare_data(df)
    trends_df = calc_relative_trends(df)

    print(f"\nAnalyzing {len(KEYWORDS)} keywords over {len(trends_df)} days")
    print("\nAverage mention rates (% of articles):")
    for keyword in KEYWORDS:
        avg_pct = trends_df[f'{keyword}_pct'].mean()
        print(f"  {keyword}: {avg_pct:.2f}%")

    plot_relative_trends(trends_df)

    print("\nDone!")


if __name__ == "__main__":
    main()
