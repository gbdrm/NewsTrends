"""
Analyzing when articles are published - daily patterns, weekly patterns, etc.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.explore import load_all_news_data


def prepare_dates(df):
    """Parse dates and extract useful time info"""
    df['datetime'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['week'] = df['datetime'].dt.to_period('W')
    return df


def analyze_daily_volume(df):
    """Count articles per day"""
    daily = df.groupby('date').size()

    # Add rolling average
    daily_df = pd.DataFrame({'count': daily})
    daily_df['rolling_7day'] = daily_df['count'].rolling(window=7, center=True).mean()

    return daily_df


def analyze_weekly_pattern(df):
    """See which days of the week have most articles"""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
    return weekly


def plot_time_trends(daily_df, weekly):
    """Create visualizations of time patterns"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Daily volume over time
    dates = pd.to_datetime(daily_df.index)
    ax1.plot(dates, daily_df['count'], marker='o', linestyle='-',
             markersize=3, linewidth=1.5, label='Daily Count', alpha=0.7)

    # Add rolling average if we have enough data
    if not daily_df['rolling_7day'].isna().all():
        ax1.plot(dates, daily_df['rolling_7day'], linestyle='--',
                linewidth=2, label='7-day Average', color='red')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Articles')
    ax1.set_title('Daily Article Volume', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Weekly pattern
    ax2.bar(range(len(weekly)), weekly.values)
    ax2.set_xticks(range(len(weekly)))
    ax2.set_xticklabels([d[:3] for d in weekly.index])
    ax2.set_ylabel('Total Articles')
    ax2.set_title('Articles by Day of Week', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(weekly.values):
        ax2.text(i, v + max(weekly.values)*0.01, str(v), ha='center', va='bottom')

    plt.tight_layout()

    output = 'visuals/Time_Trends.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*50)
    print("TIME ANALYSIS")
    print("="*50)

    df = load_all_news_data()
    if df.empty:
        print("No data")
        return

    df = prepare_dates(df)

    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total days: {(df['date'].max() - df['date'].min()).days + 1}")

    daily_df = analyze_daily_volume(df)
    weekly = analyze_weekly_pattern(df)

    print("\nDaily stats:")
    print(f"  Average: {daily_df['count'].mean():.1f} articles/day")
    print(f"  Max: {daily_df['count'].max()} articles")
    print(f"  Min: {daily_df['count'].min()} articles")

    print("\nMost active day:", weekly.idxmax())
    print("Least active day:", weekly.idxmin())

    plot_time_trends(daily_df, weekly)

    print("\nDone!")


if __name__ == "__main__":
    main()
