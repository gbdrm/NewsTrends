"""
Looking at which keywords show up most in tech news
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.explore import load_all_news_data

# Keywords we want to track
KEYWORDS = {
    'AI & ML': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'openai'],
    'Blockchain': ['blockchain', 'bitcoin', 'crypto', 'ethereum', 'nft'],
    'Business': ['startup', 'funding', 'investment', 'ipo'],
    'Big Tech': ['google', 'apple', 'microsoft', 'amazon', 'meta', 'tesla', 'nvidia'],
    'Development': ['python', 'javascript', 'api', 'cloud', 'aws', 'github'],
    'Other Tech': ['quantum', 'robotics', '5g', 'cybersecurity']
}


def prepare_text(df):
    """Combine title and description into one searchable text field"""
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['text'] = df['text'].str.lower()
    return df


def count_keywords(df):
    """Count how many times each keyword category appears"""
    counts = {}

    for category, keywords in KEYWORDS.items():
        total = 0
        for keyword in keywords:
            # Count articles that mention this keyword
            total += df['text'].str.contains(keyword, case=False, na=False).sum()
        counts[category] = total

    return counts


def plot_keyword_frequencies(counts):
    """Make a bar chart of keyword frequencies"""
    categories = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(range(len(categories)))
    ax.barh(categories, values, color=colors)
    ax.set_xlabel('Number of Mentions')
    ax.set_title('Keyword Frequency in Tech News', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add counts on bars
    for i, v in enumerate(values):
        ax.text(v + max(values)*0.01, i, str(v), va='center')

    plt.tight_layout()

    output = 'visuals/Category_Frequencies.png'
    os.makedirs('visuals', exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    print("\n" + "="*50)
    print("KEYWORD ANALYSIS")
    print("="*50)

    df = load_all_news_data()
    if df.empty:
        print("No data")
        return

    df = prepare_text(df)
    counts = count_keywords(df)

    print("\nKeyword mentions:")
    for cat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    plot_keyword_frequencies(counts)

    print("\nDone!")


if __name__ == "__main__":
    main()
