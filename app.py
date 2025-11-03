"""
Simple dashboard to look at tech news keyword trends over time.
Just showing what keywords show up in tech news articles.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.explore import load_all_news_data

# Basic page setup
st.set_page_config(page_title="Tech News Keywords", layout="wide")

st.title("Tech News Keyword Trends")
st.write("Simple dashboard showing how often different tech keywords appear in news articles")

# Load the data
@st.cache_data
def get_data():
    df = load_all_news_data()

    # Basic date parsing
    df['date'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df['date_only'] = df['date'].dt.date

    # Combine title and description for searching
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['text'] = df['text'].str.lower()

    return df

df = get_data()

st.sidebar.header("Options")

# Simple date filter
min_date = df['date_only'].min()
max_date = df['date_only'].max()

st.sidebar.write("**Date Range**")
start_date = st.sidebar.date_input("Start", min_date)
end_date = st.sidebar.date_input("End", max_date)

# Filter data by date
filtered_df = df[(df['date_only'] >= start_date) & (df['date_only'] <= end_date)]

st.sidebar.write(f"Showing {len(filtered_df)} articles")

# Keywords to track - just the main ones
keywords = {
    'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt'],
    'Crypto': ['bitcoin', 'crypto', 'blockchain'],
    'Startup': ['startup', 'funding'],
    'Apple': ['apple', 'iphone', 'ipad'],
    'Google': ['google', 'android'],
}

# Let user pick which keywords to show
st.sidebar.write("**Select Keywords**")
selected = []
for name in keywords.keys():
    if st.sidebar.checkbox(name, value=True):
        selected.append(name)

if not selected:
    st.warning("Please select at least one keyword")
    st.stop()

# Count keywords per day
st.subheader("Keyword Mentions Over Time")

dates = sorted(filtered_df['date_only'].unique())
results = []

for date in dates:
    day_df = filtered_df[filtered_df['date_only'] == date]
    row = {'date': date}

    for name in selected:
        # Count how many articles mention these keywords
        count = 0
        for keyword in keywords[name]:
            count += day_df['text'].str.contains(keyword, case=False).sum()
        row[name] = count

    results.append(row)

trend_df = pd.DataFrame(results)

# Plot the trends
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i, name in enumerate(selected):
    ax.plot(pd.to_datetime(trend_df['date']), trend_df[name],
            marker='o', label=name, linewidth=2,
            color=colors[i % len(colors)])

ax.set_xlabel('Date')
ax.set_ylabel('Number of Mentions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

st.pyplot(fig)
plt.close()

# Show some basic stats
st.subheader("Summary Stats")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Articles", len(filtered_df))

with col2:
    st.metric("Date Range", f"{(end_date - start_date).days + 1} days")

with col3:
    most_mentioned = max(selected, key=lambda x: trend_df[x].sum())
    st.metric("Most Mentioned", most_mentioned)

# Show detailed counts
st.subheader("Detailed Counts")

stats = {}
for name in selected:
    total = trend_df[name].sum()
    avg = trend_df[name].mean()
    stats[name] = {'Total Mentions': int(total), 'Avg per Day': f"{avg:.1f}"}

stats_df = pd.DataFrame(stats).T
st.dataframe(stats_df)

# Optional: show raw data
if st.checkbox("Show raw trend data"):
    st.write(trend_df)
