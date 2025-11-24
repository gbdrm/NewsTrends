"""
Simple dashboard to explore tech news trends.
Shows keyword trends, sources, and time patterns.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.explore import load_all_news_data

st.set_page_config(page_title="Tech News Dashboard", layout="wide")

# Load data
@st.cache_data
def get_data():
    df = load_all_news_data()

    df['date'] = pd.to_datetime(df['collection_date'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df['date_only'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()

    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['text'] = df['text'].str.lower()

    # Get source names
    if 'source.name' in df.columns:
        df['source_name'] = df['source.name']
    elif 'source' in df.columns:
        df['source_name'] = df['source'].apply(
            lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else str(x)
        )
    else:
        df['source_name'] = 'Unknown'

    return df

df = get_data()

# Sidebar
st.sidebar.title("Filters")

min_date = df['date_only'].min()
max_date = df['date_only'].max()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

filtered_df = df[(df['date_only'] >= start_date) & (df['date_only'] <= end_date)]

st.sidebar.metric("Articles", len(filtered_df))
st.sidebar.metric("Sources", filtered_df['source_name'].nunique())

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio("Select Page", ["Overview", "Keyword Trends", "Source Analysis", "Time Patterns"])

# Main content
st.title("Tech News Dashboard")

# Page 0: Overview
if page == "Overview":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", f"{len(filtered_df):,}")
    with col2:
        st.metric("News Sources", filtered_df['source_name'].nunique())
    with col3:
        days = (end_date - start_date).days + 1
        st.metric("Days Collected", days)
    with col4:
        avg_per_day = len(filtered_df) / days if days > 0 else 0
        st.metric("Avg/Day", f"{avg_per_day:.1f}")

    # Content quality metrics
    st.subheader("Content Quality")

    col1, col2 = st.columns(2)

    with col1:
        # Quality metrics
        has_desc = filtered_df['description'].notna().sum()
        has_image = filtered_df['urlToImage'].notna().sum() if 'urlToImage' in filtered_df.columns else 0
        has_content = filtered_df['content'].notna().sum() if 'content' in filtered_df.columns else 0

        quality_data = {
            'Metric': ['Has Description', 'Has Image', 'Has Content'],
            'Count': [has_desc, has_image, has_content],
            'Percentage': [
                has_desc / len(filtered_df) * 100,
                has_image / len(filtered_df) * 100,
                has_content / len(filtered_df) * 100
            ]
        }

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(quality_data['Metric'], quality_data['Percentage'], color=['#3498db', '#2ecc71', '#f39c12'])
        ax.set_xlabel('% of Articles')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{quality_data["Percentage"][i]:.1f}%',
                   ha='left', va='center', fontsize=9)

        st.pyplot(fig)
        plt.close()

    with col2:
        # Top keywords snapshot
        keywords = {
            'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt'],
            'Crypto': ['bitcoin', 'crypto', 'blockchain'],
            'Startup': ['startup', 'funding'],
            'Apple': ['apple', 'iphone'],
            'Google': ['google', 'android'],
        }

        keyword_counts = {}
        for name, terms in keywords.items():
            count = 0
            for term in terms:
                count += filtered_df['text'].str.contains(term, case=False).sum()
            keyword_counts[name] = count

        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(keyword_counts.keys())
        values = list(keyword_counts.values())
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        bars = ax.bar(names, values, color=colors, alpha=0.7)
        ax.set_ylabel('Mentions')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)

        st.pyplot(fig)
        plt.close()

    # Article length distribution
    st.subheader("Article Text Lengths")

    filtered_df['title_len'] = filtered_df['title'].fillna('').str.len()
    filtered_df['desc_len'] = filtered_df['description'].fillna('').str.len()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filtered_df['title_len'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Title Length (characters)')
        ax.set_ylabel('Count')
        ax.axvline(filtered_df['title_len'].mean(), color='red', linestyle='--',
                   label=f"Avg: {filtered_df['title_len'].mean():.0f}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filtered_df['desc_len'], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Description Length (characters)')
        ax.set_ylabel('Count')
        ax.axvline(filtered_df['desc_len'].mean(), color='red', linestyle='--',
                   label=f"Avg: {filtered_df['desc_len'].mean():.0f}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

# Page 1: Keyword Trends
if page == "Keyword Trends":
    st.header("Keyword Trends")

    keywords = {
        'AI': ['ai', 'artificial intelligence', 'machine learning', 'chatgpt'],
        'Crypto': ['bitcoin', 'crypto', 'blockchain'],
        'Startup': ['startup', 'funding'],
        'Apple': ['apple', 'iphone'],
        'Google': ['google', 'android'],
    }

    st.write("**Select keywords to track:**")
    cols = st.columns(len(keywords))
    selected = []
    for i, name in enumerate(keywords.keys()):
        with cols[i]:
            if st.checkbox(name, value=True):
                selected.append(name)

    if not selected:
        st.warning("Select at least one keyword")
        st.stop()

    # Count keywords per day
    dates = sorted(filtered_df['date_only'].unique())
    results = []

    for date in dates:
        day_df = filtered_df[filtered_df['date_only'] == date]
        row = {'date': date}

        for name in selected:
            count = 0
            for keyword in keywords[name]:
                count += day_df['text'].str.contains(keyword, case=False).sum()
            row[name] = count

        results.append(row)

    trend_df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, name in enumerate(selected):
        ax.plot(pd.to_datetime(trend_df['date']), trend_df[name],
                marker='o', label=name, linewidth=2, markersize=4,
                color=colors[i % len(colors)])

    ax.set_xlabel('Date')
    ax.set_ylabel('Mentions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    st.pyplot(fig)
    plt.close()

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(filtered_df))
    with col2:
        st.metric("Days", f"{(end_date - start_date).days + 1}")
    with col3:
        most_mentioned = max(selected, key=lambda x: trend_df[x].sum())
        st.metric("Most Mentioned", most_mentioned)

    # Details
    with st.expander("Show detailed stats"):
        stats = {}
        for name in selected:
            total = trend_df[name].sum()
            avg = trend_df[name].mean()
            stats[name] = {'Total': int(total), 'Avg/Day': f"{avg:.1f}"}
        st.dataframe(pd.DataFrame(stats).T)

# Page 2: Source Analysis
elif page == "Source Analysis":
    st.header("Source Analysis")

    source_counts = filtered_df['source_name'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Sources")
        top_n = st.slider("Number of sources to show", 5, 20, 10)

        top_sources = source_counts.head(top_n)

        fig, ax = plt.subplots(figsize=(6, max(4, top_n*0.3)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_sources)))

        ax.barh(range(len(top_sources)), top_sources.values, color=colors)
        ax.set_yticks(range(len(top_sources)))
        ax.set_yticklabels([s[:30] for s in top_sources.index], fontsize=9)
        ax.set_xlabel('Articles')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Source Distribution")

        # Pie chart
        top5 = source_counts.head(5).sum()
        next10 = source_counts.iloc[5:15].sum()
        rest = source_counts.iloc[15:].sum()

        fig, ax = plt.subplots(figsize=(6, 6))
        sizes = [top5, next10, rest]
        labels = [f'Top 5\n{top5}', f'Next 10\n{next10}', f'Others\n{rest}']
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Article Distribution')

        st.pyplot(fig)
        plt.close()

    # Source stats table
    with st.expander("Show all sources"):
        source_df = pd.DataFrame({
            'Source': source_counts.index,
            'Articles': source_counts.values,
            'Percentage': (source_counts.values / len(filtered_df) * 100).round(2)
        })
        st.dataframe(source_df, height=400)

# Page 3: Time Patterns
elif page == "Time Patterns":
    st.header("Time Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Articles by Day of Week")

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = filtered_df['day_of_week'].value_counts().reindex(day_order, fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(day_counts)))

        bars = ax.bar(range(len(day_counts)), day_counts.values, color=colors)
        ax.set_xticks(range(len(day_counts)))
        ax.set_xticklabels([d[:3] for d in day_order])
        ax.set_ylabel('Articles')
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)

        st.pyplot(fig)
        plt.close()

        st.metric("Most Active Day", day_counts.idxmax())

    with col2:
        st.subheader("Daily Volume Over Time")

        daily = filtered_df.groupby('date_only').size()

        fig, ax = plt.subplots(figsize=(8, 5))
        dates = pd.to_datetime(daily.index)

        ax.plot(dates, daily.values, marker='o', linewidth=2, markersize=3)

        # Add 7-day average if enough data
        if len(daily) >= 7:
            rolling = pd.Series(daily.values).rolling(window=7, center=True).mean()
            ax.plot(dates, rolling, '--', linewidth=2, label='7-day avg', color='red')
            ax.legend()

        ax.set_xlabel('Date')
        ax.set_ylabel('Articles')
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)

        st.pyplot(fig)
        plt.close()

        st.metric("Avg Articles/Day", f"{daily.mean():.1f}")
