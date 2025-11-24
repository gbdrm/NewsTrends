# Tech News Trends

A Python data analysis project that collects tech news from multiple APIs and analyzes trends using Pandas, NumPy, and Matplotlib. Includes automated data collection, analysis scripts, visualizations, and an interactive Streamlit dashboard.

### Run the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Run Analysis Scripts

```bash
# Keyword frequency analysis
python -m src.keyword_analysis

# Source comparison
python -m src.source_analysis

# Time pattern analysis
python -m src.time_analysis

# Trend detection (growth rates)
python -m src.trend_detection

# Generate visual reports
python -m src.visual_reports
```
## Setup (Optional - for data collection)

The project includes pre-collected data, but you can set up your own data collection:

1. Get API keys:
   - [GNews API](https://gnews.io) - 100 requests/day free
   - [NewsAPI](https://newsapi.org) - 1000 requests/month free

2. Create `.env` file:
   ```
   GNEWS_API_KEY=your_key_here
   NEWS_API_KEY=your_key_here
   ```

3. Run data collection:
   ```bash
   python -m src.fetch
   ```

## Analysis Features

### Keyword Tracking
Monitors mentions of:
- AI & Machine Learning (ChatGPT, AI, ML)
- Cryptocurrency (Bitcoin, Blockchain)
- Business (Startups, Funding)
- Big Tech (Google, Apple, Microsoft, Amazon)
- Cloud & Development

### Interactive Dashboard
Three pages:
1. **Keyword Trends** - Track keywords over time with customizable selection
2. **Source Analysis** - See top publishers and their distribution
3. **Time Patterns** - Weekly and hourly publication patterns

### Visualizations
Automatically generates:
- Keyword frequency charts
- Source distribution comparisons
- Time trend graphs
- Growth rate analysis
- Multi-panel summary dashboards

## Tech Stack

- **Python 3.12**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Static visualizations
- **Streamlit** - Interactive web dashboard
- **GitHub Actions** - Automated daily collection

## Key Findings

See [FINDINGS.md](FINDINGS.md) for detailed analysis results, including:
- AI dominates tech news coverage
- Most active news sources
- Weekly publishing patterns
- Trend observations over time

## Development

This project demonstrates:
- Python data analysis workflow
- API integration and data collection
- Pandas dataframe operations
- Time series analysis
- Data visualization
- Interactive dashboard development

Built as a learning project to practice data science skills with real-world data.
