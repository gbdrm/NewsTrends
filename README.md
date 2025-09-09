# Tech News Trends

A minimal prototype for demonstrating a working pipeline that fetches technology news headlines from multiple APIs, saves raw JSON responses with timestamps, and runs daily via GitHub Actions. The system supports NewsAPI and GNews API, with graceful fallback to stub data when no API keys are available.

## API Provider Selection

The system automatically selects the best available news provider in this order:

1. **GNews API** - If `GNEWS_API_KEY` is available (preferred for more generous rate limits)
2. **NewsAPI** - If `NEWS_API_KEY` is available
3. **Stub Data** - Fallback when no API keys are configured

You can force a specific provider using the `NEWS_PROVIDER` environment variable:
- `NEWS_PROVIDER=gnews` - Force GNews API
- `NEWS_PROVIDER=newsapi` - Force NewsAPI  
- `NEWS_PROVIDER=auto` - Automatic selection (default)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```

5. (Optional) Add your API keys to `.env`:
   ```
   # GNews API (recommended - higher rate limits)
   GNEWS_API_KEY=your_gnews_api_key_here
   
   # NewsAPI (alternative)
   NEWS_API_KEY=your_newsapi_key_here
   
   # Force specific provider (optional)
   NEWS_PROVIDER=auto
   ```

6. Run the fetcher:
   ```bash
   python -m src.fetch
   ```

## API Key Setup

### GNews API (Recommended)
- Sign up at [gnews.io](https://gnews.io)
- Free tier: 100 requests/day
- Better content quality and fewer restrictions

### NewsAPI
- Sign up at [newsapi.org](https://newsapi.org)
- Free tier: 1000 requests/month
- More comprehensive but limited content in free tier