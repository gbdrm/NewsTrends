import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
NEWS_API_BASE = os.getenv("NEWS_API_BASE", "https://newsapi.org/v2")
GNEWS_API_BASE = os.getenv("GNEWS_API_BASE", "https://gnews.io/api/v4")
NEWS_QUERY = os.getenv("NEWS_QUERY", "technology")
NEWS_PROVIDER = os.getenv("NEWS_PROVIDER", "auto")
COLLECT_BOTH = os.getenv("COLLECT_BOTH", "true").lower() == "true"

RAW_DIR = "data/raw"
GNEWS_RAW_DIR = "data/raw/gnews"
NEWSAPI_RAW_DIR = "data/raw/newsapi"