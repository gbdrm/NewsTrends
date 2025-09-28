import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import requests

from . import config


def fetch_newsapi(query: str, page_size: int = 50, language: str = "en") -> Dict:
    """
    Fetch headlines from NewsAPI.
    """
    headers = {"X-API-Key": config.NEWS_API_KEY}
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "language": language,
    }
    
    response = requests.get(
        f"{config.NEWS_API_BASE}/everything",
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_gnews(query: str, page_size: int = 50, language: str = "en") -> Dict:
    """
    Fetch headlines from GNews API.
    """
    params = {
        "q": query,
        "lang": language,
        "max": min(page_size, 100),  # GNews max is 100
        "token": config.GNEWS_API_KEY,
    }
    
    response = requests.get(
        f"{config.GNEWS_API_BASE}/search",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def normalize_articles(raw_response: Dict, provider: str) -> List[Dict]:
    """
    Normalize article data from different providers into a common schema.
    """
    if provider == "newsapi":
        articles = raw_response.get("articles", [])
        return [
            {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "author": article.get("author", ""),
                "image_url": article.get("urlToImage", ""),
                "content": article.get("content", "")
            }
            for article in articles
        ]
    elif provider == "gnews":
        articles = raw_response.get("articles", [])
        return [
            {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", ""),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
                "author": "",  # GNews doesn't provide author
                "image_url": article.get("image", ""),
                "content": article.get("content", "")
            }
            for article in articles
        ]
    else:  # stub
        return [
            {
                "title": "AI Breakthrough in Machine Learning",
                "description": "Researchers achieve new milestone in artificial intelligence.",
                "source": "TechStub",
                "url": "https://example.com/ai-breakthrough",
                "published_at": "2024-01-15T10:00:00Z",
                "author": "Tech Reporter",
                "image_url": "https://example.com/image1.jpg",
                "content": "Stub content for AI breakthrough news..."
            },
            {
                "title": "Cloud Computing Trends for 2024",
                "description": "Latest developments in cloud infrastructure.",
                "source": "TechStub",
                "url": "https://example.com/cloud-trends",
                "published_at": "2024-01-15T09:30:00Z",
                "author": "Cloud Expert",
                "image_url": "https://example.com/image2.jpg",
                "content": "Stub content for cloud computing trends..."
            },
            {
                "title": "Cybersecurity Best Practices Updated",
                "description": "New guidelines for digital security released.",
                "source": "TechStub",
                "url": "https://example.com/security-practices",
                "published_at": "2024-01-15T09:00:00Z",
                "author": "Security Analyst",
                "image_url": "https://example.com/image3.jpg",
                "content": "Stub content for cybersecurity practices..."
            },
            {
                "title": "Mobile Development Framework Released",
                "description": "New cross-platform framework simplifies app development.",
                "source": "TechStub",
                "url": "https://example.com/mobile-framework",
                "published_at": "2024-01-15T08:30:00Z",
                "author": "Mobile Dev",
                "image_url": "https://example.com/image4.jpg",
                "content": "Stub content for mobile development..."
            },
            {
                "title": "Blockchain Technology Advances",
                "description": "Latest innovations in distributed ledger technology.",
                "source": "TechStub",
                "url": "https://example.com/blockchain-advances",
                "published_at": "2024-01-15T08:00:00Z",
                "author": "Blockchain Writer",
                "image_url": "https://example.com/image5.jpg",
                "content": "Stub content for blockchain technology..."
            }
        ]


def fetch_headlines(query: str, page_size: int = 50, language: str = "en") -> Dict:
    """
    Fetch headlines from the appropriate provider based on configuration.
    """
    provider_used = "stub"
    raw_response = {}
    
    # Determine which provider to use
    if config.NEWS_PROVIDER == "gnews" and config.GNEWS_API_KEY:
        provider_used = "gnews"
        raw_response = fetch_gnews(query, page_size, language)
    elif config.NEWS_PROVIDER == "newsapi" and config.NEWS_API_KEY:
        provider_used = "newsapi"
        raw_response = fetch_newsapi(query, page_size, language)
    elif config.NEWS_PROVIDER == "auto":
        if config.GNEWS_API_KEY:
            provider_used = "gnews"
            raw_response = fetch_gnews(query, page_size, language)
        elif config.NEWS_API_KEY:
            provider_used = "newsapi"
            raw_response = fetch_newsapi(query, page_size, language)
    
    # Normalize the response
    normalized_articles = normalize_articles(raw_response, provider_used)
    
    # Return in a consistent format
    return {
        "provider": provider_used,
        "status": "ok",
        "totalResults": len(normalized_articles) if provider_used == "stub" else raw_response.get("totalResults", len(normalized_articles)),
        "articles": normalized_articles,
        "raw_response": raw_response  # Keep original for debugging
    }


def save_raw(payload: Dict, tag: str = "tech", provider: str = None) -> str:
    """
    Save raw JSON payload to appropriate directory with timestamp.
    Returns the file path.
    """
    # Determine save directory based on provider
    if provider == "gnews":
        save_dir = config.GNEWS_RAW_DIR
    elif provider == "newsapi":
        save_dir = config.NEWSAPI_RAW_DIR
    else:
        save_dir = config.RAW_DIR  # Legacy/fallback
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{tag}.json"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    return filepath


def collect_from_both_providers(query: str, page_size: int = 50, language: str = "en") -> List[Dict]:
    """
    Collect data from both providers when both API keys are available.
    Returns list of results with provider info.
    """
    results = []
    
    # Collect from GNews if key available
    if config.GNEWS_API_KEY:
        try:
            raw_response = fetch_gnews(query, page_size, language)
            normalized_articles = normalize_articles(raw_response, "gnews")
            result = {
                "provider": "gnews",
                "status": "ok",
                "totalResults": raw_response.get("totalResults", len(normalized_articles)),
                "articles": normalized_articles,
                "raw_response": raw_response
            }
            results.append(result)
            print(f"GNews: Fetched {len(normalized_articles)} articles")
        except Exception as e:
            print(f"GNews API error: {e}")
    
    # Collect from NewsAPI if key available
    if config.NEWS_API_KEY:
        try:
            raw_response = fetch_newsapi(query, page_size, language)
            normalized_articles = normalize_articles(raw_response, "newsapi")
            result = {
                "provider": "newsapi",
                "status": "ok",
                "totalResults": raw_response.get("totalResults", len(normalized_articles)),
                "articles": normalized_articles,
                "raw_response": raw_response
            }
            results.append(result)
            print(f"NewsAPI: Fetched {len(normalized_articles)} articles")
        except Exception as e:
            print(f"NewsAPI error: {e}")
    
    return results


if __name__ == "__main__":
    if config.COLLECT_BOTH and (config.GNEWS_API_KEY or config.NEWS_API_KEY):
        # Collect from both providers
        print("Collecting from multiple providers...")
        results = collect_from_both_providers(config.NEWS_QUERY)
        
        if not results:
            print("No data collected from any provider")
        else:
            for result in results:
                provider = result["provider"]
                articles = result["articles"]
                filepath = save_raw(result, provider=provider)
                print(f"Saved {len(articles)} {provider} articles to {filepath}")
                
                if provider != "stub":
                    print(f"  Total {provider} results available: {result.get('totalResults', 0)}")
    else:
        # Single provider collection (legacy behavior)
        print("Collecting from single provider...")
        headlines = fetch_headlines(config.NEWS_QUERY)
        provider = headlines.get("provider", "unknown")
        filepath = save_raw(headlines, provider=provider)
        
        article_count = headlines.get("totalResults", 0)
        articles_list = headlines.get("articles", [])
        actual_count = len(articles_list)
        
        print(f"Saved {actual_count} articles to {filepath}")
        print(f"Provider used: {provider}")
        
        if provider == "stub":
            print("Using stub data (no API key provided)")
        else:
            print(f"Total results available: {article_count}")