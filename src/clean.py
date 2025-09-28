import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Set
from collections import defaultdict
import hashlib

from . import config


def normalize_date(date_str: str) -> str:
    """
    Normalize various date formats to ISO format.
    """
    if not date_str:
        return ""
    
    try:
        # Handle various date formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Convert to UTC if timezone aware
                if dt.tzinfo:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
        
        # If no format matches, return original
        return date_str
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common truncation indicators
    text = re.sub(r'\[\+?\d+\s*chars?\]$', '', text)
    text = re.sub(r'\.\.\.$', '', text)
    
    # Remove HTML entities (basic)
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    return text.strip()


def is_tech_related(article: Dict) -> bool:
    """
    Determine if an article is technology-related based on content.
    """
    tech_keywords = {
        # Core tech terms
        'technology', 'tech', 'software', 'hardware', 'computing', 'computer',
        'digital', 'internet', 'web', 'online', 'cyber', 'data', 'algorithm',
        
        # AI/ML
        'artificial intelligence', 'ai', 'machine learning', 'ml', 'neural',
        'deep learning', 'automation', 'robotics', 'bot',
        
        # Programming/Development
        'programming', 'coding', 'developer', 'development', 'framework',
        'api', 'database', 'cloud', 'server', 'platform',
        
        # Devices/Products
        'smartphone', 'iphone', 'android', 'app', 'mobile', 'tablet',
        'laptop', 'pc', 'gaming', 'vr', 'ar', 'virtual reality',
        
        # Companies/Platforms
        'microsoft', 'google', 'apple', 'amazon', 'meta', 'tesla',
        'nvidia', 'intel', 'amd', 'github', 'openai',
        
        # Emerging tech
        'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'nft',
        'quantum', 'iot', 'internet of things', '5g', '6g',
        
        # IT/Security
        'cybersecurity', 'security', 'privacy', 'encryption', 'hacking',
        'startup', 'fintech', 'edtech', 'healthtech'
    }
    
    # Combine title, description, and content for analysis
    text_content = ' '.join([
        article.get('title', ''),
        article.get('description', ''),
        article.get('content', '')[:200]  # First 200 chars of content
    ]).lower()
    
    # Check if any tech keywords are present
    return any(keyword in text_content for keyword in tech_keywords)


def generate_content_hash(article: Dict) -> str:
    """
    Generate a hash based on article content for duplicate detection.
    """
    # Use title + first part of description for hashing
    content = f"{article.get('title', '')}{article.get('description', '')[:100]}"
    content = re.sub(r'\s+', ' ', content.lower().strip())
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def remove_duplicates(articles: List[Dict]) -> List[Dict]:
    """
    Remove duplicate articles based on content similarity.
    """
    seen_hashes = set()
    unique_articles = []
    
    for article in articles:
        content_hash = generate_content_hash(article)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_articles.append(article)
    
    return unique_articles


def clean_article(article: Dict) -> Dict:
    """
    Clean a single article's data.
    """
    # Handle source field that might be dict or string
    source_raw = article.get('source', '')
    if isinstance(source_raw, dict):
        source = source_raw.get('name', '')
    else:
        source = source_raw or ''
    
    cleaned = {
        'title': clean_text(article.get('title', '')),
        'description': clean_text(article.get('description', '')),
        'source': clean_text(source),
        'url': article.get('url', '').strip() if article.get('url') else '',
        'published_at': normalize_date(article.get('published_at', '') or article.get('publishedAt', '')),
        'author': clean_text(article.get('author', '')),
        'image_url': article.get('image_url', '').strip() if article.get('image_url') else article.get('urlToImage', ''),
        'content': clean_text(article.get('content', '')),
    }
    
    # Add metadata
    cleaned['_provider'] = article.get('_provider', '')
    cleaned['_file'] = article.get('_file', '')
    cleaned['_is_tech'] = is_tech_related(cleaned)
    cleaned['_content_hash'] = generate_content_hash(cleaned)
    
    return cleaned


def validate_article(article: Dict) -> bool:
    """
    Validate that an article has minimum required fields.
    """
    required_fields = ['title', 'url']
    return all(article.get(field, '').strip() for field in required_fields)


def load_raw_data() -> List[Dict]:
    """
    Load all raw data files from legacy and provider-specific directories.
    """
    all_articles = []
    
    # Define directories to scan
    directories = [
        (config.RAW_DIR, "legacy"),  # Legacy files in root raw directory
        (config.GNEWS_RAW_DIR, "gnews"),  # GNews provider directory
        (config.NEWSAPI_RAW_DIR, "newsapi")  # NewsAPI provider directory
    ]
    
    for directory, source_type in directories:
        if not os.path.exists(directory):
            continue
            
        files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
        print(f"Loading {len(files)} files from {directory}")
        
        for filename in files:
            try:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                provider = data.get('provider', source_type)  # Use directory name as fallback
                articles = data.get('articles', [])
                
                for article in articles:
                    # Add metadata
                    article['_file'] = filename
                    article['_provider'] = provider
                    article['_source_dir'] = directory
                    all_articles.append(article)
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
    
    return all_articles


def clean_data() -> Dict:
    """
    Main cleaning function that processes all raw data.
    """
    print("Loading raw data...")
    raw_articles = load_raw_data()
    print(f"Loaded {len(raw_articles)} raw articles")
    
    print("Cleaning articles...")
    cleaned_articles = [clean_article(article) for article in raw_articles]
    
    print("Validating articles...")
    valid_articles = [a for a in cleaned_articles if validate_article(a)]
    print(f"Valid articles: {len(valid_articles)}/{len(cleaned_articles)}")
    
    print("Removing duplicates...")
    unique_articles = remove_duplicates(valid_articles)
    print(f"Unique articles: {len(unique_articles)}/{len(valid_articles)}")
    
    print("Filtering tech articles...")
    tech_articles = [a for a in unique_articles if a['_is_tech']]
    print(f"Tech articles: {len(tech_articles)}/{len(unique_articles)}")
    
    # Generate summary statistics
    stats = {
        'total_raw_articles': len(raw_articles),
        'total_cleaned_articles': len(cleaned_articles),
        'total_valid_articles': len(valid_articles),
        'total_unique_articles': len(unique_articles),
        'total_tech_articles': len(tech_articles),
        'duplicates_removed': len(valid_articles) - len(unique_articles),
        'non_tech_filtered': len(unique_articles) - len(tech_articles),
        'sources': list(set(a['source'] for a in tech_articles if a['source'])),
        'date_range': {
            'start': min(a['published_at'] for a in tech_articles if a['published_at']),
            'end': max(a['published_at'] for a in tech_articles if a['published_at'])
        },
        'providers': list(set(a['_provider'] for a in tech_articles)),
        'processed_at': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    
    return {
        'articles': tech_articles,
        'all_articles': unique_articles,  # Include non-tech for reference
        'stats': stats
    }


def save_cleaned_data(cleaned_data: Dict) -> str:
    """
    Save cleaned data to the cleaned directory.
    """
    os.makedirs('data/cleaned', exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = f"data/cleaned/{timestamp}_cleaned.json"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    return filepath


if __name__ == "__main__":
    print("Starting data cleaning process...")
    
    cleaned_data = clean_data()
    filepath = save_cleaned_data(cleaned_data)
    
    stats = cleaned_data['stats']
    print(f"\nCleaning completed!")
    print(f"Saved to: {filepath}")
    print(f"\nSummary:")
    print(f"  Raw articles: {stats['total_raw_articles']}")
    print(f"  Tech articles: {stats['total_tech_articles']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Non-tech filtered: {stats['non_tech_filtered']}")
    print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"  Sources: {len(stats['sources'])}")
    print(f"  Providers: {stats['providers']}")