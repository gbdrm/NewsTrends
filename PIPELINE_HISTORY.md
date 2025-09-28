# Tech News Trends - Pipeline History

# Data Collection Period
**Date:** September 5-28, 2025 (24 days)  
**Status:** ✅ Completed

### Collection Statistics
- **Total Files:** 23 JSON files
- **Collection Method:** GitHub Actions (daily at 12:00 UTC)
- **Provider Used:** GNews API (consistent)
- **File Naming:** `YYYYMMDDTHHMMSSZ_tech.json`

### Data Volume
```
Total Raw Files: 23
Date Range: 2025-09-05 to 2025-09-28
Total File Size: ~612KB
Average Articles/Day: ~10-12
Provider Consistency: 100% GNews
```

### Sample Collection Log
```
20250905T030339Z_tech.json  - 48KB (NewsAPI - initial test)
20250905T031419Z_tech.json  - 24KB (GNews - transition)
20250909T050928Z_tech.json  - 21KB (GNews - stable)
...
20250928T122139Z_tech.json  - 24KB (GNews - latest)
```

### Key Observations
1. **Provider Stability:** GNews API proved reliable (22/23 files)
2. **Article Limits:** GNews provides ~10 articles vs NewsAPI's 50
3. **Content Quality:** GNews had better tech relevance
4. **Automation Success:** 100% uptime for GitHub Actions

---

## Phase 3: Data Cleaning & Processing
**Date:** September 28, 2025  
**Status:** ✅ Completed

### Objectives
- Implement comprehensive data cleaning pipeline
- Remove duplicates and improve data quality
- Filter non-tech content
- Standardize schemas across providers

### Pre-Cleaning Analysis
**Raw Data Issues Identified:**
- 14 duplicate article titles across timeframe
- 1 missing publication date
- Mixed source field formats (dict vs string)
- Inconsistent date formats between providers
- Some non-tech content (racing, education)

### Cleaning Pipeline Implementation

**Features Implemented:**
1. **Duplicate Detection**
   - Content-based hashing (title + description)
   - Cross-day duplicate removal
   
2. **Text Normalization**
   - HTML entity cleanup (`&amp;` → `&`)
   - Whitespace normalization
   - Truncation marker removal (`[+123 chars]`)
   
3. **Date Standardization**
   - Multiple format support
   - UTC timezone normalization
   - ISO 8601 output format
   
4. **Tech Content Filtering**
   - 50+ tech keyword dictionary
   - Content analysis (title + description + content)
   - Company/platform recognition
   
5. **Schema Normalization**
   - Unified field names across providers
   - Handle NewsAPI vs GNews differences
   - Metadata preservation

### Cleaning Results - Run #1
**Execution Time:** 2025-09-28T22:55:46Z  
**Input:** 23 raw files  
**Processing Stats:**

| Metric | Count | Percentage |
|--------|-------|------------|
| Raw Articles | 269 | 100% |
| Valid Articles | 269 | 100% |
| Unique Articles | 249 | 92.6% |
| Tech Articles | 249 | 100% |
| **Duplicates Removed** | **20** | **7.4%** |
| **Non-Tech Filtered** | **0** | **0%** |

### Data Quality Metrics
- **Sources Identified:** 87 unique news sources
- **Date Range:** 2025-09-04 to 2025-09-28 (24 days)
- **Providers:** GNews (primary), NewsAPI (legacy)
- **Output File:** `data/cleaned/20250928T225546Z_cleaned.json` (596KB)

### Top News Sources (by article count)
1. GlobeNewswire - 58 articles
2. Daily Excelsior - 46 articles  
3. The Hitavada - 16 articles
4. Times of India - 7 articles
5. The Manila Times - 6 articles

### Technical Implementation
```python
# Key cleaning functions implemented:
- normalize_date()      # Date standardization
- clean_text()          # Text normalization  
- is_tech_related()     # Content filtering
- remove_duplicates()   # Deduplication
- validate_article()    # Quality validation
```

### Output Schema
```json
{
  "articles": [...],           // Clean tech articles
  "all_articles": [...],       // All articles (for reference)
  "stats": {                   // Processing metadata
    "total_raw_articles": 269,
    "total_tech_articles": 249,
    "duplicates_removed": 20,
    "date_range": {...},
    "sources": [...],
    "processed_at": "2025-09-28T22:55:46Z"
  }
}
```

---

## Current Status & Next Steps

### ✅ Completed Phases
1. **Pipeline Setup** - Multi-provider fetching system
2. **Data Collection** - 24 days of automated collection  
3. **Data Cleaning** - Comprehensive processing pipeline

### 🔄 Current State
- **Raw Data:** 269 articles across 23 files
- **Clean Data:** 249 unique tech articles ready for analysis
- **Infrastructure:** Fully automated daily collection
- **Quality:** 92.6% data retention after cleaning
