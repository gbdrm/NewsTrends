# Tech News Trends - What I Found

Just some interesting patterns I noticed while digging through tech news from September to November 2025.

## The Data

- 3,201 articles total
- September 5 to November 23, 2025 (about 80 days)
- Got them from GNews API and NewsAPI
- 33 different news sites
- Works out to about 40 articles per day

## What's Hot in Tech News

### AI is EVERYWHERE

AI-related keywords show up constantly. ChatGPT, machine learning, artificial intelligence - they're in like half the articles. It's clearly the biggest story in tech right now.

### The Big Companies

Apple and Google get mentioned all the time. Microsoft, Amazon, Tesla, and Nvidia come up pretty regularly too. Makes sense since they're driving most of the major tech news.

## Publishing Patterns

Weekdays see way more articles than weekends. Monday through Friday is when most tech news gets published, then it drops off on Saturday and Sunday.

Daily volume bounces between 20-60 articles depending on the day. Big spikes probably mean something major happened (product launch, acquisition, whatever).

## Some Observations

**AI dominance is real** - it's not just a lot of coverage, it's THE dominant topic. Way more than anything else.

**Big tech stays in the news** - the major companies maintain steady coverage regardless of what else is happening.

**Crypto is volatile** - mentions spike and drop, probably tied to price movements or regulatory news.

**Diverse sources help** - having 33 different publishers means you get different angles on the same stories.

## Technical Notes

The analysis was pretty straightforward:
- Pandas made time series stuff easy
- Matplotlib worked fine for visualizations
- Basic keyword matching was good enough for spotting trends
- Grouping by day gave a good view of patterns

Could be better:
- Keyword matching is simple - real NLP would catch more nuance
- No sentiment analysis (is the AI coverage positive or negative?)
- Only English sources
- Free API limits mean we can't get everything
- Simple text search misses context sometimes

## What This Tells Us

Over these ~80 days, AI clearly dominated tech journalism. Big tech companies stayed consistently relevant, and crypto remained a volatile topic that journalists cover when it's moving.

The automated collection worked well - diverse content from multiple sources, good data quality overall.


