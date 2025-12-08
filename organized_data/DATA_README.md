# Data Organization - Financial Analytics Project

## Folder Structure
```
organized_data/
├── raw_data/                    # Original API data (REAL DATA)
│   ├── news/                    # NewsAPI data
│   ├── stocks/                  # Yahoo Finance data
│   ├── worldbank/               # World Bank API data
│   ├── fred/                    # FRED API data
│   └── synthetic/               # Generated demo data
└── processed_data/              # Algorithm outputs (CSV)
    ├── spark_analysis/          # PySpark cross-analysis outputs
    ├── deep_semantic/           # NLP/ML analysis outputs
    ├── batch_spark/             # Batch processing outputs
    └── economics_social/        # Economic correlation outputs
```

---

## RAW DATA (Real API Sources)

### 1. News Data - NewsAPI (REAL)
- **File:** `raw_data/news/financial_news_headlines.csv`
- **Source:** https://newsapi.org
- **API Key:** Configured in `.env`
- **Data Period:** 2025-12-03 ~ 2025-12-04
- **Rows:** 221 articles
- **Sources:** 79 different news publishers
- **Columns:** title, description, source, publishedAt, url, sentiment

### 2. Stock Data - Yahoo Finance (REAL)
- **File:** `raw_data/stocks/stock_prices.csv`
- **Source:** Yahoo Finance via `yfinance` Python library
- **Data Period:** 1962 ~ 2025
- **Rows:** 179,289 records
- **Symbols:** 18 major stocks/ETFs (AAPL, MSFT, GOOGL, AMZN, TSLA, META, JPM, GS, SPY, QQQ, etc.)
- **Columns:** symbol, date, open, high, low, close, volume, adj_close

### 3. World Bank Indicators (REAL)
- **File:** `raw_data/worldbank/world_bank_indicators.csv`
- **Source:** World Bank Open Data API (https://data.worldbank.org)
- **Data Period:** 1970 ~ 2024
- **Rows:** 85,175 records
- **Countries:** 262 countries/regions
- **Indicators:** GDP, GDP per capita, Population, Inflation, Unemployment, etc.
- **Columns:** country, indicator, year, value

### 4. FRED Economic Data (REAL)
- **File:** `raw_data/fred/economic_indicators.csv`
- **Source:** Federal Reserve Economic Data (https://fred.stlouisfed.org)
- **API Key:** Configured in `.env`
- **Data Period:** 1970 ~ 2024
- **Rows:** 40,165 records
- **Indicators:** GDP, CPI, Unemployment Rate, Federal Funds Rate, S&P 500, etc.
- **Columns:** indicator, date, value

### 5. Synthetic Data (DEMO/GENERATED)
- **Reddit Comments:** Generated programmatically for social media analysis demo
- **YouTube Comments:** Generated programmatically for sentiment analysis demo
- **Note:** These are NOT real social media data - created for demonstration purposes

---

## PROCESSED DATA (Algorithm Outputs)

### spark_analysis/ (PySpark Cross-Analysis)
| File | Description | Algorithm |
|------|-------------|-----------|
| `word_pagerank.csv` | Word importance ranking | PageRank on word co-occurrence graph |
| `word_graph_edges.csv` | Word relationship network | Co-occurrence analysis |
| `granger_correlations.csv` | Word-price causal relationships | Granger causality test |
| `word_before_price_lag1.csv` | Words preceding price movements | Lag correlation analysis |
| `word_date_counts.csv` | Word frequency by date | Time series aggregation |
| `word_comparison_sources.csv` | Cross-source word analysis | TF-IDF comparison |
| `sentiment_by_source.csv` | Source sentiment breakdown | Sentiment aggregation |
| `reddit_comments.csv` | Processed Reddit data | Text preprocessing |
| `youtube_comments.csv` | Processed YouTube data | Text preprocessing |

### deep_semantic/ (NLP/ML Analysis)
| File | Description | Algorithm |
|------|-------------|-----------|
| `tfidf_by_sentiment.csv` | TF-IDF scores by sentiment | TF-IDF vectorization |
| `lda_topics.csv` | Topic modeling results | Latent Dirichlet Allocation |
| `lda_topic_distribution.csv` | Document-topic mapping | LDA topic distribution |
| `lda_topic_sentiment.csv` | Topic sentiment analysis | LDA + Sentiment |
| `word2vec_vectors.csv` | Word embeddings | Word2Vec neural network |
| `word2vec_similarities.csv` | Semantic word similarities | Cosine similarity |
| `ngram_analysis.csv` | Phrase frequency analysis | N-gram extraction |
| `entity_price_correlation.csv` | Entity-price relationships | Correlation analysis |
| `korea_usa_gdp_comparison.csv` | Korea-USA economic comparison | GDP analysis |
| `kospi_index.csv` | KOSPI market data | Yahoo Finance (005930.KS) |
| `samsung_stock.csv` | Samsung Electronics stock | Yahoo Finance |
| `krw_exchange_rate.csv` | KRW/USD exchange rate | Yahoo Finance |
| `usa_market_performance.csv` | US market summary | Multi-index analysis |

### batch_spark/ (Batch Processing)
| File | Description | Algorithm |
|------|-------------|-----------|
| `stock_summary_stats.csv` | Stock statistical summary | Mean, std, min, max |
| `stock_correlations.csv` | Inter-stock correlations | Pearson correlation |
| `stock_sentiment_combined.csv` | Stock + sentiment merge | Data integration |
| `company_sentiment_analysis.csv` | Per-company sentiment | Aggregated sentiment |
| `daily_sentiment_analysis.csv` | Daily sentiment trends | Time series aggregation |
| `word_frequencies.csv` | Overall word frequencies | Token counting |
| `word_frequencies_positive.csv` | Positive sentiment words | Filtered word counts |
| `word_frequencies_negative.csv` | Negative sentiment words | Filtered word counts |
| `word_frequencies_neutral.csv` | Neutral sentiment words | Filtered word counts |
| `gdp_rankings.csv` | Country GDP rankings | World Bank data sort |
| `gdp_trends.csv` | GDP growth trends | Time series analysis |
| `economic_trends.csv` | Economic indicator trends | Multi-indicator analysis |

### economics_social/ (Economic Correlations)
| File | Description | Algorithm |
|------|-------------|-----------|
| `correlation_matrix.csv` | Multi-asset correlations | Pearson correlation matrix |
| `linear_regression_results.csv` | Asset relationship models | OLS regression |
| `decade_analysis.csv` | Decade-by-decade trends | Temporal grouping |
| `merged_multi_asset_data.csv` | Combined asset dataset | Data integration |

---

## Data Source Summary

| Source | Type | Real/Synthetic | Period | Records |
|--------|------|----------------|--------|---------|
| NewsAPI | News Headlines | REAL | 2025-12-03~04 | 221 |
| Yahoo Finance | Stock Prices | REAL | 1962~2025 | 179,289 |
| World Bank API | Economic Indicators | REAL | 1970~2024 | 85,175 |
| FRED API | US Economic Data | REAL | 1970~2024 | 40,165 |
| Reddit | Social Media | SYNTHETIC | Demo | Variable |
| YouTube | Comments | SYNTHETIC | Demo | Variable |

---

## API Configuration

API keys are stored in `.env` file:
- `NEWS_API_KEY` - NewsAPI key
- `FRED_API_KEY` - FRED API key
- `FMP_API_KEY` - Financial Modeling Prep key

---

Generated: 2025-12-06
