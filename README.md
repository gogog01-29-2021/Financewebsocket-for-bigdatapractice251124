# Finance Data Semantics for Big Data Analysis

A comprehensive financial data analysis pipeline using PySpark, multi-database architecture, and advanced NLP techniques for cross-source semantic analysis.

## Project Overview

This project demonstrates big data analytics capabilities by combining:
- **55+ years of economic data** (World Bank: 1970-2024)
- **Stock market data** (Yahoo Finance - 18 major symbols)
- **News headlines with sentiment analysis** (21,000+ articles)
- **Cryptocurrency data** (Bitcoin, Ethereum: 2015-2024)
- **Korea Real Estate Index** (1970-2024)
- **Advanced NLP**: TF-IDF, LDA Topic Modeling, Word2Vec, PageRank, Granger Causality

## Architecture

```
+------------------------------------------------------------------+
|                      Data Sources                                 |
+--------------+--------------+--------------+--------------+-------+
| World Bank   |    FRED      |Yahoo Finance |    News      | Social|
|  GDP Data    |  Economic    |   Stocks     |  Headlines   | Media |
|  (55+ yrs)   | Indicators   | (18 symbols) |  (21k+)      |(YT/RD)|
+------+-------+------+-------+------+-------+------+-------+---+---+
       |              |              |              |            |
       v              v              v              v            v
+------------------------------------------------------------------+
|                    PySpark Processing Engine                      |
|  +---------------+ +---------------+ +--------------------------+ |
|  | MapReduce     | |  Technical    | |   Advanced Analytics     | |
|  | Word Count    | |  Indicators   | |  - TF-IDF Analysis       | |
|  | (key,value)   | |  SMA/RSI/BB   | |  - LDA Topic Modeling    | |
|  +---------------+ +---------------+ |  - Word2Vec Embeddings   | |
|                                      |  - PageRank Algorithm    | |
|                                      |  - Granger Causality     | |
|                                      +--------------------------+ |
+------------------------------------------------------------------+
                              |
       +----------------------+----------------------+
       v                      v                      v
+--------------+      +--------------+      +--------------+
|   QuestDB    |      |   MongoDB    |      |  Cassandra   |
| Time-series  |      |  Documents   |      |  Wide-col    |
+--------------+      +--------------+      +--------------+
                              |
                              v
                    +------------------+
                    |    Streamlit     |
                    |   Dashboards     |
                    | (8501,8502,8503) |
                    +------------------+
```

## Dashboards

| Port | Dashboard | Description |
|------|-----------|-------------|
| 8501 | Live Dashboard | Real-time WebSocket data streaming |
| 8502 | Batch Dashboard | Historical data analysis |
| 8503 | Advanced Dashboard | Deep semantic analysis & Multi-asset correlation |

---

## Directory Structure

```
Finance-Data-sementiecs-for-Bigdata-analysis-251125/
│
├── batch_data_downloader.py        # Download all datasets from APIs
├── batch_data_loader.py            # Load data into databases
├── batch_spark_analytics.py        # PySpark analytics pipeline
├── batch_dashboard.py              # Main Streamlit dashboard (port 8502)
├── live_dashboard.py               # Real-time WebSocket dashboard (port 8501)
│
├── advanced_analytics/             # Advanced cross-source analysis
│   ├── advanced_spark_cross_analysis.py   # PageRank, Granger, word network
│   ├── deep_semantic_analysis.py          # TF-IDF, LDA, Word2Vec (PySpark)
│   ├── advanced_dashboard.py              # Advanced dashboard (port 8503)
│   ├── spark_output/                      # Basic analysis outputs
│   └── deep_semantic_output/              # Deep NLP analysis outputs
│
├── economics_social_cultural/      # Multi-asset correlation analysis
│   ├── multi_asset_correlation_analysis.py  # Cross-asset correlation
│   └── output/                              # Analysis results
│
├── data/                           # Downloaded datasets
│   └── batch/
│       ├── stocks/stock_prices.csv         # Stock price data
│       ├── news/financial_news_headlines.csv  # News with sentiment
│       └── worldbank/world_bank_indicators.csv  # GDP and economic data
│
└── spark_output/                   # Spark analysis results
```

---

## Features & Implementation Details

### 1. Core Analytics (`batch_spark_analytics.py`)

| Feature | Method | Output |
|---------|--------|--------|
| Technical Indicators | PySpark Window Functions | SMA(20/50), RSI, Bollinger Bands |
| Stock Correlations | Pearson Correlation | Cross-symbol correlation matrix |
| Word Frequency | MapReduce (key-value) | Sentiment-aware word counts |
| Cross-Source Analysis | Join + Aggregation | Economic vs stock correlations |

**Output Location:** `spark_output/`

---

### 2. Advanced Analytics (`advanced_analytics/`)

#### 2.1 PageRank for Word Importance (`advanced_spark_cross_analysis.py`)

Google's PageRank algorithm applied to financial text to find influential words based on co-occurrence networks.

```python
# Implementation: NetworkX PageRank on word co-occurrence graph
G = nx.Graph()
for word_pair in co_occurrences:
    G.add_edge(word1, word2, weight=count)
pagerank_scores = nx.pagerank(G, weight='weight')
```

**Output:** `advanced_analytics/spark_output/word_pagerank.csv`

| Column | Description |
|--------|-------------|
| word | Financial term |
| pagerank | Influence score (0-1) |
| frequency | Raw occurrence count |

#### 2.2 Granger Causality (`advanced_spark_cross_analysis.py`)

Tests bidirectional causality between sentiment and stock returns.

- **Sentiment → Return**: Does news sentiment predict future stock returns?
- **Return → Sentiment**: Do stock movements influence news sentiment?

**Output:** `advanced_analytics/spark_output/granger_correlations.csv`

#### 2.3 Deep Semantic Analysis (`deep_semantic_analysis.py`)

##### TF-IDF Analysis
Identifies distinctive words for each sentiment category.

```python
# PySpark ML TF-IDF Pipeline
from pyspark.ml.feature import HashingTF, IDF
tf = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
```

**Output:** `deep_semantic_output/tfidf_by_sentiment.csv`

##### LDA Topic Modeling
Discovers hidden topics in financial news.

```python
# PySpark LDA with 5 topics
from pyspark.ml.clustering import LDA
lda = LDA(k=5, maxIter=10, optimizer='em')
# Fixed: Using vector_to_array() to avoid Python worker crash
from pyspark.ml.functions import vector_to_array
```

**Output:** `deep_semantic_output/lda_topics.csv`, `lda_topic_distribution.csv`

##### Word2Vec Embeddings
Learns semantic word representations from context.

```python
# Using Gensim Word2Vec (PySpark Word2Vec had serialization issues)
from gensim.models import Word2Vec
model = Word2Vec(sentences=sentences, vector_size=100, window=5)
```

**Output:** `deep_semantic_output/word2vec_similarities.csv`

##### N-gram Analysis
Extracts meaningful phrases (bigrams, trigrams).

**Output:** `deep_semantic_output/ngram_analysis.csv`

##### Entity-Price Correlation
Analyzes how company mentions correlate with stock price movements.

```python
# Date normalization fix for join operation
news_normalized = news_df.withColumn("date_normalized", to_date(col("date")))
stock_normalized = stock_df.withColumn("date_normalized", to_date(col("date")))
```

**Output:** `deep_semantic_output/entity_price_correlation.csv`

| Column | Description |
|--------|-------------|
| symbol | Stock symbol (AAPL, MSFT, etc.) |
| same_day_corr | Correlation on same day |
| next_day_corr | Predictive correlation (t+1) |
| sentiment_return_corr | Sentiment vs return correlation |

##### Korea vs USA GDP Comparison

```python
# Fixed: Filter by indicator_code for GDP data
gdp_indicator = 'NY.GDP.MKTP.CD'  # GDP (current US$)
gdp_data = gdp_data.filter(col("indicator_code") == gdp_indicator)
```

**Output:** `deep_semantic_output/korea_usa_gdp_comparison.csv`

---

### 3. Multi-Asset Correlation Analysis (`economics_social_cultural/`)

Cross-asset correlation study including:
- USA GDP, Korea GDP
- S&P 500 (SPY)
- Cryptocurrency (Bitcoin, Ethereum)
- Korea Real Estate Index

#### 3.1 Implementation (`multi_asset_correlation_analysis.py`)

```python
# Data Sources
- World Bank API: GDP data (USA, Korea)
- yfinance: S&P 500 (SPY), Bitcoin (BTC-USD), Ethereum (ETH-USD)
- Simulated: Korea Real Estate Index (based on historical patterns)

# Analysis Methods
- Pearson Correlation Matrix
- Pairwise Linear Regression (sklearn)
- Decade-by-Decade Growth Analysis
```

**Key Fix:** yfinance MultiIndex column handling
```python
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = [col[0] if col[1] == '' else col[0] for col in spy.columns]
```

#### 3.2 Outputs

| File | Description | Format |
|------|-------------|--------|
| `merged_multi_asset_data.csv` | Combined time series (1970-2024) | year, usa_gdp, korea_gdp, sp500_price, btc_price, eth_price, korea_real_estate |
| `correlation_matrix.csv` | 9x9 Pearson correlation matrix | Symmetric matrix |
| `linear_regression_results.csv` | All pairwise regressions | x, y, slope, r², p-value |
| `decade_analysis.csv` | Growth by decade (1970s-2020s) | Decade, growth rates, averages |
| `analysis_report.json` | Summary statistics | Key findings, significant correlations |

#### 3.3 Key Findings

| Relationship | Correlation | R² | Significance |
|--------------|-------------|----|--------------|
| USA GDP ↔ Korea GDP | 0.977 | 0.954 | p < 0.001 |
| S&P 500 ↔ Korea Real Estate | 0.964 | 0.928 | p < 0.001 |
| Bitcoin ↔ Ethereum | 0.957 | 0.917 | p < 0.001 |
| USA GDP ↔ S&P 500 | 0.900 | 0.811 | p < 0.001 |

**Decade Growth Highlights:**
- Korea 1970s: 643% GDP growth (rapid industrialization)
- Bitcoin 2010s: 1933% growth (cryptocurrency emergence)
- S&P 500 2010s: 205% growth (post-2008 recovery)

---

---

## Complete Data Pipeline Documentation

This section provides comprehensive documentation of all raw datasets, their transformations, output results, units, and the code snippets used for each analysis across all three dashboards.

---

## RAW DATASETS

### 1. Stock Price Data

| Attribute | Details |
|-----------|---------|
| **File Location** | `data/batch/stocks/stock_prices.csv` |
| **Source** | Yahoo Finance API (`yfinance` library) |
| **Records** | ~179,146 rows |
| **Time Range** | 1980-2024 (varies by symbol) |
| **Symbols** | 18 stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, GS, BAC, WMT, DIS, NFLX, AMD, INTC, BA, SPY, DIA |

**Schema:**
```csv
date,symbol,open,high,low,close,volume
1980-12-12 00:00:00-05:00,AAPL,0.128348,0.128906,0.128348,0.128348,469033600
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| date | ISO 8601 datetime with timezone | Trading date |
| symbol | string | Stock ticker symbol |
| open | USD ($) | Opening price |
| high | USD ($) | Highest price of the day |
| low | USD ($) | Lowest price of the day |
| close | USD ($) | Closing price (adjusted) |
| volume | shares (integer) | Number of shares traded |

**Download Code:**
```python
# batch_data_downloader.py
import yfinance as yf

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
           'JPM', 'GS', 'BAC', 'WMT', 'DIS', 'NFLX', 'AMD', 'INTC',
           'BA', 'SPY', 'DIA']

all_data = []
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max")
    hist['symbol'] = symbol
    all_data.append(hist)

stock_df = pd.concat(all_data)
stock_df.to_csv('data/batch/stocks/stock_prices.csv')
```

**Connected Dashboard Sections:**
- Port 8502: Stock Analysis, Technical Indicators
- Port 8503: Entity-Price Link, Word-Price Causality

---

### 2. News Headlines Data

| Attribute | Details |
|-----------|---------|
| **File Location** | `data/batch/news/financial_news_headlines.csv` |
| **Source** | Kaggle Financial News Dataset (synthetic/augmented) |
| **Records** | ~21,389 headlines |
| **Time Range** | 2019-2023 |

**Schema:**
```csv
headline,sentiment,date,source
"Stock market rallies on positive earnings",positive,2019-01-15,news
"Federal Reserve raises interest rates",negative,2019-03-20,news
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| headline | string | News headline text |
| sentiment | categorical | positive, negative, neutral |
| date | YYYY-MM-DD | Publication date |
| source | string | Source type (news, youtube, reddit) |

**Connected Dashboard Sections:**
- Port 8502: Word Frequency Analysis
- Port 8503: All NLP analyses (TF-IDF, LDA, Word2Vec, N-grams, PageRank)

---

### 3. World Bank Economic Indicators

| Attribute | Details |
|-----------|---------|
| **File Location** | `data/batch/worldbank/world_bank_indicators.csv` |
| **Source** | World Bank API (`wbgapi` library) |
| **Records** | ~77,194 rows |
| **Countries** | 262 countries |
| **Time Range** | 1960-2023 |

**Schema:**
```csv
country_code,country_name,indicator_code,indicator_name,year,value
USA,United States,NY.GDP.MKTP.CD,GDP (current US$),1970,1073303000000.0
KOR,Korea Rep.,NY.GDP.MKTP.CD,GDP (current US$),1970,9005144969.11666
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| country_code | ISO 3166-1 alpha-3 | Country code |
| country_name | string | Full country name |
| indicator_code | string | World Bank indicator code |
| indicator_name | string | Human-readable indicator name |
| year | integer | Year of observation |
| value | varies by indicator | See indicator-specific units below |

**Key Indicators Used:**
| Indicator Code | Name | Unit |
|----------------|------|------|
| NY.GDP.MKTP.CD | GDP (current US$) | USD ($) |
| NY.GDP.MKTP.KD.ZG | GDP growth (annual %) | Percentage (%) |
| FP.CPI.TOTL.ZG | Inflation, consumer prices | Percentage (%) |
| SL.UEM.TOTL.ZS | Unemployment rate | Percentage (%) |

**Download Code:**
```python
# batch_data_downloader.py
import wbgapi as wb

indicators = ['NY.GDP.MKTP.CD', 'NY.GDP.MKTP.KD.ZG', 'FP.CPI.TOTL.ZG']
countries = ['USA', 'KOR', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA']

data = wb.data.DataFrame(indicators, countries, time=range(1960, 2024))
data.to_csv('data/batch/worldbank/world_bank_indicators.csv')
```

**Connected Dashboard Sections:**
- Port 8502: Economic Analysis
- Port 8503: Korea vs USA GDP Comparison, Multi-Asset Correlation

---

### 4. Cryptocurrency Data

| Attribute | Details |
|-----------|---------|
| **File Location** | Generated in-memory (economics_social_cultural) |
| **Source** | Yahoo Finance API (`yfinance`) |
| **Symbols** | BTC-USD (Bitcoin), ETH-USD (Ethereum) |
| **Time Range** | 2014-2024 |

**Schema:**
```csv
Date,Open,High,Low,Close,Volume
2014-09-17,465.86,468.17,452.42,457.33,21056800
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| Date | YYYY-MM-DD | Trading date |
| Open/High/Low/Close | USD ($) | Price in US dollars |
| Volume | USD ($) | Trading volume in USD |

**Download Code:**
```python
# economics_social_cultural/multi_asset_correlation_analysis.py
import yfinance as yf

btc = yf.download("BTC-USD", period="max", progress=False, auto_adjust=True)
eth = yf.download("ETH-USD", period="max", progress=False, auto_adjust=True)

# Fix MultiIndex columns (new yfinance version)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = [col[0] for col in btc.columns]
```

**Connected Dashboard Sections:**
- Port 8503: Multi-Asset Correlation, Linear Regression, Decade Analysis

---

### 5. S&P 500 Index Data (SPY ETF)

| Attribute | Details |
|-----------|---------|
| **File Location** | Generated in-memory |
| **Source** | Yahoo Finance API |
| **Symbol** | SPY (SPDR S&P 500 ETF) |
| **Time Range** | 1993-2024 |

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| Close | USD ($) | ETF price (represents S&P 500 / 10) |
| Volume | shares | Number of shares traded |

**Connected Dashboard Sections:**
- Port 8503: Multi-Asset Correlation, Decade Analysis

---

### 6. Korea Real Estate Index (Simulated)

| Attribute | Details |
|-----------|---------|
| **File Location** | Generated in-memory |
| **Source** | Simulated based on historical patterns |
| **Time Range** | 1970-2024 |
| **Base Year** | 1970 = 100 |

**Generation Code:**
```python
# economics_social_cultural/multi_asset_correlation_analysis.py
korea_re = pd.DataFrame({'year': range(1970, 2025)})
korea_re['korea_real_estate'] = 100.0  # Base index

# Apply historical growth rates by decade
growth_rates = {
    (1970, 1990): 0.00,   # No growth data
    (1991, 1997): 0.08,   # Pre-Asian crisis boom
    (1998, 1999): -0.05,  # Asian financial crisis
    (2000, 2007): 0.12,   # Recovery and boom
    (2008, 2009): 0.02,   # Global financial crisis
    (2010, 2021): 0.20,   # Low interest rate era
    (2022, 2024): -0.03   # Interest rate hikes
}
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| year | integer | Year |
| korea_real_estate | index (1970=100) | Price index relative to 1970 |

---

## ANALYSIS OUTPUTS BY DASHBOARD

---

## Dashboard 1: Live Dashboard (Port 8501)

### WebSocket Real-time Data

| Feature | Raw Data | Output | Unit |
|---------|----------|--------|------|
| Real-time Prices | WebSocket stream | Live price chart | USD ($) |
| Volume Ticker | WebSocket stream | Volume bar chart | Shares |

---

## Dashboard 2: Batch Dashboard (Port 8502)

### 2.1 Technical Indicators

**Input:** `stock_prices.csv`
**Output:** `spark_output/technical_indicators/`

| Indicator | Formula | Unit |
|-----------|---------|------|
| SMA(20) | Mean of last 20 closing prices | USD ($) |
| SMA(50) | Mean of last 50 closing prices | USD ($) |
| RSI(14) | 100 - (100 / (1 + RS)) | 0-100 (dimensionless) |
| Bollinger Upper | SMA(20) + 2×STD(20) | USD ($) |
| Bollinger Lower | SMA(20) - 2×STD(20) | USD ($) |

**Code Snippet:**
```python
# batch_spark_analytics.py
from pyspark.sql import Window
from pyspark.sql.functions import avg, stddev, col, lag

# Window for 20-day SMA
window_20 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)

# Calculate SMA
stock_df = stock_df.withColumn("sma_20", avg("close").over(window_20))
stock_df = stock_df.withColumn("sma_50", avg("close").over(window_50))

# Calculate RSI
delta = col("close") - lag("close", 1).over(window_symbol)
gain = when(delta > 0, delta).otherwise(0)
loss = when(delta < 0, -delta).otherwise(0)
avg_gain = avg(gain).over(window_14)
avg_loss = avg(loss).over(window_14)
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
```

### 2.2 Stock Correlations

**Input:** `stock_prices.csv`
**Output:** `spark_output/stock_correlations.csv`

**Schema:**
```csv
symbol1,symbol2,correlation
AAPL,MSFT,0.923
SPY,DIA,0.951
```

**Units:**
| Column | Unit | Range |
|--------|------|-------|
| correlation | Pearson coefficient | -1.0 to +1.0 |

**Code Snippet:**
```python
# batch_spark_analytics.py
from pyspark.sql.functions import corr

# Pivot to get prices by symbol
pivoted = stock_df.groupBy("date").pivot("symbol").agg(first("close"))

# Calculate correlation matrix
correlations = []
for s1 in symbols:
    for s2 in symbols:
        if s1 < s2:
            r = pivoted.stat.corr(s1, s2)
            correlations.append((s1, s2, r))
```

### 2.3 Word Frequency Analysis

**Input:** `financial_news_headlines.csv`
**Output:** `spark_output/word_frequencies.csv`

**Schema:**
```csv
word,sentiment,count,total_count
market,positive,1234,3456
stock,negative,567,2345
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| count | integer | Occurrences in that sentiment |
| total_count | integer | Total occurrences across all sentiments |

**Code Snippet (MapReduce):**
```python
# batch_spark_analytics.py
from pyspark.sql.functions import explode, split, lower, regexp_replace

# Tokenize headlines
words_df = news_df.withColumn(
    "word",
    explode(split(regexp_replace(lower(col("headline")), "[^a-z\\s]", ""), "\\s+"))
)

# MapReduce word count
word_counts = words_df.groupBy("word", "sentiment").count()

# Filter stopwords and short words
word_counts = word_counts.filter(
    (length(col("word")) > 3) &
    (~col("word").isin(stopwords))
)
```

---

## Dashboard 3: Advanced Dashboard (Port 8503)

### 3.1 PageRank Word Importance

**Input:** `financial_news_headlines.csv`
**Output:** `advanced_analytics/spark_output/word_pagerank.csv`

**Schema:**
```csv
word,pagerank,frequency
market,0.0847,15234
stock,0.0723,12456
price,0.0651,10234
```

**Units:**
| Column | Unit | Range |
|--------|------|-------|
| pagerank | probability | 0.0 to 1.0 (sum = 1.0) |
| frequency | count | Integer |

**Code Snippet:**
```python
# advanced_analytics/advanced_spark_cross_analysis.py
import networkx as nx
from itertools import combinations

# Build co-occurrence graph
G = nx.Graph()

for headline in headlines:
    words = tokenize(headline)
    # Add edges for all word pairs in same headline
    for w1, w2 in combinations(words, 2):
        if G.has_edge(w1, w2):
            G[w1][w2]['weight'] += 1
        else:
            G.add_edge(w1, w2, weight=1)

# Run PageRank
pagerank_scores = nx.pagerank(G, weight='weight', alpha=0.85)

# Convert to DataFrame
pagerank_df = pd.DataFrame([
    {'word': word, 'pagerank': score, 'frequency': word_freq[word]}
    for word, score in pagerank_scores.items()
]).sort_values('pagerank', ascending=False)
```

### 3.2 Granger Causality Analysis

**Input:** `financial_news_headlines.csv` + `stock_prices.csv`
**Output:** `advanced_analytics/spark_output/granger_correlations.csv`

**Schema:**
```csv
direction,lag,correlation,p_value
sentiment_to_return,1,0.0234,0.0456
return_to_sentiment,1,0.0123,0.0789
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| lag | days | Time lag (1-5 days) |
| correlation | Pearson r | -1.0 to +1.0 |
| p_value | probability | Statistical significance |

**Code Snippet:**
```python
# advanced_analytics/advanced_spark_cross_analysis.py
from pyspark.sql.functions import lag, corr
from pyspark.sql import Window

# Create time-aligned data
window = Window.orderBy("date")

# Add lagged sentiment
merged = merged.withColumn("sentiment_lag1", lag("avg_sentiment", 1).over(window))
merged = merged.withColumn("sentiment_lag2", lag("avg_sentiment", 2).over(window))

# Calculate correlation: sentiment(t-1) vs return(t)
for lag_days in range(1, 6):
    sentiment_col = f"sentiment_lag{lag_days}"
    r = merged.stat.corr(sentiment_col, "daily_return")
    results.append({
        'direction': 'sentiment_to_return',
        'lag': lag_days,
        'correlation': r
    })
```

### 3.3 TF-IDF Analysis

**Input:** `financial_news_headlines.csv`
**Output:** `advanced_analytics/deep_semantic_output/tfidf_by_sentiment.csv`

**Schema:**
```csv
sentiment,word,tfidf_score,doc_frequency
positive,growth,0.234,1234
negative,decline,0.345,567
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| tfidf_score | float | TF × IDF weight |
| doc_frequency | integer | Documents containing word |

**Code Snippet:**
```python
# advanced_analytics/deep_semantic_analysis.py
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

# Tokenize
tokenizer = Tokenizer(inputCol="headline", outputCol="words")
wordsData = tokenizer.transform(news_df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)

# TF-IDF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
tf = hashingTF.transform(filteredData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)
```

### 3.4 LDA Topic Modeling

**Input:** `financial_news_headlines.csv`
**Output:**
- `advanced_analytics/deep_semantic_output/lda_topics.csv`
- `advanced_analytics/deep_semantic_output/lda_topic_distribution.csv`

**lda_topics.csv Schema:**
```csv
topic_id,topic_name,word,weight
0,Market Trends,market,0.0523
0,Market Trends,stock,0.0456
1,Federal Reserve,fed,0.0678
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| topic_id | integer | Topic number (0 to k-1) |
| weight | probability | Word probability in topic (sum = 1.0) |

**lda_topic_distribution.csv Schema:**
```csv
dominant_topic,count
0,4523
1,3456
2,2789
```

**Code Snippet:**
```python
# advanced_analytics/deep_semantic_analysis.py
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.functions import vector_to_array

# Vectorize
cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=5000)
cv_model = cv.fit(filteredData)
vectorized = cv_model.transform(filteredData)

# LDA
lda = LDA(k=5, maxIter=10, optimizer='em')
lda_model = lda.fit(vectorized)

# Get topics
topics = lda_model.describeTopics(maxTermsPerTopic=10)
vocab = cv_model.vocabulary

# Assign dominant topic (FIXED: using vector_to_array)
transformed = lda_model.transform(vectorized)
transformed_with_array = transformed.withColumn(
    "topic_array",
    vector_to_array("topicDistribution")
)
transformed_with_topic = transformed_with_array.withColumn(
    "dominant_topic",
    expr("array_position(topic_array, array_max(topic_array)) - 1").cast("int")
)
```

### 3.5 Word2Vec Embeddings

**Input:** `financial_news_headlines.csv`
**Output:** `advanced_analytics/deep_semantic_output/word2vec_similarities.csv`

**Schema:**
```csv
term,similar_word,similarity
stock,equity,0.876
stock,share,0.834
market,trading,0.789
```

**Units:**
| Column | Unit | Range |
|--------|------|-------|
| similarity | cosine similarity | 0.0 to 1.0 |

**Code Snippet:**
```python
# advanced_analytics/deep_semantic_analysis.py
from gensim.models import Word2Vec

# Extract sentences using collect() (avoids PySpark serialization issues)
sentences_rows = df.select("filtered").collect()
sentences = [row['filtered'] for row in sentences_rows if row['filtered']]

# Train Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=1,
    epochs=10
)

# Find similar words
key_terms = ['stock', 'market', 'price', 'growth', 'economy', 'fed', 'earnings']
results = []
for term in key_terms:
    if term in model.wv:
        similar = model.wv.most_similar(term, topn=5)
        for word, sim in similar:
            results.append({'term': term, 'similar_word': word, 'similarity': sim})
```

### 3.6 N-gram Analysis

**Input:** `financial_news_headlines.csv`
**Output:** `advanced_analytics/deep_semantic_output/ngram_analysis.csv`

**Schema:**
```csv
ngram,n,total_count,positive,negative,neutral
interest rate,2,1234,456,567,211
federal reserve,2,987,234,567,186
stock market rally,3,456,345,67,44
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| n | integer | N-gram size (2=bigram, 3=trigram) |
| total_count | integer | Total occurrences |
| positive/negative/neutral | integer | Count by sentiment |

**Code Snippet:**
```python
# advanced_analytics/deep_semantic_analysis.py
from pyspark.ml.feature import NGram

# Generate bigrams
bigram = NGram(n=2, inputCol="filtered", outputCol="bigrams")
bigram_df = bigram.transform(filteredData)

# Generate trigrams
trigram = NGram(n=3, inputCol="filtered", outputCol="trigrams")
trigram_df = trigram.transform(filteredData)

# Count n-grams
bigram_counts = bigram_df.select(
    explode(col("bigrams")).alias("ngram"),
    col("sentiment")
).groupBy("ngram", "sentiment").count()
```

### 3.7 Entity-Price Correlation

**Input:** `financial_news_headlines.csv` + `stock_prices.csv`
**Output:** `advanced_analytics/deep_semantic_output/entity_price_correlation.csv`

**Schema:**
```csv
symbol,company,total_mentions,same_day_corr,next_day_corr,sentiment_return_corr
AAPL,Apple,636,-0.0028,-0.0076,-0.0027
MSFT,Microsoft,653,0.0259,-0.0050,-0.0185
```

**Units:**
| Column | Unit | Range |
|--------|------|-------|
| total_mentions | count | Integer |
| same_day_corr | Pearson r | -1.0 to +1.0 |
| next_day_corr | Pearson r | -1.0 to +1.0 |
| sentiment_return_corr | Pearson r | -1.0 to +1.0 |

**Code Snippet:**
```python
# advanced_analytics/deep_semantic_analysis.py
from pyspark.sql.functions import to_date, col

# Entity mapping
entities = {
    'AAPL': ['apple', 'iphone', 'mac'],
    'MSFT': ['microsoft', 'windows', 'azure'],
    'GOOGL': ['google', 'alphabet', 'android'],
    # ... more entities
}

# Normalize dates for join (FIXED)
news_normalized = news_df.withColumn("date_normalized", to_date(col("date")))
stock_normalized = stock_df.withColumn("date_normalized", to_date(col("date")))

# Join and calculate correlations
merged = news_normalized.join(
    stock_normalized,
    (news_normalized.date_normalized == stock_normalized.date_normalized) &
    (news_normalized.entity == stock_normalized.symbol)
)

# Calculate same-day correlation
same_day_corr = merged.stat.corr("mention_count", "daily_return")
```

### 3.8 Korea vs USA GDP Comparison

**Input:** `world_bank_indicators.csv`
**Output:** `advanced_analytics/deep_semantic_output/korea_usa_gdp_comparison.csv`

**Schema:**
```csv
year,korea_gdp,usa_gdp,ratio_korea_usa,korea_growth,usa_growth,growth_diff
1970,9005144969.12,1073303000000.0,0.0084,,,
1971,9903571248.53,1164850000000.0,0.0085,9.98,8.53,1.45
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| korea_gdp | USD ($) | Korea GDP in current US dollars |
| usa_gdp | USD ($) | USA GDP in current US dollars |
| ratio_korea_usa | ratio | Korea GDP / USA GDP |
| korea_growth | % | Year-over-year growth rate |
| usa_growth | % | Year-over-year growth rate |
| growth_diff | percentage points | Korea growth - USA growth |

**Code Snippet:**
```python
# advanced_analytics/deep_semantic_analysis.py
from pyspark.sql.functions import col, lag
from pyspark.sql import Window

# Filter for GDP indicator only (FIXED)
gdp_indicator = 'NY.GDP.MKTP.CD'
gdp_data = worldbank_df.filter(col("indicator_code") == gdp_indicator)

# Separate Korea and USA
korea_gdp = gdp_data.filter(col("country_code") == "KOR").select("year", col("value").alias("korea_gdp"))
usa_gdp = gdp_data.filter(col("country_code") == "USA").select("year", col("value").alias("usa_gdp"))

# Join and calculate metrics
combined = korea_gdp.join(usa_gdp, "year")
combined = combined.withColumn("ratio_korea_usa", col("korea_gdp") / col("usa_gdp"))

# Calculate growth rates
window = Window.orderBy("year")
combined = combined.withColumn(
    "korea_growth",
    ((col("korea_gdp") - lag("korea_gdp", 1).over(window)) /
     lag("korea_gdp", 1).over(window)) * 100
)
```

### 3.9 Multi-Asset Correlation Matrix

**Input:** World Bank API + yfinance API (SPY, BTC-USD, ETH-USD)
**Output:** `economics_social_cultural/output/correlation_matrix.csv`

**Schema:**
```csv
,usa_gdp,korea_gdp,sp500_price,sp500_volume,btc_price,btc_volume,eth_price,eth_volume,korea_real_estate
usa_gdp,1.0,0.977,0.900,0.624,0.634,0.657,0.542,0.620,0.943
korea_gdp,0.977,1.0,0.838,0.674,0.525,0.585,0.426,0.550,0.918
```

**Units:**
| Value | Unit | Range |
|-------|------|-------|
| correlation | Pearson r | -1.0 to +1.0 |

**Code Snippet:**
```python
# economics_social_cultural/multi_asset_correlation_analysis.py
import pandas as pd
from scipy.stats import pearsonr

# Merge all data
merged = pd.merge(usa_gdp, korea_gdp, on='year')
merged = pd.merge(merged, sp500_data, on='year')
merged = pd.merge(merged, btc_data, on='year')
merged = pd.merge(merged, eth_data, on='year')
merged = pd.merge(merged, korea_re, on='year')

# Calculate correlation matrix
numeric_cols = ['usa_gdp', 'korea_gdp', 'sp500_price', 'sp500_volume',
                'btc_price', 'btc_volume', 'eth_price', 'eth_volume',
                'korea_real_estate']
corr_matrix = merged[numeric_cols].corr()
corr_matrix.to_csv('output/correlation_matrix.csv')
```

### 3.10 Linear Regression Analysis

**Input:** Multi-asset merged data
**Output:** `economics_social_cultural/output/linear_regression_results.csv`

**Schema:**
```csv
x_variable,y_variable,slope,intercept,r_squared,correlation,p_value,n_observations
usa_gdp,korea_gdp,0.0787,-133984450375.95,0.954,0.977,3.18e-37,55
sp500_price,korea_real_estate,2.498,79.15,0.928,0.964,5.09e-32,55
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| slope | y-unit / x-unit | Linear regression coefficient |
| intercept | y-unit | Y-intercept |
| r_squared | ratio | Coefficient of determination (0-1) |
| correlation | Pearson r | -1.0 to +1.0 |
| p_value | probability | Statistical significance |
| n_observations | count | Number of data points |

**Code Snippet:**
```python
# economics_social_cultural/multi_asset_correlation_analysis.py
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np

variables = ['usa_gdp', 'korea_gdp', 'sp500_price', 'btc_price',
             'eth_price', 'korea_real_estate']

results = []
for x_var in variables:
    for y_var in variables:
        if x_var != y_var:
            X = merged[[x_var]].values
            y = merged[y_var].values

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate metrics
            r, p_value = pearsonr(merged[x_var], merged[y_var])
            r_squared = model.score(X, y)

            results.append({
                'x_variable': x_var,
                'y_variable': y_var,
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': r_squared,
                'correlation': r,
                'p_value': p_value,
                'n_observations': len(merged)
            })

regression_df = pd.DataFrame(results)
regression_df = regression_df.sort_values('r_squared', ascending=False)
```

### 3.11 Decade Analysis

**Input:** Multi-asset merged data
**Output:** `economics_social_cultural/output/decade_analysis.csv`

**Schema:**
```csv
decade,usa_gdp_growth_%,usa_gdp_avg,korea_gdp_growth_%,korea_gdp_avg,sp500_price_growth_%,btc_price_growth_%
1970s,144.79,1710695600000.0,643.43,27224364685.54,0.0,0.0
1980s,97.44,4173163100000.0,277.58,121456288205.86,0.0,0.0
2010s,43.13,18030002600000.0,44.40,1449571916942.17,205.43,1933.38
```

**Units:**
| Column | Unit | Description |
|--------|------|-------------|
| decade | string | Decade label (1970s, 1980s, etc.) |
| *_growth_% | percentage | Growth from start to end of decade |
| *_avg | varies | Average value across decade |

**Code Snippet:**
```python
# economics_social_cultural/multi_asset_correlation_analysis.py
def calculate_decade_stats(df):
    decades = []
    for start_year in range(1970, 2030, 10):
        end_year = start_year + 9
        decade_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

        if len(decade_data) > 0:
            row = {'decade': f"{start_year}s"}

            for col in numeric_cols:
                first_val = decade_data[col].iloc[0]
                last_val = decade_data[col].iloc[-1]

                # Growth rate
                if first_val > 0:
                    growth = ((last_val - first_val) / first_val) * 100
                else:
                    growth = 0

                row[f'{col}_growth_%'] = growth
                row[f'{col}_avg'] = decade_data[col].mean()

            decades.append(row)

    return pd.DataFrame(decades)
```

---

## Data Formats

### Raw Data

#### Stock Prices (`data/batch/stocks/stock_prices.csv`)
```csv
date,symbol,open,high,low,close,volume
1980-12-12 00:00:00-05:00,AAPL,0.128348,0.128906,0.128348,0.128348,469033600
```

#### News Headlines (`data/batch/news/financial_news_headlines.csv`)
```csv
headline,sentiment,date,source
"Stock market rallies on positive earnings",positive,2019-01-15,news
```

#### World Bank Indicators (`data/batch/worldbank/world_bank_indicators.csv`)
```csv
country_code,country_name,indicator_code,indicator_name,year,value
KOR,Korea Rep.,NY.GDP.MKTP.CD,GDP (current US$),1970,9005144969.11666
```

### Analysis Outputs

#### Correlation Matrix Format
```csv
,usa_gdp,korea_gdp,sp500_price,btc_price,eth_price
usa_gdp,1.0,0.977,0.900,0.634,0.542
korea_gdp,0.977,1.0,0.838,0.525,0.426
```

#### Linear Regression Results Format
```csv
x_variable,y_variable,slope,intercept,r_squared,correlation,p_value,n_observations
usa_gdp,korea_gdp,0.0787,-133984450375.95,0.954,0.977,3.18e-37,55
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- Java 17+ (for PySpark 4.0.1)
- Conda (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/gogog01-29-2021/Finance-Data-sementiecs-for-Bigdata-analysis-251125.git
cd Finance-Data-sementiecs-for-Bigdata-analysis-251125

# Install Java 17 (required for PySpark 4.0.1)
conda install -y -c conda-forge openjdk=17

# Install Python dependencies
pip install pyspark pandas numpy yfinance wbgapi streamlit plotly networkx scipy gensim scikit-learn
```

### Running the Full Pipeline

```bash
# Step 1: Download all datasets
python batch_data_downloader.py

# Step 2: Run basic Spark analytics
python batch_spark_analytics.py

# Step 3: Run advanced analytics (PySpark NLP)
cd advanced_analytics
python advanced_spark_cross_analysis.py
python deep_semantic_analysis.py

# Step 4: Run multi-asset correlation analysis
cd ../economics_social_cultural
python multi_asset_correlation_analysis.py

# Step 5: Launch dashboards
streamlit run batch_dashboard.py --server.port 8502
streamlit run advanced_analytics/advanced_dashboard.py --server.port 8503
```

---

## Technical Fixes Applied

### 1. PySpark 4.0.1 Vector Serialization Issue
**Problem:** Python worker crash at Stage 120 when using `toPandas()` with Vector types.

**Solution:** Use native Spark functions instead of Python UDFs.
```python
# Before (crashes):
get_max_idx = udf(lambda v: int(np.argmax(v)), IntegerType())

# After (works):
from pyspark.ml.functions import vector_to_array
transformed_with_array = transformed.withColumn("topic_array", vector_to_array("topicDistribution"))
```

### 2. Date Format Mismatch in Joins
**Problem:** News dates (2019-01-01) vs Stock dates (1980-12-12 00:00:00-05:00) caused empty joins.

**Solution:** Normalize dates before join.
```python
from pyspark.sql.functions import to_date
news_normalized = news_df.withColumn("date_normalized", to_date(col("date")))
```

### 3. World Bank GDP Indicator Filtering
**Problem:** Multiple indicators mixed (GDP, inflation, etc.) causing incorrect values.

**Solution:** Filter by specific indicator code.
```python
gdp_indicator = 'NY.GDP.MKTP.CD'  # GDP (current US$)
gdp_data = gdp_data.filter(col("indicator_code") == gdp_indicator)
```

### 4. yfinance MultiIndex Columns
**Problem:** New yfinance version returns MultiIndex columns, causing KeyError.

**Solution:** Flatten MultiIndex columns.
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]
```

---

## Technologies Used

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Processing | PySpark | 4.0.1 | Distributed data processing |
| ML | PySpark ML | 4.0.1 | TF-IDF, LDA, CountVectorizer |
| NLP | Gensim | 4.4.0 | Word2Vec embeddings |
| Statistics | scikit-learn | latest | Linear regression |
| Visualization | Plotly | latest | Interactive charts |
| Dashboard | Streamlit | latest | Web interface |
| Data | yfinance | latest | Stock & crypto data |
| Data | wbgapi | latest | World Bank economic data |

---

## Key Results Summary

### Most Influential Financial Terms (PageRank)
1. market (0.0847)
2. stock (0.0723)
3. price (0.0651)
4. growth (0.0589)
5. economy (0.0534)

### Stock Correlations (Top 5)
| Pair | Correlation |
|------|-------------|
| DIA-SPY | 0.951 |
| MSFT-AAPL | 0.923 |
| JPM-GS | 0.912 |

### Multi-Asset Correlations (Top 5)
| Pair | Correlation |
|------|-------------|
| BTC Volume ↔ ETH Volume | 0.991 |
| USA GDP ↔ Korea GDP | 0.977 |
| S&P 500 ↔ Korea Real Estate | 0.964 |
| Bitcoin ↔ Ethereum | 0.957 |
| USA GDP ↔ Korea Real Estate | 0.943 |

---

## Dashboard Screenshots

### Advanced Dashboard (localhost:8503)

**Multi-Asset Correlation Section:**
- GDP comparison time series (USA vs Korea)
- S&P 500 vs Cryptocurrency price charts
- Correlation heatmap (9x9 matrix)
- Interactive linear regression scatter plots
- Decade-by-decade growth analysis

**Deep Semantic Section:**
- TF-IDF word importance by sentiment
- LDA topic distribution pie chart
- Word2Vec similarity network
- Entity-price correlation bar charts

---

## Future Enhancements

1. **Real-time Analysis**: Stream processing with Kafka + Spark Streaming
2. **Sentiment Models**: BERT/FinBERT for advanced sentiment analysis
3. **Korean Data**: KOSPI index, Korean news sources
4. **Portfolio Optimization**: Modern Portfolio Theory integration
5. **Predictive Models**: LSTM for price prediction

---

## License

MIT License

## Author

**Mars** (gogog01-29-2021)

Big Data Practice - November 2024

---

## Changelog

### v2.0 (2024-11-26)
- Added deep semantic analysis (TF-IDF, LDA, Word2Vec)
- Added multi-asset correlation module (economics_social_cultural)
- Fixed PySpark Vector serialization issues
- Fixed date format normalization for joins
- Fixed World Bank GDP indicator filtering
- Added yfinance MultiIndex column handling
- Extended dashboard with Multi-Asset sections

### v1.0 (2024-11-25)
- Initial release with batch analytics
- PageRank and Granger causality analysis
- Basic dashboard implementation

---

## Quick Reference: Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA → ANALYSIS → DASHBOARD                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  STOCK PRICES (stock_prices.csv)                                                 │
│  └─→ Technical Indicators (SMA, RSI, BB)     → Port 8502: Stock Analysis        │
│  └─→ Stock Correlations                       → Port 8502: Correlation Matrix    │
│  └─→ Entity-Price Correlation                → Port 8503: Entity-Price Link     │
│  └─→ Word-Price Causality                    → Port 8503: Word-Price Causality  │
│                                                                                   │
│  NEWS HEADLINES (financial_news_headlines.csv)                                   │
│  └─→ Word Frequency (MapReduce)              → Port 8502: Word Analysis         │
│  └─→ TF-IDF Analysis                         → Port 8503: TF-IDF Analysis       │
│  └─→ LDA Topic Modeling                      → Port 8503: Topic Modeling        │
│  └─→ Word2Vec Embeddings                     → Port 8503: Word2Vec              │
│  └─→ N-gram Analysis                         → Port 8503: N-gram Phrases        │
│  └─→ PageRank (Co-occurrence)                → Port 8503: PageRank              │
│  └─→ Granger Causality                       → Port 8503: Granger Causality     │
│                                                                                   │
│  WORLD BANK (world_bank_indicators.csv)                                          │
│  └─→ Korea vs USA GDP                        → Port 8503: Korea vs USA          │
│  └─→ Multi-Asset Correlation                 → Port 8503: Multi-Asset           │
│                                                                                   │
│  YFINANCE (BTC-USD, ETH-USD, SPY)                                                │
│  └─→ Crypto + S&P 500 Time Series           → Port 8503: Multi-Asset           │
│  └─→ Correlation Matrix                      → Port 8503: Correlation Heatmap   │
│  └─→ Linear Regression                       → Port 8503: Linear Regression     │
│  └─→ Decade Analysis                         → Port 8503: Decade Analysis       │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Unit Summary Table

| Metric | Unit | Range/Example |
|--------|------|---------------|
| Stock Price | USD ($) | $0.12 - $500+ |
| Volume | Shares | 10,000 - 500,000,000 |
| GDP | USD ($) | $9B - $29T |
| Correlation | Pearson r | -1.0 to +1.0 |
| R² | Ratio | 0.0 to 1.0 |
| P-value | Probability | 0.0 to 1.0 |
| PageRank | Probability | 0.0 to 1.0 |
| TF-IDF | Weight | 0.0 to 1.0+ |
| Similarity | Cosine | 0.0 to 1.0 |
| Growth Rate | Percentage (%) | -100% to +1000%+ |
| RSI | Index | 0 to 100 |
| Real Estate Index | Index (1970=100) | 100 to 1100 |

## Files Quick Reference

| File | Location | Size | Purpose |
|------|----------|------|---------|
| stock_prices.csv | data/batch/stocks/ | ~25 MB | Raw stock data |
| financial_news_headlines.csv | data/batch/news/ | ~3 MB | News with sentiment |
| world_bank_indicators.csv | data/batch/worldbank/ | ~8 MB | Economic indicators |
| word_pagerank.csv | advanced_analytics/spark_output/ | ~50 KB | PageRank scores |
| tfidf_by_sentiment.csv | advanced_analytics/deep_semantic_output/ | ~200 KB | TF-IDF weights |
| lda_topics.csv | advanced_analytics/deep_semantic_output/ | ~10 KB | Topic words |
| word2vec_similarities.csv | advanced_analytics/deep_semantic_output/ | ~5 KB | Word similarities |
| entity_price_correlation.csv | advanced_analytics/deep_semantic_output/ | ~1 KB | Entity correlations |
| korea_usa_gdp_comparison.csv | advanced_analytics/deep_semantic_output/ | ~5 KB | GDP comparison |
| correlation_matrix.csv | economics_social_cultural/output/ | ~2 KB | 9x9 correlation |
| linear_regression_results.csv | economics_social_cultural/output/ | ~5 KB | Regression stats |
| decade_analysis.csv | economics_social_cultural/output/ | ~2 KB | Decade growth |
| merged_multi_asset_data.csv | economics_social_cultural/output/ | ~8 KB | Combined time series |
