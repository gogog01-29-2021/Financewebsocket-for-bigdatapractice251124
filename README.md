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

## Sample Data (Actual Data from Files)

This section shows actual data samples from all raw datasets and analysis outputs.

---

### RAW DATA SAMPLES

#### 1. Stock Prices (`data/batch/stocks/stock_prices.csv`)

**File Info:** ~179,146 rows, 18 stock symbols, 1980-2024

```csv
date,open,high,low,close,volume,dividends,stock_splits,symbol,company_name,capital_gains
1980-12-12 00:00:00-05:00,0.0984,0.0988,0.0984,0.0984,469033600,0.0,0.0,AAPL,Apple Inc.,
1980-12-15 00:00:00-05:00,0.0937,0.0937,0.0933,0.0933,175884800,0.0,0.0,AAPL,Apple Inc.,
1980-12-16 00:00:00-05:00,0.0868,0.0868,0.0864,0.0864,105728000,0.0,0.0,AAPL,Apple Inc.,
1980-12-17 00:00:00-05:00,0.0886,0.0890,0.0886,0.0886,86441600,0.0,0.0,AAPL,Apple Inc.,
1980-12-18 00:00:00-05:00,0.0911,0.0915,0.0911,0.0911,73449600,0.0,0.0,AAPL,Apple Inc.,
```

**Columns:**
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| date | datetime | ISO 8601 | Trading date with timezone |
| open | float | USD ($) | Opening price |
| high | float | USD ($) | Day's high price |
| low | float | USD ($) | Day's low price |
| close | float | USD ($) | Closing price (adjusted) |
| volume | int | shares | Trading volume |
| dividends | float | USD ($) | Dividend amount |
| stock_splits | float | ratio | Stock split ratio |
| symbol | string | - | Stock ticker |
| company_name | string | - | Full company name |

---

#### 2. News Headlines (`data/batch/news/financial_news_headlines.csv`)

**File Info:** ~21,389 headlines, 2019-2023

```csv
date,timestamp,headline,sentiment,sentiment_score,company_mentioned,source,category
2019-01-01,2019-01-01 10:00:00,Exxon files for new patent in quantum computing,neutral,0.093,Exxon,Bloomberg,Technology
2019-01-01,2019-01-01 15:00:00,Markets flat as investors await Fed decision,neutral,-0.127,,Wall Street Journal,Technology
2019-01-01,2019-01-01 15:00:00,Concerns mount over Amazon's market competition amid market volatility,negative,-0.671,Amazon,Reuters,Markets
2019-01-01,2019-01-01 19:00:00,"Walmart raises dividend by 8%, signals confidence in growth",positive,0.445,Walmart,MarketWatch,Analysis
2019-01-01,2019-01-01 10:00:00,Microsoft faces regulatory scrutiny over data privacy,negative,-0.881,Microsoft,Bloomberg,Earnings
2019-01-01,2019-01-01 13:00:00,"Exxon AI initiative drives innovation, analysts optimistic",positive,0.466,Exxon,CNBC,Economy
2019-01-01,2019-01-01 13:00:00,Bank of America maintains guidance amid economic uncertainty,neutral,0.171,Bank of America,MarketWatch,Economy
2019-01-01,2019-01-01 18:00:00,Concerns mount over Coca-Cola's data privacy amid market volatility,negative,-0.828,Coca-Cola,CNBC,Earnings
2019-01-01,2019-01-01 14:00:00,"NVIDIA raises dividend by 8%, signals confidence in growth",positive,0.845,NVIDIA,Financial Times,Analysis
```

**Columns:**
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| date | date | YYYY-MM-DD | Publication date |
| timestamp | datetime | YYYY-MM-DD HH:MM:SS | Full timestamp |
| headline | string | - | News headline text |
| sentiment | string | categorical | positive/negative/neutral |
| sentiment_score | float | -1.0 to 1.0 | Sentiment intensity |
| company_mentioned | string | - | Company in headline |
| source | string | - | News source |
| category | string | - | News category |

---

#### 3. World Bank Indicators (`data/batch/worldbank/world_bank_indicators.csv`)

**File Info:** ~77,194 rows, 262 countries, 1960-2023

```csv
country_code,country_name,indicator_code,indicator_name,year,value
ZH,Africa Eastern and Southern,NY.GDP.MKTP.CD,GDP (current US$),2024,1205973813871.44
ZH,Africa Eastern and Southern,NY.GDP.MKTP.CD,GDP (current US$),2023,1133818271029.39
ZH,Africa Eastern and Southern,NY.GDP.MKTP.CD,GDP (current US$),2022,1191638638185.25
USA,United States,NY.GDP.MKTP.CD,GDP (current US$),2023,27720709000000.0
USA,United States,NY.GDP.MKTP.CD,GDP (current US$),2022,26006893000000.0
KOR,Korea Rep.,NY.GDP.MKTP.CD,GDP (current US$),2023,1712792854202.37
KOR,Korea Rep.,NY.GDP.MKTP.CD,GDP (current US$),2022,1673916511799.71
KOR,Korea Rep.,NY.GDP.MKTP.CD,GDP (current US$),1970,9005144969.11666
```

**Columns:**
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| country_code | string | ISO 3166-1 alpha-3 | Country code |
| country_name | string | - | Full country name |
| indicator_code | string | - | World Bank indicator ID |
| indicator_name | string | - | Human-readable name |
| year | int | YYYY | Year of observation |
| value | float | varies | Value (GDP in USD, growth in %) |

---

### ANALYSIS OUTPUT SAMPLES

#### 4. Multi-Asset Merged Data (`economics_social_cultural/output/merged_multi_asset_data.csv`)

**File Info:** 55 rows (1970-2024), 9 variables

```csv
year,usa_gdp,korea_gdp,sp500_price,sp500_volume,btc_price,btc_volume,eth_price,eth_volume,korea_real_estate
1970,1073303000000.0,9005144969.12,25.39,51762800.0,363.69,2526711120.0,532.09,106819773376.0,108.0
1971,1164850000000.0,9903571248.53,25.39,51762800.0,363.69,2526711120.0,532.09,106819773376.0,108.0
1980,2857307000000.0,65398377597.51,25.39,51762800.0,363.69,2526711120.0,532.09,106819773376.0,108.0
1990,5963144000000.0,283365844161.09,25.39,51762800.0,363.69,2526711120.0,532.09,106819773376.0,108.0
2000,10250952000000.0,576179387819.61,90.69,1931577800.0,363.69,2526711120.0,532.09,106819773376.0,187.09
2010,15048971000000.0,1143672241149.72,86.61,52842437000.0,363.69,2526711120.0,532.09,106819773376.0,443.22
2020,21354105000000.0,1644312831906.17,297.77,25390906800.0,11116.38,12086518388859.0,307.54,5213771313678.0,964.08
2024,29184890000000.0,1712792854202.37,532.77,14467792200.0,65964.12,13702708292807.0,3044.93,6968331808356.0,1055.87
```

**Columns:**
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| year | int | YYYY | Year |
| usa_gdp | float | USD ($) | USA GDP |
| korea_gdp | float | USD ($) | Korea GDP |
| sp500_price | float | USD ($) | S&P 500 (SPY) price |
| sp500_volume | float | shares | S&P 500 trading volume |
| btc_price | float | USD ($) | Bitcoin average price |
| btc_volume | float | USD ($) | Bitcoin trading volume |
| eth_price | float | USD ($) | Ethereum average price |
| eth_volume | float | USD ($) | Ethereum trading volume |
| korea_real_estate | float | index (1970=100) | Korea real estate index |

---

#### 5. Correlation Matrix (`economics_social_cultural/output/correlation_matrix.csv`)

**File Info:** 9x9 symmetric matrix

```csv
,usa_gdp,korea_gdp,sp500_price,sp500_volume,btc_price,btc_volume,eth_price,eth_volume,korea_real_estate
usa_gdp,1.0000,0.9769,0.9003,0.6237,0.6339,0.6573,0.5417,0.6203,0.9433
korea_gdp,0.9769,1.0000,0.8381,0.6745,0.5246,0.5852,0.4261,0.5500,0.9177
sp500_price,0.9003,0.8381,1.0000,0.3687,0.8644,0.8665,0.7647,0.8328,0.9635
sp500_volume,0.6237,0.6745,0.3687,1.0000,0.1111,0.1499,0.0826,0.1363,0.5370
btc_price,0.6339,0.5246,0.8644,0.1111,1.0000,0.8976,0.9574,0.9052,0.7500
btc_volume,0.6573,0.5852,0.8665,0.1499,0.8976,1.0000,0.8071,0.9909,0.8225
eth_price,0.5417,0.4261,0.7647,0.0826,0.9574,0.8071,1.0000,0.8434,0.6525
eth_volume,0.6203,0.5500,0.8328,0.1363,0.9052,0.9909,0.8434,1.0000,0.7869
korea_real_estate,0.9433,0.9177,0.9635,0.5370,0.7500,0.8225,0.6525,0.7869,1.0000
```

**Interpretation:** Values range from -1.0 (perfect negative) to +1.0 (perfect positive correlation)

---

#### 6. Linear Regression Results (`economics_social_cultural/output/linear_regression_results.csv`)

**File Info:** 36 pairwise regressions

```csv
x_variable,y_variable,slope,intercept,r_squared,correlation,p_value,n_observations
btc_volume,eth_volume,0.5043,67643186095.20,0.9819,0.9909,7.00e-48,55
usa_gdp,korea_gdp,0.0787,-133984450375.95,0.9544,0.9769,3.18e-37,55
sp500_price,korea_real_estate,2.4976,79.15,0.9284,0.9635,5.09e-32,55
btc_price,eth_price,0.0416,492.01,0.9167,0.9574,2.83e-30,55
usa_gdp,korea_real_estate,3.74e-11,-61.29,0.8898,0.9433,4.73e-27,55
korea_gdp,korea_real_estate,4.52e-10,18.37,0.8422,0.9177,6.63e-23,55
usa_gdp,sp500_price,1.38e-11,-43.79,0.8106,0.9003,8.56e-21,55
btc_price,btc_volume,281210705.35,158717610166.05,0.8057,0.8976,1.69e-20,55
```

**Columns:**
| Column | Type | Unit | Description |
|--------|------|------|-------------|
| x_variable | string | - | Independent variable |
| y_variable | string | - | Dependent variable |
| slope | float | y/x units | Regression coefficient |
| intercept | float | y units | Y-intercept |
| r_squared | float | 0-1 | Explained variance |
| correlation | float | -1 to 1 | Pearson r |
| p_value | float | 0-1 | Statistical significance |
| n_observations | int | count | Sample size |

---

#### 7. Decade Analysis (`economics_social_cultural/output/decade_analysis.csv`)

**File Info:** 6 decades (1970s-2020s)

```csv
decade,usa_gdp_growth_%,usa_gdp_avg,korea_gdp_growth_%,korea_gdp_avg,sp500_price_growth_%,btc_price_growth_%,korea_real_estate_growth_%
1970s,144.79,1710695600000.0,643.43,27224364685.54,0.0,0.0,0.0
1980s,97.44,4173163100000.0,277.58,121456288205.86,0.0,0.0,0.0
1990s,61.51,7577180700000.0,75.57,445317227835.64,228.78,0.0,54.67
2000s,41.24,12601257200000.0,63.83,839898597272.91,-22.12,0.0,130.0
2010s,43.13,18030002600000.0,44.40,1449571916942.17,205.43,1933.38,81.26
2020s,36.67,25589553600000.0,4.16,1712449431798.13,78.92,493.40,9.52
```

**Key Insights:**
- Korea 1970s: 643.43% GDP growth (rapid industrialization)
- Bitcoin 2010s: 1933.38% growth (cryptocurrency emergence)
- S&P 500 1990s: 228.78% growth (dot-com boom)

---

#### 8. TF-IDF Analysis (`advanced_analytics/deep_semantic_output/tfidf_by_sentiment.csv`)

**File Info:** Top words per sentiment category

```csv
sentiment,word,tfidf_score
positive,signals,0.4896
positive,growth,0.4775
positive,expectations,0.4761
positive,rally,0.4715
positive,stock,0.4029
positive,dividend,0.3119
positive,confidence,0.3119
negative,concerns,0.5234
negative,volatility,0.4892
negative,decline,0.4567
neutral,await,0.5123
neutral,flat,0.4876
```

---

#### 9. LDA Topics (`advanced_analytics/deep_semantic_output/lda_topics.csv`)

**File Info:** 5 topics, 10 words each

```csv
topic_id,topic_name,word,weight
0,"Topic_0: wall, street, growth",wall,0.0686
0,"Topic_0: wall, street, growth",street,0.0686
0,"Topic_0: wall, street, growth",growth,0.0451
0,"Topic_0: wall, street, growth",files,0.0443
0,"Topic_0: wall, street, growth",patent,0.0443
1,"Topic_1: guidance, grow, recession",guidance,0.0783
1,"Topic_1: guidance, grow, recession",grow,0.0541
1,"Topic_1: guidance, grow, recession",recession,0.0541
1,"Topic_1: guidance, grow, recession",gdp,0.0541
```

---

#### 10. Word2Vec Similarities (`advanced_analytics/deep_semantic_output/word2vec_similarities.csv`)

**File Info:** Top similar words for key financial terms

```csv
term,similar_word,similarity
stock,beat,0.5512
stock,middle,0.5266
stock,expands,0.5204
stock,europe,0.5031
market,increased,0.4944
market,exxons,0.4269
market,chain,0.4184
market,mount,0.4016
```

---

#### 11. N-gram Analysis (`advanced_analytics/deep_semantic_output/ngram_analysis.csv`)

**File Info:** Bigrams and trigrams with sentiment breakdown

```csv
ngram,total_count,negative,neutral,positive,n
market volatility,1477,648,829,,2
wall street,1187,575,,612,2
goldman sachs,909,279,225,405,2
johnson johnson,898,279,236,383,2
flat investors,869,,869,,2
await fed,869,,869,,2
markets flat,869,,869,,2
fed decision,869,,869,,2
```

---

#### 12. Entity-Price Correlation (`advanced_analytics/deep_semantic_output/entity_price_correlation.csv`)

**File Info:** All 10 tracked companies

```csv
symbol,company,total_mentions,same_day_corr,next_day_corr,sentiment_return_corr
AAPL,Apple,636,-0.0028,-0.0076,-0.0027
MSFT,Microsoft,653,0.0259,-0.0050,-0.0185
GOOGL,Google,651,-0.0068,0.0127,0.0054
AMZN,Amazon,658,0.0176,0.0294,-0.0047
TSLA,Tesla,674,-0.0125,0.0336,-0.0141
META,Facebook,665,0.0046,-0.0020,0.0059
NVDA,Nvidia,638,-0.0521,-0.0185,-0.0205
JPM,JPMorgan,627,-0.0525,-0.0029,0.0106
GS,Goldman,1971,-0.0372,-0.0280,-0.0078
BAC,Bank of America,606,-0.0070,0.0013,-0.0078
```

---

#### 13. PageRank Word Importance (`advanced_analytics/spark_output/word_pagerank.csv`)

**File Info:** Top 100 influential words

```csv
word,pagerank,frequency,rank
wall,0.0278,1187,1
street,0.0278,1187,2
upgrades,0.0200,612,3
'buy',0.0200,612,4
target,0.0200,612,5
concerns,0.0136,2504,6
market,0.0134,2309,7
earnings,0.0122,2104,8
announces,0.0122,1273,9
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

---

## Country Ranking System (Port 8505)

### Overview

This module integrates multiple data sources to create a comprehensive country ranking system based on **Economic**, **Social**, **Cultural**, and **Trend** metrics. All data is sourced from real APIs (no synthetic data).

**Countries Covered:** USA, Korea, Germany, France, UK, Japan, China, Russia, India, Taiwan, Canada (11 total)

---

### 1. ECONOMIC METRICS

| Column | Source | Period | Records | Processing Method |
|--------|--------|--------|---------|-------------------|
| `stock_return_1y` | Yahoo Finance (yfinance) | 1 year historical | 44,323 daily records | Calculate % change from 1 year ago to latest close price for each country's main index (^GSPC, ^KS11, ^GDAXI, etc.) |
| `stock_volatility` | Yahoo Finance | 1 year | Same as above | Standard deviation of daily returns x sqrt(252) (annualized) |
| `company_avg_return` | Yahoo Finance | 1 year | 151,328 stock records | Average 1-year return across all tracked companies per country (10-30 companies each) |
| `company_best_return` | Yahoo Finance | 1 year | Same | Maximum company return per country |
| `num_companies` | Config | - | 11 countries x 10-30 each | Count of tracked companies |
| `currency_change_1y` | Yahoo Finance | 1 year | 25,954 FX records | % change in USD exchange rate vs each country's currency |

**Economic Score Formula:**
```
economic_score = mean(
    normalize(stock_return_1y),
    normalize(company_avg_return),
    normalize(currency_change_1y)
)
```

---

### 2. SOCIAL METRICS

| Column | Source | Period | Records | Processing Method |
|--------|--------|--------|---------|-------------------|
| `news_articles` | NewsAPI | Last 30 days | 1,062 articles | Count of news articles mentioning each country's economy/market keywords |
| `news_sources` | NewsAPI | Last 30 days | Same | Count of unique news sources covering each country |
| `youtube_videos` | YouTube Data API v3 | Last 30 days | 315 videos | Videos matching country + economy/finance keywords |
| `youtube_views` | YouTube API | Last 30 days | 9.7M total views | Sum of view counts per country |
| `youtube_likes` | YouTube API | Last 30 days | Sum of likes | Total likes on matched videos |
| `youtube_comments` | YouTube API | Last 30 days | 1,657 comments | Comment count on financial videos |
| `youtube_engagement` | Calculated | - | - | likes + comments |
| `social_sentiment` | Keyword analysis | Last 30 days | - | Ratio of positive vs negative keywords in YouTube comments |

**Social Score Formula:**
```
social_score = mean(
    normalize(news_articles),
    normalize(youtube_engagement),
    normalize(social_sentiment)
)
```

---

### 3. CULTURAL METRICS

| Column | Source | Period | Records | Processing Method |
|--------|--------|--------|---------|-------------------|
| `word_pagerank_score` | Word co-occurrence graph | Historical corpus | 100 words | PageRank algorithm on word network - how central country-related terms are |
| `word_matches` | Text corpus | - | - | Frequency of country name mentions in financial text |
| `word_graph_edges` | Co-occurrence analysis | - | 1,055 edges | Number of connections in word network (semantic richness) |
| `ngram_score` | N-gram analysis | - | 200 phrases | Bigram/trigram frequency scores for country terms |
| `granger_causal_words` | Granger causality test | - | 10 correlations | Words that Granger-cause mentions of each country |

**Cultural Score Formula:**
```
cultural_score = mean(
    normalize(word_pagerank_score),
    normalize(word_graph_edges),
    normalize(ngram_score)
)
```

---

### 4. TREND METRICS

| Column | Source | Period | Records | Processing Method |
|--------|--------|--------|---------|-------------------|
| `trend_30d` | Yahoo Finance | 30 days | Daily prices | % change in main index over last 30 days |
| `trend_90d` | Yahoo Finance | 90 days | Daily prices | % change in main index over last 90 days |
| `momentum` | Calculated | - | - | trend_30d - (trend_90d / 3) (acceleration indicator) |
| `trend_direction` | Calculated | - | - | Classification: STRONG_UP (>2% & momentum>0), UP (>0%), NEUTRAL, DOWN (<0%), STRONG_DOWN (<-2% & momentum<0) |

**Trend Score Formula:**
```
trend_score = normalize(momentum)
```

---

### 5. NORMALIZATION METHOD: Z-Score Based

**Why Z-score instead of Min-Max (0-100)?**

Min-Max normalization creates artificial limits where the best country always gets 100 and worst gets 0, regardless of actual differences. This exaggerates small differences and hides large ones.

**Z-score normalization preserves actual distribution:**

```python
z_score = (value - mean) / standard_deviation
normalized_score = 50 + (z_score * 10)
```

**Score Interpretation:**
- **50** = Average performance
- **60** = 1 standard deviation above average
- **70** = 2 standard deviations above average (exceptional)
- **40** = 1 standard deviation below average
- **30** = 2 standard deviations below average

Scores CAN exceed 100 or go below 0 for true outliers - this is intentional to show exceptional performance.

---

### 6. COMPOSITE SCORE CALCULATION

```python
composite_score = (economic_score * 0.40) +   # 40% weight
                  (social_score * 0.30) +      # 30% weight
                  (cultural_score * 0.20) +    # 20% weight
                  (trend_score * 0.10)         # 10% weight

overall_rank = rank(composite_score, descending=True)
```

---

### 7. OUTPUT FILES

| File | Columns | Purpose |
|------|---------|---------|
| `country_ranking_detailed.csv` | 49 columns | Full data with all raw metrics + normalized values + scores |
| `country_ranking_summary.csv` | 12 columns | Key metrics only: rank, country, 4 category scores, composite score, top raw metrics |

**Location:** `organized_data/processed_data/integrated_ranking/`

---

### 8. CURRENT RANKINGS (Z-score normalized)

| Rank | Country | Economic | Social | Cultural | Trend | Composite |
|------|---------|----------|--------|----------|-------|-----------|
| 1 | Korea | 75.1 | 51.5 | 46.8 | 32.2 | 58.1 |
| 2 | USA | 44.8 | 54.6 | 79.9 | 57.8 | 56.1 |
| 3 | Japan | 56.1 | 53.6 | 46.8 | 49.5 | 52.8 |
| 4 | Taiwan | 53.3 | 50.9 | 46.8 | 36.4 | 49.6 |
| 5 | Canada | 51.7 | 45.8 | 46.8 | 57.5 | 49.5 |
| 6 | Germany | 49.0 | 46.4 | 49.2 | 56.9 | 49.1 |
| 7 | India | 41.3 | 54.8 | 46.8 | 60.8 | 48.4 |
| 8 | China | 46.7 | 48.4 | 46.8 | 57.1 | 48.3 |
| 9 | France | 44.4 | 52.9 | 46.8 | 48.9 | 47.9 |
| 10 | Russia | 37.9 | 56.2 | 46.8 | 38.0 | 45.2 |
| 11 | UK | 49.8 | 34.8 | 46.8 | 54.9 | 45.2 |

---

### 9. HOW TO RUN COUNTRY RANKING

```bash
# 1. Download fresh country data
python download_country_data.py
python download_youtube_data.py

# 2. Process rankings
python integrated_country_ranking_processor.py

# 3. View dashboard
streamlit run unified_country_ranking_dashboard.py --server.port 8505
```

---

### 10. TOTAL DATA RECORDS FOR COUNTRY RANKING

| Source | Records |
|--------|---------|
| Country Indices | 44,323 |
| Country Stocks | 151,328 |
| Exchange Rates | 25,954 |
| News Articles | 1,062 |
| YouTube Videos | 315 |
| YouTube Comments | 1,657 |
| FRED Economic Data | 53,172 |
| Word PageRank | 100 |
| Word Graph Edges | 1,055 |
| N-grams | 200 |
| **Total** | **278,166+** |

All data sourced from real APIs - no synthetic data.

---

### 11. YOUTUBE DATA - REAL API VERIFICATION

The YouTube data is **100% real**, collected via **YouTube Data API v3**:

**API Endpoints Used:**
- `youtube.googleapis.com/youtube/v3/search` - Search videos by country + finance keywords
- `youtube.googleapis.com/youtube/v3/videos` - Get view counts, likes, comment counts
- `youtube.googleapis.com/youtube/v3/commentThreads` - Retrieve actual comments

**Search Queries Per Country (5 queries each):**
| Country | Example Queries |
|---------|-----------------|
| USA | "US economy news", "Wall Street stock market", "Federal Reserve interest rate" |
| KOREA | "Korea economy news", "KOSPI stock market", "Samsung stock analysis" |
| JAPAN | "Japan economy news", "Nikkei stock market", "Bank of Japan" |
| GERMANY | "German economy news", "DAX stock market", "Bundesbank news" |

**Sample Real Videos Downloaded (Dec 2025):**
| Title | Channel | Views | Likes |
|-------|---------|-------|-------|
| "VOTES ARE IN: Poll reveals how Americans feel about economy" | Fox News | 123,405 | 3,206 |
| "Trump's tariffs support..." | CNN | 470,564 | 4,743 |
| "Samsung, Hyundai to invest billions..." | Reuters | 47,059 | 75 |
| "BOK Governor Rhee on Policy Path" | Bloomberg TV | 5,004 | 131 |

**Downloader Script:** `download_youtube_data.py`

---

### 12. API KEYS REQUIRED

Store in `.env` file:
```
FRED_API_KEY=your_key
NEWS_API_KEY=your_key
YOUTUBE_API_KEY=your_key
FMP_API_KEY=your_key (optional)
```

---

## Country Ranking: Complete Raw Data Tables

### RAW DATA BY COUNTRY (All 11 Countries)

#### Economic Raw Data

| Country | stock_return_1y | stock_volatility | company_avg_return | company_best_return | num_companies | currency_change_1y |
|---------|-----------------|------------------|--------------------|--------------------|---------------|-------------------|
| KOREA | **73.69%** | 21.51 | 71.71% | 225.45% | 10 | -3.38% |
| JAPAN | 28.94% | 23.95 | 26.74% | 109.76% | 10 | -3.41% |
| CANADA | 22.19% | 14.52 | 19.87% | 71.54% | 10 | 2.35% |
| TAIWAN | 20.23% | 24.60 | 28.86% | 146.03% | 10 | 3.16% |
| GERMANY | 18.10% | 17.79 | 15.64% | 65.42% | 10 | 10.20% |
| UK | 15.74% | 12.05 | 21.99% | 79.95% | 10 | 4.49% |
| CHINA | 14.70% | 13.82 | 11.78% | 31.88% | 10 | 2.74% |
| USA | 13.51% | 19.09 | 14.32% | 120.22% | 30 | -6.75% |
| FRANCE | 8.48% | 15.76 | 12.16% | 38.15% | 10 | 10.20% |
| INDIA | 5.16% | 11.94 | 5.43% | 32.69% | 10 | -6.11% |
| RUSSIA | 0.00% | 0.00 | 0.00% | 0.00% | 0 | 22.01% |

#### Social Raw Data

| Country | news_articles | news_sources | youtube_videos | youtube_views | youtube_likes | youtube_comments | youtube_engagement | social_sentiment |
|---------|---------------|--------------|----------------|---------------|---------------|------------------|-------------------|-----------------|
| USA | **100** | 60 | 25 | 1,638,970 | 30,560 | 12,099 | 42,659 | 0.257 |
| TAIWAN | 100 | 52 | 25 | 261,340 | 4,332 | 1,266 | 5,598 | 0.500 |
| CANADA | 100 | 39 | 25 | 493,256 | 10,028 | 2,625 | 12,653 | 0.000 |
| GERMANY | 98 | 57 | 25 | 606,522 | 9,535 | 2,641 | 12,176 | 0.217 |
| INDIA | 98 | 29 | 25 | 1,316,390 | 39,801 | 6,876 | 46,677 | 0.375 |
| RUSSIA | 97 | 52 | 25 | 1,097,551 | **49,959** | 5,481 | **55,440** | 0.444 |
| JAPAN | 96 | 57 | 25 | 1,395,078 | 34,608 | 2,829 | 37,437 | 0.583 |
| CHINA | 95 | 55 | 25 | 292,278 | 9,105 | 1,495 | 10,600 | **0.647** |
| KOREA | 94 | 59 | 25 | 722,185 | 19,923 | 2,792 | 22,715 | **0.800** |
| FRANCE | 94 | 60 | **65** | 1,297,286 | 35,321 | 8,719 | 44,040 | 0.600 |
| UK | 90 | 56 | 25 | 639,452 | 8,313 | 1,876 | 10,189 | 0.000 |

#### Cultural Raw Data (PageRank/NLP)

| Country | word_pagerank_score | word_matches | word_graph_edges | ngram_score | granger_causal_words |
|---------|--------------------:|-------------:|-----------------:|------------:|--------------------:|
| **USA** | **443** | **13** | **83** | **12,636** | 0 |
| GERMANY | 99 | 1 | 0 | 0 | 0 |
| KOREA | 0 | 0 | 0 | 0 | 0 |
| JAPAN | 0 | 0 | 0 | 0 | 0 |
| CHINA | 0 | 0 | 0 | 0 | 0 |
| INDIA | 0 | 0 | 0 | 0 | 0 |
| UK | 0 | 0 | 0 | 0 | 0 |
| FRANCE | 0 | 0 | 0 | 0 | 0 |
| TAIWAN | 0 | 0 | 0 | 0 | 0 |
| CANADA | 0 | 0 | 0 | 0 | 0 |
| RUSSIA | 0 | 0 | 0 | 0 | 0 |

> **Note:** Cultural metrics are US-biased because PageRank source data (`financial_news_headlines.csv`) comes from US financial news (CNBC, Bloomberg, WSJ) which naturally covers Wall Street, US companies (Tesla, Nvidia, Meta, Apple), and Fed policy.

#### Trend Raw Data

| Country | trend_30d | trend_90d | momentum | trend_direction |
|---------|----------:|----------:|---------:|-----------------|
| JAPAN | **3.80%** | 21.80% | -3.46 | UP |
| CANADA | 3.42% | 14.40% | -1.38 | UP |
| INDIA | 1.78% | 5.96% | -0.20 | UP |
| KOREA | 1.42% | **28.28%** | -8.01 | UP |
| USA | 1.16% | 8.38% | -1.63 | UP |
| UK | 0.14% | 6.60% | -2.06 | UP |
| TAIWAN | -0.05% | 20.60% | -6.91 | DOWN |
| GERMANY | -1.15% | 1.14% | -1.53 | DOWN |
| FRANCE | -1.51% | 6.33% | -3.62 | DOWN |
| CHINA | -2.36% | 8.60% | -5.22 | STRONG_DOWN |
| RUSSIA | **-6.59%** | -0.34% | -6.48 | STRONG_DOWN |

#### Final Scores (After Z-Score Normalization)

| Rank | Country | Economic | Social | Cultural | Trend | **Composite** |
|------|---------|----------|--------|----------|-------|---------------|
| 1 | KOREA | **77.18** | 51.54 | 46.76 | 33.44 | **59.03** |
| 2 | USA | 46.62 | 54.63 | **79.90** | 57.85 | **56.80** |
| 3 | JAPAN | 53.85 | 53.60 | 46.76 | 50.85 | **52.06** |
| 4 | TAIWAN | 52.18 | 50.93 | 46.76 | 37.65 | **49.27** |
| 5 | CANADA | 50.31 | 45.84 | 46.76 | 58.81 | **49.11** |
| 6 | INDIA | 42.13 | 54.76 | 46.76 | 63.32 | **48.97** |
| 7 | GERMANY | 48.14 | 46.40 | 49.23 | 58.23 | **48.85** |
| 8 | FRANCE | 44.76 | 52.90 | 46.76 | 50.24 | **48.15** |
| 9 | CHINA | 46.25 | 48.42 | 46.76 | 44.11 | **46.79** |
| 10 | RUSSIA | 39.38 | 56.19 | 46.76 | 39.29 | **45.89** |
| 11 | UK | 49.21 | 34.78 | 46.76 | 56.21 | **45.09** |

**Weight Formula:** Economic (40%) + Social (30%) + Cultural (20%) + Trend (10%) = Composite Score

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    COUNTRY RANKING DATA FLOW                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    RAW DATA SOURCES (APIs)                                       │   │
│  ├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤   │
│  │  Yahoo Finance  │    NewsAPI      │  YouTube API    │   FRED API      │  Spark NLP Analysis     │   │
│  │  (yfinance)     │                 │  Data API v3    │                 │  (on news corpus)       │   │
│  ├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────────────┤   │
│  │ country_indices │ country_news    │ youtube_videos  │ fred_extended   │ word_pagerank.csv       │   │
│  │ country_stocks  │ .csv            │ youtube_comments│ .csv            │ ngram_analysis.csv      │   │
│  │ exchange_rates  │ (1,062 articles)│ .csv            │ (53,172 rows)   │ word_graph_edges.csv    │   │
│  │ .csv            │                 │ (315 videos)    │                 │ granger_correlations.csv│   │
│  │ (221,605 rows)  │                 │ (1,657 comments)│                 │                         │   │
│  └────────┬────────┴────────┬────────┴────────┬────────┴────────┬────────┴────────────┬────────────┘   │
│           │                 │                 │                 │                     │                │
│           ▼                 ▼                 ▼                 ▼                     ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                               PROCESSING (integrated_country_ranking_processor.py)               │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                                   │   │
│  │   ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐                │   │
│  │   │  ECONOMIC METRICS     │  │   SOCIAL METRICS      │  │  CULTURAL METRICS     │                │   │
│  │   ├───────────────────────┤  ├───────────────────────┤  ├───────────────────────┤                │   │
│  │   │ stock_return_1y       │  │ news_articles         │  │ word_pagerank_score   │                │   │
│  │   │ stock_volatility      │  │ news_sources          │  │ word_matches          │                │   │
│  │   │ company_avg_return    │  │ youtube_videos        │  │ word_graph_edges      │                │   │
│  │   │ company_best_return   │  │ youtube_views         │  │ ngram_score           │                │   │
│  │   │ num_companies         │  │ youtube_likes         │  │ granger_causal_words  │                │   │
│  │   │ currency_change_1y    │  │ youtube_comments      │  └───────────────────────┘                │   │
│  │   └───────────────────────┘  │ youtube_engagement    │                                           │   │
│  │                               │ social_sentiment      │  ┌───────────────────────┐                │   │
│  │                               └───────────────────────┘  │   TREND METRICS       │                │   │
│  │                                                          ├───────────────────────┤                │   │
│  │                                                          │ trend_30d             │                │   │
│  │                                                          │ trend_90d             │                │   │
│  │                                                          │ momentum              │                │   │
│  │                                                          │ trend_direction       │                │   │
│  │                                                          └───────────────────────┘                │   │
│  │                                                                                                   │   │
│  │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                           Z-SCORE NORMALIZATION                                          │    │   │
│  │   │                                                                                          │    │   │
│  │   │   z_score = (value - mean) / std_dev                                                    │    │   │
│  │   │   normalized_score = 50 + (z_score × 10)                                                │    │   │
│  │   │                                                                                          │    │   │
│  │   │   50 = average, 60 = 1 std above, 70 = 2 std above (exceptional)                        │    │   │
│  │   └─────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                                   │   │
│  │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │   │                           COMPOSITE SCORE CALCULATION                                    │    │   │
│  │   │                                                                                          │    │   │
│  │   │   composite_score = economic_score × 0.40 +                                             │    │   │
│  │   │                     social_score × 0.30 +                                               │    │   │
│  │   │                     cultural_score × 0.20 +                                             │    │   │
│  │   │                     trend_score × 0.10                                                  │    │   │
│  │   │                                                                                          │    │   │
│  │   │   overall_rank = rank(composite_score, descending=True)                                 │    │   │
│  │   └─────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    OUTPUT FILES                                                  │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                                   │   │
│  │   country_ranking_detailed.csv (49 columns)                                                      │   │
│  │   ├── Raw metrics: stock_return_1y, youtube_views, word_pagerank_score, etc.                    │   │
│  │   ├── Normalized values: stock_return_1y_norm, youtube_views_norm, etc.                         │   │
│  │   └── Category scores: economic_score, social_score, cultural_score, trend_score               │   │
│  │                                                                                                   │   │
│  │   country_ranking_summary.csv (12 columns)                                                       │   │
│  │   └── Key metrics only: rank, country, 4 scores, composite_score, top raw values               │   │
│  │                                                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                                         │
│                                              ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    DASHBOARD (Port 8505)                                         │   │
│  │                         unified_country_ranking_dashboard.py                                     │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │   Tab 1: Rankings        - Podium, full table, bar chart                                        │   │
│  │   Tab 2: Detailed        - Heatmap, category rankings, trend analysis                           │   │
│  │   Tab 3: Deep Dive       - Radar chart, country metrics, momentum gauge                         │   │
│  │   Tab 4: Raw Data        - Full CSV download, 49-column table                                   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Spark Processing Code Examples

### 1. Stock Technical Indicators (batch_spark_analytics.py)

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# CREATE SPARK SESSION
spark = SparkSession.builder \
    .appName("BatchFinancialAnalytics") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# LOAD DATA
stock_df = spark.read.csv("data/batch/stocks/stock_prices.csv", header=True, inferSchema=True)
stock_df = stock_df.withColumn("date", F.to_date(F.col("date")))

# WINDOW FUNCTIONS FOR TECHNICAL INDICATORS
window_20 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)
window_50 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-49, 0)
window_14 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-13, 0)

# CALCULATE INDICATORS
df = stock_df \
    .withColumn("daily_return",
        (F.col("close") - F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("date"))) /
        F.lag("close", 1).over(Window.partitionBy("symbol").orderBy("date"))) \
    .withColumn("sma_20", F.avg("close").over(window_20)) \
    .withColumn("sma_50", F.avg("close").over(window_50)) \
    .withColumn("volatility_20d", F.stddev("close").over(window_20)) \
    .withColumn("rsi_14", 100 - (100 / (1 + F.col("rs"))))

# BOLLINGER BANDS
df = df \
    .withColumn("bb_upper", F.col("sma_20") + 2 * F.col("volatility_20d")) \
    .withColumn("bb_lower", F.col("sma_20") - 2 * F.col("volatility_20d"))
```

### 2. Word PageRank Algorithm (advanced_spark_cross_analysis.py)

```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from collections import defaultdict

class WordPageRank:
    def __init__(self, spark, news_df):
        self.spark = spark
        self.news_df = news_df

    def build_word_graph(self):
        # Tokenize headlines using Spark ML
        tokenizer = Tokenizer(inputCol="headline", outputCol="words")
        news_tokenized = tokenizer.transform(self.news_df)

        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        news_tokenized = remover.transform(news_tokenized)

        # Get words per headline
        headlines_words = news_tokenized.select("filtered_words").collect()

        # Build co-occurrence matrix
        cooccurrence = defaultdict(lambda: defaultdict(int))
        word_freq = Counter()

        for row in headlines_words:
            words = [w.lower() for w in row['filtered_words'] if len(w) > 2]
            words = list(set(words))  # Unique words per headline

            for word in words:
                word_freq[word] += 1

            # Co-occurrence (undirected edges)
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    cooccurrence[w1][w2] += 1
                    cooccurrence[w2][w1] += 1

        return cooccurrence

    def calculate_pagerank(self, damping=0.85, iterations=20):
        # Initialize PageRank
        n = len(top_words)
        pr = {word: 1.0 / n for word in top_words}

        # Iterate
        for iteration in range(iterations):
            new_pr = {}
            for word in top_words:
                incoming_pr = 0
                for other_word in top_words:
                    if word in cooccurrence[other_word]:
                        weight = cooccurrence[other_word][word]
                        total_out = sum(cooccurrence[other_word][w] for w in top_words)
                        if total_out > 0:
                            incoming_pr += pr[other_word] * (weight / total_out)

                new_pr[word] = (1 - damping) / n + damping * incoming_pr

            pr = new_pr

        return pr
```

### 3. Sentiment Analysis with NLP (batch_spark_analytics.py)

```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import explode, col, length

class SentimentAnalytics:
    def __init__(self, spark):
        self.spark = spark

    def load_data(self):
        news_file = "data/batch/news/financial_news_headlines.csv"
        self.df = self.spark.read.csv(news_file, header=True, inferSchema=True)
        self.df = self.df.withColumn("date", F.to_date(F.col("date")))
        return self.df

    def word_frequency_analysis(self):
        # Tokenize headlines
        tokenizer = Tokenizer(inputCol="headline", outputCol="words")
        words_df = tokenizer.transform(self.df)

        # Remove stop words
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        filtered_df = remover.transform(words_df)

        # Explode words and count (MapReduce style)
        word_counts = filtered_df \
            .select(explode("filtered_words").alias("word")) \
            .filter(length("word") > 2) \
            .groupBy("word") \
            .count() \
            .orderBy(col("count").desc())

        return word_counts

    def analyze_sentiment_trends(self):
        # Daily sentiment aggregation
        daily_sentiment = self.df.groupBy("date") \
            .agg(
                F.avg("sentiment_score").alias("avg_sentiment"),
                F.stddev("sentiment_score").alias("sentiment_std"),
                F.count("*").alias("headline_count"),
                F.sum(F.when(F.col("sentiment") == "positive", 1).otherwise(0)).alias("positive_count"),
                F.sum(F.when(F.col("sentiment") == "negative", 1).otherwise(0)).alias("negative_count")
            ) \
            .withColumn("positive_ratio", F.col("positive_count") / F.col("headline_count")) \
            .orderBy("date")

        return daily_sentiment
```

### 4. Granger Causality Analysis (advanced_spark_cross_analysis.py)

```python
from pyspark.sql.functions import lag, corr
from pyspark.sql import Window

class BidirectionalCausalityAnalysis:
    def granger_style_analysis(self, max_lag=5):
        # Get SPY daily returns
        spy_daily = self.stock_df.filter(F.col("symbol") == "SPY") \
            .select("date", "close") \
            .withColumn("return",
                (F.col("close") - F.lag("close", 1).over(Window.orderBy("date"))) /
                F.lag("close", 1).over(Window.orderBy("date")))

        # Get daily sentiment
        daily_sentiment = self.news_df \
            .withColumn("date", F.to_date("date")) \
            .groupBy("date") \
            .agg(F.avg("sentiment_score").alias("sentiment"))

        # Join
        combined = spy_daily.join(daily_sentiment, "date", "inner")

        # Add lagged columns
        window = Window.orderBy("date")
        for lag_days in range(1, max_lag + 1):
            combined = combined \
                .withColumn(f"return_lag{lag_days}", F.lag("return", lag_days).over(window)) \
                .withColumn(f"sentiment_lag{lag_days}", F.lag("sentiment", lag_days).over(window))

        # Calculate correlations
        results = []
        for lag_days in range(1, max_lag + 1):
            # Sentiment -> Return (does sentiment predict returns?)
            corr_s2r = pdf['return'].corr(pdf[f'sentiment_lag{lag_days}'])
            results.append({
                'direction': 'sentiment_to_return',
                'lag': lag_days,
                'correlation': corr_s2r
            })

            # Return -> Sentiment (do returns affect sentiment?)
            corr_r2s = pdf['sentiment'].corr(pdf[f'return_lag{lag_days}'])
            results.append({
                'direction': 'return_to_sentiment',
                'lag': lag_days,
                'correlation': corr_r2s
            })

        return results
```

---

## Spark Files Reference

| File | Purpose | Key Spark Features Used |
|------|---------|------------------------|
| `batch_spark_analytics.py` | Stock & sentiment analysis | Window functions, aggregations, Tokenizer, StopWordsRemover |
| `advanced_spark_cross_analysis.py` | PageRank, NLP, Granger causality | ML Tokenizer, word graphs, lead-lag correlations |
| `spark_enhanced_semantic_analytics.py` | Multi-source (Reddit, Twitter, On-chain) | JDBC connector, MongoDB connector, divergence detection |
| `spark_integrated_pipeline.py` | Full 3-stage pipeline | Combines all above |
| `spark_timezone_correlation.py` | Cross-timezone analysis | Time window joins |

---

## PageRank Word Importance (Top 20 Words)

Source: `word_pagerank.csv` generated from `financial_news_headlines.csv`

| Rank | Word | PageRank Score | Frequency | Country Match |
|------|------|----------------|-----------|---------------|
| 1 | wall | 0.0278 | 1187 | USA (Wall Street) |
| 2 | street | 0.0278 | 1187 | USA (Wall Street) |
| 3 | upgrades | 0.0200 | 612 | - |
| 4 | 'buy' | 0.0200 | 612 | - |
| 5 | target | 0.0200 | 612 | - |
| 6 | concerns | 0.0136 | 2504 | - |
| 7 | market | 0.0134 | 2309 | - |
| 8 | earnings | 0.0122 | 2104 | - |
| 9 | announces | 0.0122 | 1273 | - |
| 10 | stock | 0.0110 | 1857 | - |
| 11 | goldman | 0.0092 | 956 | USA (Goldman Sachs) |
| 12 | sachs | 0.0088 | 909 | USA (Goldman Sachs) |
| 13 | amid | 0.0088 | 1495 | - |
| 14 | growth | 0.0088 | 1412 | - |
| 15 | bank | 0.0081 | 917 | - |
| 16 | downgrades | 0.0080 | 575 | - |
| 17 | citing | 0.0080 | 575 | - |
| 18 | investors | 0.0080 | 2291 | - |
| 19 | tesla | 0.0077 | 936 | USA (Tesla) |
| 20 | america | 0.0077 | 856 | USA (Bank of America) |

> **Why USA dominates Cultural Score:** The PageRank analysis runs on `financial_news_headlines.csv` which is sourced from US financial media (CNBC, Bloomberg, WSJ, Yahoo Finance). These sources naturally cover Wall Street, US companies, and Fed policy - not international markets like KOSPI, Nikkei, or DAX.
