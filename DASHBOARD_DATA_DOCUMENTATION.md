# 대시보드 데이터 출처 및 처리 방법 문서
# Dashboard Data Source & Processing Documentation

---

# Ⅰ. Advanced Dashboard (Port 8503)

## 1. 🏆 PageRank Word Importance

### 1.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/news/financial_news_headlines.csv` |
| **데이터 소스** | NewsAPI (https://newsapi.org) |
| **API 엔드포인트** | `https://newsapi.org/v2/everything`, `https://newsapi.org/v2/top-headlines` |
| **수집 컬럼** | `headline`, `sentiment`, `sentiment_score`, `date`, `source` |
| **레코드 수** | ~221개 실제 뉴스 헤드라인 |

### 1.2 처리 스크립트 (Processing Script)
**파일:** `advanced_analytics/advanced_spark_cross_analysis.py` (Lines 111-243)

### 1.3 처리 코드 스니펫 (Code Snippet)
```python
# Step 1: Tokenize headlines into words
tokenizer = Tokenizer(inputCol="headline", outputCol="words")
words_df = tokenizer.transform(news_df)

# Step 2: Remove common stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(words_df)

# Step 3: Build word co-occurrence graph
cooccurrence = defaultdict(lambda: defaultdict(int))
for row in headlines_words:
    words = [w.lower() for w in row['filtered_words'] if len(w) > 2]
    words = list(set(words))  # Unique words per headline

    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            cooccurrence[w1][w2] += 1  # w1 -> w2 edge
            cooccurrence[w2][w1] += 1  # w2 -> w1 edge

# Step 4: PageRank algorithm (20 iterations)
damping = 0.85
n = len(top_words)
pr = {word: 1.0 / n for word in top_words}  # Initial: equal probability

for iteration in range(20):
    new_pr = {}
    for word in top_words:
        incoming_pr = 0
        for other_word in top_words:
            if word in cooccurrence[other_word]:
                weight = cooccurrence[other_word][word]
                total_out = sum(cooccurrence[other_word].values())
                incoming_pr += pr[other_word] * (weight / total_out)

        new_pr[word] = (1 - damping) / n + damping * incoming_pr
    pr = new_pr
```

### 1.4 출력 파일 (Output Files)
| 파일 | 내용 |
|------|------|
| `spark_output/word_pagerank.csv` | word, pagerank, frequency, rank |
| `spark_output/word_graph_edges.csv` | source, target, weight |

---

## 2. 🕸️ Word Network Graph

### 2.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/news/financial_news_headlines.csv` |
| **데이터 소스** | NewsAPI |
| **수집 컬럼** | `headline` |

### 2.2 처리 스크립트
**파일:** `advanced_analytics/advanced_spark_cross_analysis.py` (Lines 245-320)

### 2.3 처리 코드 스니펫
```python
# Build NetworkX graph from co-occurrence matrix
import networkx as nx

G = nx.Graph()
for word1, connections in cooccurrence.items():
    for word2, weight in connections.items():
        if weight >= min_weight:
            G.add_edge(word1, word2, weight=weight)

# Calculate network metrics
density = nx.density(G)
centrality = nx.degree_centrality(G)

# Spring layout for visualization
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
```

### 2.4 시각화 라이브러리
- **Plotly**: Interactive network graph
- **NetworkX**: Graph computation and layout

---

## 3. 🔄 Granger Causality Analysis

### 3.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **뉴스 감성** | `data/batch/news/financial_news_headlines.csv` |
| **주가 데이터** | `data/batch/stocks/stock_prices.csv` (SPY ETF) |
| **데이터 소스** | NewsAPI + Yahoo Finance (yfinance) |

### 3.2 처리 스크립트
**파일:** `advanced_analytics/advanced_spark_cross_analysis.py` (Lines 708-769)

### 3.3 처리 코드 스니펫
```python
# Step 1: Get daily returns for SPY
window = Window.orderBy("date")
spy_daily = stock_df.filter(F.col("symbol") == "SPY") \
    .withColumn("return",
        (F.col("close") - F.lag("close", 1).over(window)) /
        F.lag("close", 1).over(window))

# Step 2: Get daily average sentiment
daily_sentiment = news_df.groupBy("date").agg(
    F.avg("sentiment_score").alias("sentiment")
)

# Step 3: Join by date
combined = spy_daily.join(daily_sentiment, "date", "inner")

# Step 4: Create lagged columns (1-5 days)
for lag in range(1, max_lag + 1):
    combined = combined \
        .withColumn(f"return_lag{lag}", F.lag("return", lag).over(window)) \
        .withColumn(f"sentiment_lag{lag}", F.lag("sentiment", lag).over(window))

# Step 5: Calculate lagged correlations
pdf = combined.toPandas().dropna()

granger_results = []
for lag in range(1, max_lag + 1):
    # Does past sentiment predict current return?
    corr_s2r = pdf['return'].corr(pdf[f'sentiment_lag{lag}'])

    # Does past return predict current sentiment?
    corr_r2s = pdf['sentiment'].corr(pdf[f'return_lag{lag}'])

    granger_results.append({
        'direction': 'sentiment_to_return',
        'lag': lag,
        'correlation': corr_s2r
    })
```

### 3.4 출력 파일
| 파일 | 내용 |
|------|------|
| `spark_output/granger_correlations.csv` | direction, lag, correlation |

---

## 4. 📈 Word-Price Causality

### 4.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **뉴스 헤드라인** | `data/batch/news/financial_news_headlines.csv` |
| **주가 데이터** | `data/batch/stocks/stock_prices.csv` |

### 4.2 처리 스크립트
**파일:** `advanced_analytics/advanced_spark_cross_analysis.py` (Lines 599-701)

### 4.3 처리 코드 스니펫
```python
# Step 1: Count words per date
word_date_counts = word_df.groupBy("word", "date").agg(
    F.count("*").alias("count"),
    F.avg("sentiment_score").alias("avg_sentiment")
)

# Step 2: Get next day's price direction
price_direction = stock_df \
    .withColumn("next_day_return", F.lead("daily_return", 1).over(window)) \
    .withColumn("next_price_direction",
        F.when(F.col("next_day_return") > 0, "up")
         .otherwise("down"))

# Step 3: Join words with next-day price
word_before_price = word_date_counts.join(price_direction, "date")

# Step 4: Calculate predictive bias
word_direction_stats = word_before_price.groupBy("word", "next_price_direction").agg(
    F.sum("count").alias("total_count")
)

# Pivot to get up_count and down_count per word
word_direction_pivot = word_direction_stats.groupBy("word").pivot("next_price_direction").sum("total_count")

# Calculate bias: (up_count - down_count) / total
word_bias = word_direction_pivot.withColumn(
    "predictive_bias",
    (F.col("up") - F.col("down")) / (F.col("up") + F.col("down"))
)
```

### 4.4 출력 파일
| 파일 | 내용 |
|------|------|
| `spark_output/word_before_price_lag1.csv` | word, up_count, down_count, predictive_bias |

---

## 5. 🏷️ Topic Modeling (LDA)

### 5.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/news/financial_news_headlines.csv` |
| **수집 컬럼** | `headline` |

### 5.2 처리 스크립트
**파일:** `advanced_analytics/deep_semantic_analysis.py`

### 5.3 처리 코드 스니펫
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Create document-term matrix
vectorizer = CountVectorizer(
    max_df=0.95,      # Ignore terms in >95% of docs
    min_df=2,         # Ignore terms in <2 docs
    stop_words='english',
    max_features=1000
)
doc_term_matrix = vectorizer.fit_transform(headlines)

# Step 2: Fit LDA model
n_topics = 5
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,
    learning_method='online',
    random_state=42
)
lda.fit(doc_term_matrix)

# Step 3: Extract top words per topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
```

### 5.4 출력 파일
| 파일 | 내용 |
|------|------|
| `deep_semantic_output/lda_topics.csv` | topic_id, word, weight, topic_name |
| `deep_semantic_output/lda_topic_distribution.csv` | dominant_topic, count |

---

## 6. 🎯 TF-IDF Analysis

### 6.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/news/financial_news_headlines.csv` |
| **수집 컬럼** | `headline`, `sentiment` |

### 6.2 처리 스크립트
**파일:** `advanced_analytics/deep_semantic_analysis.py`

### 6.3 처리 코드 스니펫
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Group headlines by sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    sentiment_docs = news_df[news_df['sentiment'] == sentiment]['headline']

    # Step 2: Calculate TF-IDF
    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(sentiment_docs)

    # Step 3: Get feature names and scores
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    # Step 4: Create word-score pairs
    word_scores = list(zip(feature_names, scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
```

### 6.4 출력 파일
| 파일 | 내용 |
|------|------|
| `deep_semantic_output/tfidf_by_sentiment.csv` | word, sentiment, tfidf_score |

---

## 7. 🧠 Word2Vec Embeddings

### 7.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/news/financial_news_headlines.csv` |
| **수집 컬럼** | `headline` |

### 7.2 처리 스크립트
**파일:** `advanced_analytics/deep_semantic_analysis.py`

### 7.3 처리 코드 스니펫
```python
from gensim.models import Word2Vec

# Step 1: Tokenize all headlines
sentences = [headline.lower().split() for headline in headlines]

# Step 2: Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=100,    # 100-dimensional vectors
    window=5,           # Context window size
    min_count=2,        # Ignore words with freq < 2
    workers=4,
    epochs=10
)

# Step 3: Find similar words
key_terms = ['stock', 'market', 'earnings', 'rate', 'growth']
for term in key_terms:
    if term in model.wv:
        similar = model.wv.most_similar(term, topn=10)
        # similar = [('equity', 0.89), ('share', 0.85), ...]
```

### 7.4 출력 파일
| 파일 | 내용 |
|------|------|
| `deep_semantic_output/word2vec_similarities.csv` | term, similar_word, similarity |

---

## 8. 📊 Multi-Asset Correlation

### 8.1 데이터 출처 (Raw Data Source)
| 자산 | 데이터 소스 | 파일 |
|------|-------------|------|
| **USA GDP** | World Bank API | `data/batch/worldbank/world_bank_indicators.csv` |
| **Korea GDP** | World Bank API | `data/batch/worldbank/world_bank_indicators.csv` |
| **S&P 500** | Yahoo Finance | `data/batch/stocks/stock_prices.csv` (SPY) |
| **Bitcoin** | Synthetic | Generated |
| **Ethereum** | Synthetic | Generated |
| **Korea Real Estate** | Synthetic | Generated |

### 8.2 처리 스크립트
**파일:** `economics_social_cultural/multi_asset_correlation_analysis.py`

### 8.3 처리 코드 스니펫
```python
import pandas as pd
from scipy import stats

# Step 1: Merge all data by year
merged = usa_gdp.merge(korea_gdp, on='year')
merged = merged.merge(sp500_annual, on='year')
merged = merged.merge(crypto_annual, on='year')
merged = merged.merge(real_estate, on='year')

# Step 2: Calculate correlation matrix
numeric_cols = ['usa_gdp', 'korea_gdp', 'sp500_price', 'btc_price', 'korea_real_estate']
corr_matrix = merged[numeric_cols].corr()

# Step 3: Pairwise linear regression
from sklearn.linear_model import LinearRegression

for x_var in numeric_cols:
    for y_var in numeric_cols:
        if x_var != y_var:
            X = merged[[x_var]].dropna()
            y = merged[y_var].dropna()

            model = LinearRegression()
            model.fit(X, y)

            r_squared = model.score(X, y)
            correlation, p_value = stats.pearsonr(X[x_var], y)
```

### 8.4 출력 파일
| 파일 | 내용 |
|------|------|
| `output/merged_multi_asset_data.csv` | year, usa_gdp, korea_gdp, sp500_price, btc_price, ... |
| `output/correlation_matrix.csv` | Correlation matrix |
| `output/linear_regression_results.csv` | x_var, y_var, r_squared, correlation, p_value |
| `output/decade_analysis.csv` | decade, *_growth_%, *_avg |

---

# Ⅱ. Batch Dashboard (Port 8502)

## 1. Stock Analytics

### 1.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/stocks/stock_prices.csv` |
| **데이터 소스** | Yahoo Finance (yfinance library) |
| **API** | 무료, API 키 불필요 |
| **수집 종목** | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, SPY, QQQ 등 18종목 |
| **수집 기간** | 1962~2025년 (종목별 상이) |
| **레코드 수** | ~179,289개 |

### 1.2 처리 스크립트
**파일:** `batch_spark_analytics.py` (Lines 74-227)

### 1.3 처리 코드 스니펫
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Window definitions
window_20 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)
window_50 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-49, 0)
window_14 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-13, 0)

# Technical Indicators
df = self.df \
    # Daily Return
    .withColumn("daily_return",
        (F.col("close") - F.lag("close", 1).over(window)) /
        F.lag("close", 1).over(window)) \

    # 20-day Simple Moving Average
    .withColumn("sma_20", F.avg("close").over(window_20)) \

    # 50-day Simple Moving Average
    .withColumn("sma_50", F.avg("close").over(window_50)) \

    # 20-day Volatility (Standard Deviation)
    .withColumn("volatility_20d", F.stddev("close").over(window_20)) \

    # RSI (Relative Strength Index)
    .withColumn("price_change", F.col("close") - F.lag("close", 1).over(window)) \
    .withColumn("gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0)) \
    .withColumn("loss", F.when(F.col("price_change") < 0, F.abs(F.col("price_change"))).otherwise(0)) \
    .withColumn("avg_gain", F.avg("gain").over(window_14)) \
    .withColumn("avg_loss", F.avg("loss").over(window_14)) \
    .withColumn("rsi_14", 100 - (100 / (1 + F.col("avg_gain") / F.col("avg_loss")))) \

    # Bollinger Bands
    .withColumn("bb_upper", F.col("sma_20") + 2 * F.col("volatility_20d")) \
    .withColumn("bb_lower", F.col("sma_20") - 2 * F.col("volatility_20d"))

# Correlation Matrix
pivot_df = df.groupBy("date").pivot("symbol").agg(F.first("daily_return"))
pdf = pivot_df.toPandas()
corr_matrix = pdf.drop('date', axis=1).corr()
```

### 1.4 출력 파일
| 파일 | 내용 |
|------|------|
| `spark_output/stock_summary_stats.csv` | symbol, avg_price, volatility, total_return |
| `spark_output/stock_correlations.csv` | Stock correlation matrix |

---

## 2. News Sentiment Analytics

### 2.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/news/financial_news_headlines.csv` |
| **데이터 소스** | NewsAPI (https://newsapi.org) |
| **API 키** | 필요 (무료 티어: 100 req/day) |
| **수집 컬럼** | `headline`, `sentiment`, `sentiment_score`, `date`, `source`, `category` |

### 2.2 처리 스크립트
**파일:** `batch_spark_analytics.py` (Lines 337-454)

### 2.3 처리 코드 스니펫
```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Daily Sentiment Aggregation
daily_sentiment = self.df.groupBy("date").agg(
    F.avg("sentiment_score").alias("avg_sentiment"),
    F.stddev("sentiment_score").alias("sentiment_std"),
    F.count("*").alias("headline_count"),

    # Count by sentiment category
    F.sum(F.when(F.col("sentiment") == "positive", 1).otherwise(0)).alias("positive_count"),
    F.sum(F.when(F.col("sentiment") == "negative", 1).otherwise(0)).alias("negative_count"),
    F.sum(F.when(F.col("sentiment") == "neutral", 1).otherwise(0)).alias("neutral_count")
) \
.withColumn("positive_ratio", F.col("positive_count") / F.col("headline_count")) \
.withColumn("negative_ratio", F.col("negative_count") / F.col("headline_count"))

# Word Frequency Analysis (NLP)
tokenizer = Tokenizer(inputCol="headline", outputCol="words")
words_df = tokenizer.transform(self.df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_df = remover.transform(words_df)

# MapReduce pattern: Explode -> GroupBy -> Count
word_counts = filtered_df \
    .select(F.explode("filtered_words").alias("word")) \
    .filter(F.length("word") > 2) \
    .groupBy("word") \
    .count() \
    .orderBy(F.col("count").desc())
```

### 2.4 출력 파일
| 파일 | 내용 |
|------|------|
| `spark_output/daily_sentiment_analysis.csv` | date, avg_sentiment, positive_ratio, negative_ratio |
| `spark_output/word_frequencies.csv` | word, count |
| `spark_output/word_frequencies_positive.csv` | Positive headlines word freq |
| `spark_output/word_frequencies_negative.csv` | Negative headlines word freq |

---

## 3. Economic Indicators (FRED)

### 3.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/fred/economic_indicators.csv` |
| **데이터 소스** | FRED API (https://fred.stlouisfed.org) |
| **API 엔드포인트** | `https://api.stlouisfed.org/fred/series/observations` |
| **API 키** | 필요 (무료) |
| **수집 지표** | GDP, UNRATE, CPIAUCSL, FEDFUNDS, DGS10, SP500, M2SL 등 14개 |
| **레코드 수** | ~40,165개 |

### 3.2 처리 스크립트
**파일:** `batch_spark_analytics.py` (Lines 230-275)

### 3.3 처리 코드 스니펫
```python
# Year-over-Year Changes
window = Window.partitionBy("series_id").orderBy("year")

df = self.df \
    .withColumn("prev_value", F.lag("value", 1).over(window)) \
    .withColumn("yoy_change",
        (F.col("value") - F.col("prev_value")) / F.col("prev_value") * 100)

# Rolling 5-Year Average
window_5 = Window.partitionBy("series_id").orderBy("year").rowsBetween(-4, 0)
df = df.withColumn("rolling_avg_5y", F.avg("value").over(window_5))
```

### 3.4 출력 파일
| 파일 | 내용 |
|------|------|
| `spark_output/economic_trends.csv` | series_id, year, value, yoy_change, rolling_avg |

---

## 4. World Bank GDP Analysis

### 4.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **원본 파일** | `data/batch/worldbank/world_bank_indicators.csv` |
| **데이터 소스** | World Bank API (https://api.worldbank.org/v2) |
| **API 키** | 불필요 (무료 공개 API) |
| **수집 기간** | 1970~2024년 |
| **수집 국가** | 20개 주요 경제대국 |
| **레코드 수** | ~85,175개 |

### 4.2 처리 스크립트
**파일:** `batch_spark_analytics.py` (Lines 278-334)

### 4.3 처리 코드 스니펫
```python
# Filter to GDP indicator
gdp_df = self.df.filter(F.col("indicator_code") == "NY.GDP.MKTP.CD")

# Year-over-Year growth
window = Window.partitionBy("country_code").orderBy("year")
gdp_df = gdp_df \
    .withColumn("prev_gdp", F.lag("value", 1).over(window)) \
    .withColumn("gdp_growth",
        (F.col("value") - F.col("prev_gdp")) / F.col("prev_gdp") * 100)

# Country Rankings (latest year)
latest_year = gdp_df.agg(F.max("year")).collect()[0][0]
rankings = gdp_df \
    .filter(F.col("year") == latest_year) \
    .orderBy(F.col("value").desc()) \
    .withColumn("rank", F.row_number().over(Window.orderBy(F.col("value").desc())))
```

### 4.4 출력 파일
| 파일 | 내용 |
|------|------|
| `spark_output/gdp_trends.csv` | country, year, gdp, gdp_growth |
| `spark_output/gdp_rankings.csv` | country, gdp, rank |

---

# Ⅲ. Live Dashboard (Port 8501)

## 1. Real-time Orderbook Data

### 1.1 데이터 출처 (Raw Data Source)
| 거래소 | WebSocket URL |
|--------|---------------|
| **Binance** | `wss://fstream.binance.com/stream` |
| **OKX** | `wss://ws.okx.com:8443/ws/v5/public` |
| **Bybit** | `wss://stream.bybit.com/v5/public/spot` |
| **Upbit** | `wss://api.upbit.com/websocket/v1` |
| **Bithumb** | `wss://pubwss.bithumb.com/pub/ws` |
| **Coinone** | `wss://stream.coinone.co.kr` |
| **Korbit** | `wss://ws-api.korbit.co.kr/v2/public` |

### 1.2 처리 스크립트
**파일:** `websocket3.py` (Lines 751-827)

### 1.3 처리 코드 스니펫
```python
# Batcher: Collect events before writing to DB
async def batcher(src_q, dst_q, batch_size=100, max_wait=0.5):
    buf = []
    while True:
        ev = await asyncio.wait_for(src_q.get(), timeout=max_wait)
        buf.append(ev)
        if len(buf) >= batch_size or (now - last_flush) >= max_wait:
            await dst_q.put(buf)  # Send batch to DB consumer

# QuestDB Writer
async def questdb_consumer(batch_q):
    for ev in batch:
        base, quote, normalized = split_symbol(ex, venue_sym)
        region = get_region(ex, quote)  # "KR" or "GLOBAL"

        mid_price = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000

        sender.row("orderbook",
            symbols={"exchange": ex, "symbol": normalized, "region": region},
            columns={"best_bid": best_bid, "best_ask": best_ask, "spread_bps": spread_bps}
        )
```

### 1.4 저장 위치
| 저장소 | 내용 |
|--------|------|
| **QuestDB** | 실시간 오더북 데이터 (시계열) |
| **JSONL 파일** | `orderbook_stream.jsonl` (백업) |

---

## 2. Reddit Sentiment

### 2.1 데이터 출처 (Raw Data Source)
| 항목 | 내용 |
|------|------|
| **API** | Reddit Public JSON API |
| **엔드포인트** | `https://www.reddit.com/r/{subreddit}/new.json` |
| **API 키** | 불필요 |
| **모니터링 서브레딧** | Economics, stockMarket, investing, CryptoCurrency, Bitcoin |

### 2.2 처리 스크립트
**파일:** `reddit_sentiment.py` (Lines 68-141)

### 2.3 처리 코드 스니펫
```python
from textblob import TextBlob

class SentimentAnalyzer:
    CRYPTO_SYMBOLS = {'bitcoin': 'BTC', 'btc': 'BTC', 'ethereum': 'ETH', ...}

    @staticmethod
    def analyze(text: str) -> Dict:
        # 1. TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity      # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # 2. Extract crypto entities
        entities = {}
        dollar_mentions = re.findall(r'\$([A-Z]{2,5})\b', text)
        for coin_name, symbol in CRYPTO_SYMBOLS.items():
            count = len(re.findall(r'\b' + coin_name + r'\b', text_lower))
            if count > 0:
                entities[symbol] = count

        # 3. Detect topics
        topics = []
        if any(word in text_lower for word in ['moon', 'pump', 'bullish']):
            topics.append('bullish')

        return {"polarity": polarity, "entities": entities, "topics": topics}
```

### 2.4 저장 위치
| 저장소 | 내용 |
|--------|------|
| **LMDB** | `reddit_sentiment_lmdb.db` |
| **Key 패턴** | `post:{id}`, `sentiment:{id}` |

---

# Ⅳ. API 키 정보 요약

| API | 발급 URL | 무료 한도 | 프로젝트 사용처 |
|-----|----------|-----------|-----------------|
| **FRED** | https://fred.stlouisfed.org/docs/api/api_key.html | 무제한 | 경제 지표 |
| **NewsAPI** | https://newsapi.org/register | 100 req/day | 뉴스 헤드라인 |
| **World Bank** | 불필요 | 무제한 | GDP, 경제 지표 |
| **Yahoo Finance** | 불필요 (yfinance) | 무제한 | 주가 데이터 |
| **OpenSky** | https://opensky-network.org/apidoc | 400 req/day | 항공 데이터 |

---

# Ⅴ. 실행 방법

```bash
# 1. 배치 데이터 다운로드 (REAL API)
python batch_data_downloader_real_api.py

# 2. PySpark 분석 실행
python batch_spark_analytics.py
python advanced_analytics/advanced_spark_cross_analysis.py
python advanced_analytics/deep_semantic_analysis.py

# 3. 대시보드 실행
streamlit run live_dashboard.py --server.port 8501
streamlit run batch_dashboard.py --server.port 8502
streamlit run advanced_analytics/advanced_dashboard.py --server.port 8503
```
