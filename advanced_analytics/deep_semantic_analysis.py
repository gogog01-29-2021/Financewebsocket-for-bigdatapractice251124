#!/usr/bin/env python3
"""
DEEP SEMANTIC ANALYSIS
Advanced NLP with PySpark:
- TF-IDF Vectorization
- N-gram (Phrase) Analysis
- Topic Modeling (LDA)
- Word2Vec Embeddings
- Entity-Specific Price Correlation
- Korea vs USA Economic Comparison
"""

import os
import sys

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    # Fix PySpark Python path issue on Windows
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, split, explode, count, avg, sum as spark_sum,
    when, lit, concat_ws, collect_list, size, array_contains, udf, window,
    date_format, to_date, datediff, lag, lead, corr, stddev, variance,
    first, last, min as spark_min, max as spark_max, row_number, dense_rank,
    monotonically_increasing_id, expr, struct, create_map, arrays_zip
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StringType, FloatType, ArrayType, StructType, StructField, IntegerType, DoubleType
)
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer, IDF, NGram, Word2Vec,
    HashingTF, RegexTokenizer
)
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict
from gensim.models import Word2Vec as GensimWord2Vec

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "batch"
NEWS_DIR = DATA_DIR / "news"
STOCK_DIR = DATA_DIR / "stocks"
WORLDBANK_DIR = DATA_DIR / "worldbank"
FRED_DIR = DATA_DIR / "fred"
OUTPUT_DIR = Path(__file__).parent / "deep_semantic_output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("DEEP SEMANTIC ANALYSIS")
print("TF-IDF, N-grams, LDA, Word2Vec, Korea-USA Comparison")
print("=" * 60)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")


def create_spark_session():
    """Create Spark session with ML libraries"""
    spark = SparkSession.builder \
        .appName("DeepSemanticAnalysis") \
        .master("local[1]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8") \
        .config("spark.pyspark.python", sys.executable) \
        .config("spark.pyspark.driver.python", sys.executable) \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.reuse", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


class NGramAnalysis:
    """N-gram (phrase) analysis for meaningful multi-word expressions"""

    def __init__(self, spark, news_df):
        self.spark = spark
        self.news_df = news_df

    def extract_ngrams(self, n_values=[2, 3]):
        """Extract bigrams and trigrams from headlines"""
        print("\n" + "=" * 60)
        print("N-GRAM (PHRASE) ANALYSIS")
        print("=" * 60)

        # Tokenize
        tokenizer = RegexTokenizer(
            inputCol="headline_clean",
            outputCol="words",
            pattern="\\W"
        )

        # Remove stopwords
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )

        # Prepare data
        df = self.news_df.withColumn(
            "headline_clean",
            lower(regexp_replace(col("headline"), "[^a-zA-Z\\s]", ""))
        )

        df = tokenizer.transform(df)
        df = remover.transform(df)

        all_ngrams = []

        for n in n_values:
            print(f"\n[N-gram] Extracting {n}-grams...")

            ngram = NGram(n=n, inputCol="filtered_words", outputCol=f"ngrams_{n}")
            df_ngram = ngram.transform(df)

            # Explode and count
            ngram_counts = df_ngram.select(
                explode(col(f"ngrams_{n}")).alias("ngram"),
                col("sentiment")
            ).groupBy("ngram", "sentiment").count()

            # Get top ngrams
            top_ngrams = ngram_counts.groupBy("ngram") \
                .agg(spark_sum("count").alias("total_count")) \
                .orderBy(col("total_count").desc()) \
                .limit(100)

            # Add sentiment breakdown
            ngram_sentiment = ngram_counts.groupBy("ngram").pivot("sentiment").sum("count")

            result = top_ngrams.join(ngram_sentiment, "ngram", "left")

            result_pd = result.toPandas()
            result_pd['n'] = n
            all_ngrams.append(result_pd)

            print(f"  Top {n}-grams:")
            for _, row in result_pd.head(10).iterrows():
                print(f"    '{row['ngram']}': {row['total_count']}")

        # Combine and save
        combined = pd.concat(all_ngrams, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "ngram_analysis.csv", index=False)
        print(f"\n  Saved {len(combined)} n-grams")

        return combined


class TFIDFAnalysis:
    """TF-IDF to find unique/important words per category"""

    def __init__(self, spark, news_df):
        self.spark = spark
        self.news_df = news_df

    def compute_tfidf_by_sentiment(self):
        """Compute TF-IDF scores grouped by sentiment"""
        print("\n" + "=" * 60)
        print("TF-IDF ANALYSIS")
        print("=" * 60)

        # Prepare text
        df = self.news_df.withColumn(
            "headline_clean",
            lower(regexp_replace(col("headline"), "[^a-zA-Z\\s]", ""))
        )

        # Tokenize
        tokenizer = RegexTokenizer(
            inputCol="headline_clean",
            outputCol="words",
            pattern="\\W"
        )

        remover = StopWordsRemover(inputCol="words", outputCol="filtered")

        df = tokenizer.transform(df)
        df = remover.transform(df)

        # Count vectorizer
        cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", minDF=5)
        cv_model = cv.fit(df)
        df = cv_model.transform(df)

        # IDF
        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        idf_model = idf.fit(df)
        df = idf_model.transform(df)

        vocabulary = cv_model.vocabulary

        # Get TF-IDF by sentiment
        results = []

        for sentiment in ['positive', 'negative', 'neutral']:
            print(f"\n[TF-IDF] Analyzing {sentiment} headlines...")

            sentiment_df = df.filter(col("sentiment") == sentiment)

            if sentiment_df.count() == 0:
                continue

            # Aggregate TF-IDF scores
            tfidf_vectors = sentiment_df.select("tfidf_features").collect()

            # Sum TF-IDF scores across documents
            word_scores = defaultdict(float)
            for row in tfidf_vectors:
                vector = row.tfidf_features
                for idx, value in zip(vector.indices, vector.values):
                    word_scores[vocabulary[idx]] += value

            # Normalize by document count
            doc_count = len(tfidf_vectors)
            for word in word_scores:
                word_scores[word] /= doc_count

            # Top words
            top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:50]

            print(f"  Top distinctive words for {sentiment}:")
            for word, score in top_words[:10]:
                print(f"    {word}: {score:.4f}")
                results.append({
                    'sentiment': sentiment,
                    'word': word,
                    'tfidf_score': score
                })

        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "tfidf_by_sentiment.csv", index=False)
        print(f"\n  Saved TF-IDF results")

        return results_df


class TopicModelingLDA:
    """Latent Dirichlet Allocation for topic discovery"""

    def __init__(self, spark, news_df, num_topics=8):
        self.spark = spark
        self.news_df = news_df
        self.num_topics = num_topics

    def discover_topics(self):
        """Run LDA topic modeling"""
        print("\n" + "=" * 60)
        print(f"TOPIC MODELING (LDA) - {self.num_topics} Topics")
        print("=" * 60)

        # Prepare text
        df = self.news_df.withColumn(
            "headline_clean",
            lower(regexp_replace(col("headline"), "[^a-zA-Z\\s]", ""))
        )

        # Pipeline
        tokenizer = RegexTokenizer(inputCol="headline_clean", outputCol="words", pattern="\\W")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        cv = CountVectorizer(inputCol="filtered", outputCol="features", minDF=10, maxDF=0.8)

        df = tokenizer.transform(df)
        df = remover.transform(df)

        cv_model = cv.fit(df)
        df = cv_model.transform(df)

        vocabulary = cv_model.vocabulary

        # LDA
        print("\n[LDA] Training model...")
        lda = LDA(k=self.num_topics, maxIter=20, optimizer="online")
        lda_model = lda.fit(df)

        # Get topics
        topics = lda_model.describeTopics(maxTermsPerTopic=15)

        print("\n[LDA] Discovered Topics:")
        topic_results = []

        topics_pd = topics.toPandas()
        for idx, row in topics_pd.iterrows():
            topic_words = [vocabulary[i] for i in row['termIndices']]
            topic_weights = row['termWeights']

            # Auto-name topic based on top words
            topic_name = f"Topic_{idx}: {', '.join(topic_words[:3])}"

            print(f"\n  {topic_name}")
            for word, weight in zip(topic_words[:10], topic_weights[:10]):
                print(f"    {word}: {weight:.4f}")
                topic_results.append({
                    'topic_id': idx,
                    'topic_name': topic_name,
                    'word': word,
                    'weight': weight
                })

        # Get document-topic distribution (PySpark LDA)
        transformed = lda_model.transform(df)

        # Assign dominant topic using vector_to_array (native Spark function avoids Python worker crash)
        from pyspark.ml.functions import vector_to_array

        # Convert Vector to Array type using native Spark function
        transformed_with_array = transformed.withColumn(
            "topic_array",
            vector_to_array("topicDistribution")
        )

        # Find dominant topic using Spark SQL array_position (finds max value index)
        # array_position returns 1-based index, so subtract 1
        transformed_with_topic = transformed_with_array.withColumn(
            "dominant_topic",
            expr("array_position(topic_array, array_max(topic_array)) - 1").cast("int")
        )

        doc_topics = transformed_with_topic.select("sentiment", "dominant_topic")

        # Topic distribution
        topic_dist = doc_topics.groupBy("dominant_topic").count().orderBy("dominant_topic")
        topic_dist_pd = topic_dist.toPandas()

        print("\n[LDA] Topic Distribution:")
        for _, row in topic_dist_pd.iterrows():
            print(f"  Topic {row['dominant_topic']}: {row['count']} documents")

        # Save results
        pd.DataFrame(topic_results).to_csv(OUTPUT_DIR / "lda_topics.csv", index=False)
        topic_dist_pd.to_csv(OUTPUT_DIR / "lda_topic_distribution.csv", index=False)

        # Topic-sentiment relationship
        topic_sentiment = doc_topics.groupBy("dominant_topic", "sentiment").count()
        topic_sentiment_pd = topic_sentiment.toPandas()
        topic_sentiment_pd.to_csv(OUTPUT_DIR / "lda_topic_sentiment.csv", index=False)

        print(f"\n  Saved LDA results")

        return topic_results


class Word2VecAnalysis:
    """Word embeddings to find semantic similarities using Gensim Word2Vec"""

    def __init__(self, spark, news_df):
        self.spark = spark
        self.news_df = news_df
        self.model = None

    def train_word2vec(self, vector_size=100, min_count=5):
        """Train Word2Vec model on headlines using Gensim (avoids PySpark Python worker issues on Windows)"""
        print("\n" + "=" * 60)
        print("WORD2VEC EMBEDDINGS (Gensim)")
        print("=" * 60)

        # Prepare text using PySpark
        df = self.news_df.withColumn(
            "headline_clean",
            lower(regexp_replace(col("headline"), "[^a-zA-Z\\s]", ""))
        )

        tokenizer = RegexTokenizer(inputCol="headline_clean", outputCol="words", pattern="\\W")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")

        df = tokenizer.transform(df)
        df = remover.transform(df)

        # Extract tokenized sentences using collect() to avoid Python worker serialization issues
        print("\n[Word2Vec] Extracting tokenized sentences...")
        sentences_rows = df.select("filtered").collect()
        sentences = [row['filtered'] for row in sentences_rows if row['filtered']]

        # Filter out empty sentences
        sentences = [s for s in sentences if len(s) > 0]
        print(f"  Total sentences: {len(sentences)}")

        # Train Gensim Word2Vec
        print(f"\n[Word2Vec] Training Gensim Word2Vec with vector_size={vector_size}...")
        self.model = GensimWord2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=min_count,
            workers=1,  # Single worker for Windows compatibility
            epochs=10
        )

        vocab_size = len(self.model.wv)
        print(f"  Vocabulary size: {vocab_size}")

        # Find similar words for key financial terms
        key_terms = ['stock', 'market', 'price', 'growth', 'earnings',
                     'fed', 'inflation', 'bank', 'investor', 'profit']

        results = []

        print("\n[Word2Vec] Similar words for key terms:")
        for term in key_terms:
            try:
                if term not in self.model.wv:
                    print(f"  '{term}' not in vocabulary")
                    continue

                # Get top 5 similar words using Gensim's most_similar
                similar_words = self.model.wv.most_similar(term, topn=5)

                print(f"\n  '{term}' similar to:")
                for word, sim in similar_words:
                    print(f"    {word}: {sim:.4f}")
                    results.append({
                        'term': term,
                        'similar_word': word,
                        'similarity': sim
                    })
            except Exception as e:
                print(f"  '{term}' error: {e}")

        # Save results
        pd.DataFrame(results).to_csv(OUTPUT_DIR / "word2vec_similarities.csv", index=False)

        # Save word vectors for visualization
        vectors_data = []
        for word in self.model.wv.index_to_key:
            vec = self.model.wv[word].tolist()
            vectors_data.append({'word': word, 'vector': vec})

        vectors_pd = pd.DataFrame(vectors_data)
        vectors_pd.to_csv(OUTPUT_DIR / "word2vec_vectors.csv", index=False)

        # Save model for later use
        self.model.save(str(OUTPUT_DIR / "word2vec_model.bin"))

        print(f"\n  Saved Word2Vec results and model")

        return results


class EntityPriceCorrelation:
    """Analyze how specific stock mentions affect stock prices"""

    def __init__(self, spark, news_df, stock_df):
        self.spark = spark
        self.news_df = news_df
        self.stock_df = stock_df

    def analyze_entity_price(self):
        """Correlate stock mentions with price movements"""
        print("\n" + "=" * 60)
        print("ENTITY-PRICE CORRELATION")
        print("=" * 60)

        # Stock symbols to track
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'GS', 'BAC']

        # Normalize date formats - extract date portion only (YYYY-MM-DD)
        news_normalized = self.news_df.withColumn(
            "date_normalized",
            to_date(col("date"))
        )
        stock_normalized = self.stock_df.withColumn(
            "date_normalized",
            to_date(col("date"))
        )

        # Create stock mention flags
        news_with_stocks = news_normalized

        for symbol in symbols:
            # Check if symbol or company name appears in headline
            news_with_stocks = news_with_stocks.withColumn(
                f"mentions_{symbol}",
                when(
                    lower(col("headline")).contains(symbol.lower()) |
                    lower(col("headline")).contains(self._get_company_name(symbol).lower()),
                    1
                ).otherwise(0)
            )

        # Aggregate by date
        mention_cols = [f"mentions_{s}" for s in symbols]

        news_by_date = news_with_stocks.groupBy("date_normalized").agg(
            *[spark_sum(c).alias(c) for c in mention_cols],
            avg("sentiment_score").alias("avg_sentiment"),
            count("*").alias("news_count")
        )

        # Join with stock data
        results = []

        for symbol in symbols:
            print(f"\n[Entity] Analyzing {symbol}...")

            stock_data = stock_normalized.filter(col("symbol") == symbol) \
                .select("date_normalized", "close", "volume") \
                .withColumnRenamed("close", f"{symbol}_close") \
                .withColumnRenamed("volume", f"{symbol}_volume")

            # Calculate returns
            w = Window.orderBy("date_normalized")
            stock_data = stock_data.withColumn(
                f"{symbol}_return",
                (col(f"{symbol}_close") - lag(f"{symbol}_close", 1).over(w)) /
                lag(f"{symbol}_close", 1).over(w)
            )

            # Join with news using normalized dates
            joined = news_by_date.join(stock_data, "date_normalized", "inner")

            # Calculate correlation between mentions and returns
            mention_col = f"mentions_{symbol}"
            return_col = f"{symbol}_return"

            if joined.count() > 10:
                # Same-day correlation
                same_day_corr = joined.select(corr(mention_col, return_col)).collect()[0][0]

                # Next-day correlation (mentions today -> return tomorrow)
                joined_lag = joined.withColumn(
                    "next_return",
                    lead(return_col, 1).over(w)
                )
                next_day_corr = joined_lag.select(corr(mention_col, "next_return")).collect()[0][0]

                # Sentiment-return correlation
                sent_return_corr = joined.select(corr("avg_sentiment", return_col)).collect()[0][0]

                results.append({
                    'symbol': symbol,
                    'company': self._get_company_name(symbol),
                    'total_mentions': joined.agg(spark_sum(mention_col)).collect()[0][0],
                    'same_day_corr': same_day_corr if same_day_corr else 0,
                    'next_day_corr': next_day_corr if next_day_corr else 0,
                    'sentiment_return_corr': sent_return_corr if sent_return_corr else 0
                })

                print(f"    Mentions: {results[-1]['total_mentions']}")
                print(f"    Same-day correlation: {results[-1]['same_day_corr']:.4f}")
                print(f"    Next-day correlation: {results[-1]['next_day_corr']:.4f}")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "entity_price_correlation.csv", index=False)

        print(f"\n  Saved entity-price correlation results")

        return results_df

    def _get_company_name(self, symbol):
        """Map symbol to company name"""
        mapping = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
            'AMZN': 'Amazon', 'TSLA': 'Tesla', 'META': 'Facebook',
            'NVDA': 'Nvidia', 'JPM': 'JPMorgan', 'GS': 'Goldman', 'BAC': 'Bank of America'
        }
        return mapping.get(symbol, symbol)


class KoreaUSAComparison:
    """Compare Korea and USA economic data"""

    def __init__(self, spark, gdp_df, stock_df):
        self.spark = spark
        self.gdp_df = gdp_df
        self.stock_df = stock_df

    def compare_economies(self):
        """Comprehensive Korea vs USA comparison"""
        print("\n" + "=" * 60)
        print("KOREA vs USA ECONOMIC COMPARISON")
        print("=" * 60)

        results = {
            'gdp_comparison': [],
            'gdp_growth_comparison': [],
            'correlation_analysis': [],
            'key_metrics': {}
        }

        # Filter for Korea and USA - must also filter by indicator for GDP
        korea_codes = ['KOR', 'KR', 'Korea']
        usa_codes = ['USA', 'US', 'United States']
        gdp_indicator = 'NY.GDP.MKTP.CD'  # GDP (current US$)

        # Filter by GDP indicator first if available
        gdp_data = self.gdp_df
        if 'indicator_code' in gdp_data.columns:
            gdp_data = gdp_data.filter(col("indicator_code") == gdp_indicator)
            print(f"\n[Korea-USA] Filtered for GDP indicator: {gdp_indicator}")

        # Try to find Korea and USA in GDP data
        if 'country_code' in gdp_data.columns:
            korea_gdp = gdp_data.filter(col("country_code").isin(korea_codes))
            usa_gdp = gdp_data.filter(col("country_code").isin(usa_codes))
        elif 'country' in gdp_data.columns:
            korea_gdp = gdp_data.filter(
                lower(col("country")).contains("korea") |
                col("country").isin(korea_codes)
            )
            usa_gdp = gdp_data.filter(
                lower(col("country")).contains("united states") |
                lower(col("country")).contains("usa") |
                col("country").isin(usa_codes)
            )
        else:
            print("  Warning: Cannot identify country column in GDP data")
            korea_gdp = gdp_data.limit(0)
            usa_gdp = gdp_data.limit(0)

        print(f"\n[Korea-USA] GDP Records - Korea: {korea_gdp.count()}, USA: {usa_gdp.count()}")

        if korea_gdp.count() > 0 and usa_gdp.count() > 0:
            # GDP over time
            korea_pd = korea_gdp.toPandas()
            usa_pd = usa_gdp.toPandas()

            # Find common years
            if 'year' in korea_pd.columns:
                korea_pd = korea_pd.sort_values('year')
                usa_pd = usa_pd.sort_values('year')

                # GDP comparison by year
                comparison_data = []

                korea_years = set(korea_pd['year'].unique())
                usa_years = set(usa_pd['year'].unique())
                common_years = sorted(korea_years & usa_years)

                print(f"\n[Korea-USA] Common years: {len(common_years)} ({min(common_years)}-{max(common_years)})")

                for year in common_years:
                    kr_val = korea_pd[korea_pd['year'] == year]['value'].values
                    us_val = usa_pd[usa_pd['year'] == year]['value'].values

                    if len(kr_val) > 0 and len(us_val) > 0:
                        comparison_data.append({
                            'year': year,
                            'korea_gdp': kr_val[0],
                            'usa_gdp': us_val[0],
                            'ratio_korea_usa': kr_val[0] / us_val[0] if us_val[0] != 0 else 0
                        })

                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)

                    # Calculate growth rates
                    comp_df['korea_growth'] = comp_df['korea_gdp'].pct_change() * 100
                    comp_df['usa_growth'] = comp_df['usa_gdp'].pct_change() * 100
                    comp_df['growth_diff'] = comp_df['korea_growth'] - comp_df['usa_growth']

                    results['gdp_comparison'] = comp_df.to_dict('records')
                    comp_df.to_csv(OUTPUT_DIR / "korea_usa_gdp_comparison.csv", index=False)

                    # Key statistics
                    recent = comp_df.tail(10)

                    results['key_metrics'] = {
                        'korea_avg_growth_10yr': recent['korea_growth'].mean(),
                        'usa_avg_growth_10yr': recent['usa_growth'].mean(),
                        'korea_gdp_latest': comp_df['korea_gdp'].iloc[-1],
                        'usa_gdp_latest': comp_df['usa_gdp'].iloc[-1],
                        'korea_usa_ratio_latest': comp_df['ratio_korea_usa'].iloc[-1],
                        'growth_correlation': comp_df['korea_growth'].corr(comp_df['usa_growth'])
                    }

                    print("\n[Korea-USA] Key Metrics:")
                    print(f"  Korea Avg Growth (10yr): {results['key_metrics']['korea_avg_growth_10yr']:.2f}%")
                    print(f"  USA Avg Growth (10yr): {results['key_metrics']['usa_avg_growth_10yr']:.2f}%")
                    print(f"  Korea/USA GDP Ratio: {results['key_metrics']['korea_usa_ratio_latest']:.4f}")
                    print(f"  Growth Correlation: {results['key_metrics']['growth_correlation']:.4f}")

        # Stock market comparison (if we have Korean stocks)
        print("\n[Korea-USA] Comparing Market Performance...")

        # US market represented by SPY
        spy_data = self.stock_df.filter(col("symbol") == "SPY")

        if spy_data.count() > 0:
            # Calculate US market returns
            w = Window.orderBy("date")
            spy_returns = spy_data.withColumn(
                "daily_return",
                (col("close") - lag("close", 1).over(w)) / lag("close", 1).over(w)
            ).select("date", "daily_return", "close", "volume")

            spy_pd = spy_returns.toPandas()

            results['us_market'] = {
                'avg_daily_return': spy_pd['daily_return'].mean() * 100,
                'volatility': spy_pd['daily_return'].std() * 100,
                'total_return': ((spy_pd['close'].iloc[-1] / spy_pd['close'].iloc[0]) - 1) * 100,
                'sharpe_approx': spy_pd['daily_return'].mean() / spy_pd['daily_return'].std() * np.sqrt(252)
            }

            print(f"\n  US Market (SPY):")
            print(f"    Avg Daily Return: {results['us_market']['avg_daily_return']:.4f}%")
            print(f"    Volatility: {results['us_market']['volatility']:.4f}%")
            print(f"    Total Return: {results['us_market']['total_return']:.2f}%")
            print(f"    Sharpe Ratio (approx): {results['us_market']['sharpe_approx']:.2f}")

            spy_pd.to_csv(OUTPUT_DIR / "usa_market_performance.csv", index=False)

        # Save all results
        with open(OUTPUT_DIR / "korea_usa_comparison.json", 'w') as f:
            # Convert to JSON-serializable format
            json_results = {
                'key_metrics': results.get('key_metrics', {}),
                'us_market': results.get('us_market', {}),
                'years_analyzed': len(results.get('gdp_comparison', []))
            }
            json.dump(json_results, f, indent=2, default=str)

        print(f"\n  Saved Korea-USA comparison results")

        return results


def analyze_korea_markets(korea_data):
    """Analyze Korea market data and compare with US"""
    print("\n" + "=" * 60)
    print("KOREA MARKET ANALYSIS (KOSPI, KRW, Samsung)")
    print("=" * 60)

    if not korea_data:
        print("  No Korea market data available")
        return

    results = {}

    # KOSPI Analysis
    if 'kospi' in korea_data:
        kospi = korea_data['kospi']
        kospi['daily_return'] = kospi['close'].pct_change()

        results['kospi'] = {
            'avg_daily_return': kospi['daily_return'].mean() * 100,
            'volatility': kospi['daily_return'].std() * 100,
            'total_return': ((kospi['close'].iloc[-1] / kospi['close'].iloc[0]) - 1) * 100,
            'sharpe_approx': kospi['daily_return'].mean() / kospi['daily_return'].std() * np.sqrt(252),
            'latest_close': kospi['close'].iloc[-1],
            'records': len(kospi)
        }

        print(f"\n  KOSPI Index:")
        print(f"    Records: {results['kospi']['records']}")
        print(f"    Latest: {results['kospi']['latest_close']:.2f}")
        print(f"    Total Return: {results['kospi']['total_return']:.2f}%")
        print(f"    Volatility: {results['kospi']['volatility']:.4f}%")
        print(f"    Sharpe Ratio: {results['kospi']['sharpe_approx']:.2f}")

    # KRW Analysis
    if 'krw' in korea_data:
        krw = korea_data['krw']
        krw['daily_change'] = krw['close'].pct_change()

        results['krw'] = {
            'latest_rate': krw['close'].iloc[-1],
            'avg_rate': krw['close'].mean(),
            'min_rate': krw['close'].min(),
            'max_rate': krw['close'].max(),
            'volatility': krw['daily_change'].std() * 100,
            'records': len(krw)
        }

        print(f"\n  KRW/USD Exchange Rate:")
        print(f"    Latest: {results['krw']['latest_rate']:.2f} KRW/USD")
        print(f"    Range: {results['krw']['min_rate']:.2f} - {results['krw']['max_rate']:.2f}")
        print(f"    Volatility: {results['krw']['volatility']:.4f}%")

    # Samsung Analysis
    if 'samsung' in korea_data:
        samsung = korea_data['samsung']
        samsung['daily_return'] = samsung['close'].pct_change()

        results['samsung'] = {
            'avg_daily_return': samsung['daily_return'].mean() * 100,
            'volatility': samsung['daily_return'].std() * 100,
            'total_return': ((samsung['close'].iloc[-1] / samsung['close'].iloc[0]) - 1) * 100,
            'latest_close': samsung['close'].iloc[-1],
            'records': len(samsung)
        }

        print(f"\n  Samsung Electronics:")
        print(f"    Latest: {results['samsung']['latest_close']:.0f} KRW")
        print(f"    Total Return: {results['samsung']['total_return']:.2f}%")
        print(f"    Volatility: {results['samsung']['volatility']:.4f}%")

    # KOSPI-SPY Correlation (if both available)
    if 'kospi' in korea_data:
        try:
            import yfinance as yf
            spy = yf.download("SPY", period="5y", progress=False)
            if not spy.empty:
                if hasattr(spy.columns, 'droplevel'):
                    spy.columns = spy.columns.droplevel(1)
                spy = spy.reset_index()
                spy.columns = [c.lower() for c in spy.columns]
                spy['daily_return'] = spy['close'].pct_change()

                # Align dates and calculate correlation
                kospi_returns = korea_data['kospi'].set_index('date')['daily_return']
                spy_returns = spy.set_index('date')['daily_return']

                # Find common dates
                common_dates = kospi_returns.index.intersection(spy_returns.index)
                if len(common_dates) > 100:
                    correlation = kospi_returns.loc[common_dates].corr(spy_returns.loc[common_dates])
                    results['kospi_spy_correlation'] = correlation

                    print(f"\n  KOSPI-SPY Correlation: {correlation:.4f}")

                    # Compare metrics
                    spy_total_return = ((spy['close'].iloc[-1] / spy['close'].iloc[0]) - 1) * 100

                    print(f"\n  Market Comparison (5 years):")
                    print(f"    KOSPI Total Return: {results['kospi']['total_return']:.2f}%")
                    print(f"    SPY Total Return: {spy_total_return:.2f}%")
                    print(f"    KOSPI Volatility: {results['kospi']['volatility']:.4f}%")
                    print(f"    SPY Volatility: {spy['daily_return'].std() * 100:.4f}%")

                    results['comparison'] = {
                        'kospi_return': results['kospi']['total_return'],
                        'spy_return': spy_total_return,
                        'kospi_volatility': results['kospi']['volatility'],
                        'spy_volatility': spy['daily_return'].std() * 100
                    }

        except Exception as e:
            print(f"  Error calculating correlation: {e}")

    # Save results
    with open(OUTPUT_DIR / "korea_market_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Saved Korea market analysis")

    return results


def download_korea_market_data():
    """Download KOSPI index and KRW exchange rate data"""
    print("\n" + "=" * 60)
    print("DOWNLOADING KOREA MARKET DATA")
    print("=" * 60)

    try:
        import yfinance as yf

        korea_data = {}

        # Download KOSPI (^KS11)
        print("\n[Korea] Downloading KOSPI index...")
        kospi = yf.download("^KS11", period="5y", progress=False)
        if not kospi.empty:
            # Handle MultiIndex columns from yfinance
            if hasattr(kospi.columns, 'droplevel'):
                kospi.columns = kospi.columns.droplevel(1)
            kospi = kospi.reset_index()
            kospi['symbol'] = 'KOSPI'
            kospi.columns = [c.lower() for c in kospi.columns]
            korea_data['kospi'] = kospi
            print(f"    KOSPI records: {len(kospi)}")
            kospi.to_csv(OUTPUT_DIR / "kospi_index.csv", index=False)

        # Download KRW/USD exchange rate (USDKRW=X)
        print("[Korea] Downloading KRW exchange rate...")
        krw = yf.download("KRW=X", period="5y", progress=False)
        if not krw.empty:
            if hasattr(krw.columns, 'droplevel'):
                krw.columns = krw.columns.droplevel(1)
            krw = krw.reset_index()
            krw['symbol'] = 'USDKRW'
            krw.columns = [c.lower() for c in krw.columns]
            korea_data['krw'] = krw
            print(f"    KRW records: {len(krw)}")
            krw.to_csv(OUTPUT_DIR / "krw_exchange_rate.csv", index=False)

        # Download Samsung (005930.KS) - largest Korean company
        print("[Korea] Downloading Samsung Electronics...")
        samsung = yf.download("005930.KS", period="5y", progress=False)
        if not samsung.empty:
            if hasattr(samsung.columns, 'droplevel'):
                samsung.columns = samsung.columns.droplevel(1)
            samsung = samsung.reset_index()
            samsung['symbol'] = 'SAMSUNG'
            samsung.columns = [c.lower() for c in samsung.columns]
            korea_data['samsung'] = samsung
            print(f"    Samsung records: {len(samsung)}")
            samsung.to_csv(OUTPUT_DIR / "samsung_stock.csv", index=False)

        return korea_data

    except Exception as e:
        print(f"  Error downloading Korea data: {e}")
        return {}


def main():
    """Run all deep semantic analyses"""

    # Initialize Spark
    print("\n" + "=" * 60)
    print("INITIALIZING SPARK SESSION")
    print("=" * 60)

    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    # Download Korea market data first
    korea_data = download_korea_market_data()

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # News headlines - correct path
    news_path = NEWS_DIR / "financial_news_headlines.csv"
    if news_path.exists():
        news_df = spark.read.csv(str(news_path), header=True, inferSchema=True)
        # Rename column if needed
        if 'title' in news_df.columns and 'headline' not in news_df.columns:
            news_df = news_df.withColumnRenamed('title', 'headline')
        print(f"  News records: {news_df.count()}")
    else:
        print(f"  ERROR: News data not found at {news_path}!")
        return

    # Stock data - correct path
    stock_path = STOCK_DIR / "stock_prices.csv"
    if stock_path.exists():
        stock_df = spark.read.csv(str(stock_path), header=True, inferSchema=True)
        print(f"  Stock records: {stock_df.count()}")
    else:
        stock_df = None
        print("  Warning: Stock data not found")

    # GDP data - correct path
    gdp_path = WORLDBANK_DIR / "world_bank_indicators.csv"
    if gdp_path.exists():
        gdp_df = spark.read.csv(str(gdp_path), header=True, inferSchema=True)
        print(f"  GDP records: {gdp_df.count()}")
    else:
        gdp_df = None
        print("  Warning: GDP data not found")

    # 1. N-gram Analysis
    ngram = NGramAnalysis(spark, news_df)
    ngram.extract_ngrams([2, 3])

    # 2. TF-IDF Analysis
    tfidf = TFIDFAnalysis(spark, news_df)
    tfidf.compute_tfidf_by_sentiment()

    # 3. Topic Modeling (LDA)
    lda = TopicModelingLDA(spark, news_df, num_topics=8)
    lda.discover_topics()

    # 4. Word2Vec Embeddings
    w2v = Word2VecAnalysis(spark, news_df)
    w2v.train_word2vec(vector_size=100)

    # 5. Entity-Price Correlation
    if stock_df:
        entity = EntityPriceCorrelation(spark, news_df, stock_df)
        entity.analyze_entity_price()

    # 6. Korea-USA Comparison (with KOSPI data)
    if gdp_df and stock_df:
        korea_usa = KoreaUSAComparison(spark, gdp_df, stock_df)
        korea_usa.compare_economies()

    # 7. Korea Market Analysis (KOSPI, KRW, Samsung)
    analyze_korea_markets(korea_data)

    # Generate summary report
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)

    report = {
        'generated_at': datetime.now().isoformat(),
        'analyses_completed': [
            'N-gram (Bigram/Trigram) Analysis',
            'TF-IDF by Sentiment',
            'LDA Topic Modeling',
            'Word2Vec Embeddings',
            'Entity-Price Correlation',
            'Korea-USA Economic Comparison'
        ],
        'output_files': [f.name for f in OUTPUT_DIR.glob("*")]
    }

    with open(OUTPUT_DIR / "deep_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # List output files
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    spark.stop()


if __name__ == "__main__":
    main()
