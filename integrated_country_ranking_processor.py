"""
INTEGRATED COUNTRY RANKING PROCESSOR
=====================================
Combines ALL data sources into a unified country ranking system.

Data Sources Integrated:
1. ECONOMIC: GDP growth, stock market returns, exchange rates
2. SOCIAL: YouTube engagement, news coverage, word frequency
3. CULTURAL: Company mentions, brand recognition, topic trends

Output: Single CSV with Country as primary key and all metrics as columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
COUNTRY_DATA_DIR = BASE_DIR / "organized_data" / "raw_data" / "country_data"
SPARK_OUTPUT = BASE_DIR / "advanced_analytics" / "spark_output"
DEEP_OUTPUT = BASE_DIR / "advanced_analytics" / "deep_semantic_output"
BATCH_SPARK = BASE_DIR / "data" / "batch" / "spark_output"
OUTPUT_DIR = BASE_DIR / "organized_data" / "processed_data" / "integrated_ranking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Country Configuration
COUNTRIES = ['USA', 'KOREA', 'GERMANY', 'FRANCE', 'UK', 'JAPAN', 'CHINA', 'RUSSIA', 'INDIA', 'TAIWAN', 'CANADA']

# Primary index for each country (main stock market index)
PRIMARY_INDEX = {
    'USA': '^GSPC',      # S&P 500
    'KOREA': '^KS11',    # KOSPI (not KOSDAQ ^KQ11)
    'GERMANY': '^GDAXI', # DAX
    'FRANCE': '^FCHI',   # CAC 40
    'UK': '^FTSE',       # FTSE 100
    'JAPAN': '^N225',    # Nikkei 225
    'CHINA': '000001.SS', # Shanghai Composite
    'RUSSIA': 'IMOEX.ME', # MOEX
    'INDIA': '^BSESN',   # BSE Sensex
    'TAIWAN': '^TWII',   # Taiwan Weighted
    'CANADA': '^GSPTSE'  # TSX Composite
}

COUNTRY_KEYWORDS = {
    'USA': ['usa', 'us', 'america', 'american', 'united states', 'wall street', 'nasdaq', 'nyse', 'fed', 'dollar'],
    'KOREA': ['korea', 'korean', 'seoul', 'kospi', 'samsung', 'hyundai', 'lg', 'sk', 'won'],
    'GERMANY': ['germany', 'german', 'berlin', 'dax', 'volkswagen', 'bmw', 'siemens', 'euro'],
    'FRANCE': ['france', 'french', 'paris', 'cac', 'lvmh', 'total', 'bnp'],
    'UK': ['uk', 'britain', 'british', 'london', 'ftse', 'pound', 'sterling', 'boe'],
    'JAPAN': ['japan', 'japanese', 'tokyo', 'nikkei', 'yen', 'toyota', 'sony', 'honda'],
    'CHINA': ['china', 'chinese', 'beijing', 'shanghai', 'yuan', 'renminbi', 'alibaba', 'tencent', 'pboc'],
    'RUSSIA': ['russia', 'russian', 'moscow', 'ruble', 'gazprom', 'rosneft'],
    'INDIA': ['india', 'indian', 'mumbai', 'sensex', 'nifty', 'rupee', 'rbi', 'reliance', 'tata'],
    'TAIWAN': ['taiwan', 'taiwanese', 'taipei', 'tsmc', 'foxconn', 'taiex'],
    'CANADA': ['canada', 'canadian', 'toronto', 'tsx', 'loonie', 'boc']
}

COUNTRY_COMPANIES = {
    'USA': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'nvidia', 'tesla'],
    'KOREA': ['005930.KS', 'samsung', 'hyundai', 'lg', 'sk hynix', 'naver', 'kakao'],
    'GERMANY': ['SAP.DE', 'SIE.DE', 'sap', 'siemens', 'volkswagen', 'bmw', 'daimler', 'bayer', 'basf'],
    'FRANCE': ['MC.PA', 'lvmh', 'loreal', 'total', 'sanofi', 'bnp paribas', 'airbus'],
    'UK': ['SHEL.L', 'shell', 'hsbc', 'bp', 'astrazeneca', 'glaxo', 'unilever', 'barclays'],
    'JAPAN': ['7203.T', 'toyota', 'sony', 'honda', 'nintendo', 'softbank', 'mitsubishi'],
    'CHINA': ['alibaba', 'tencent', 'baidu', 'jd', 'xiaomi', 'huawei', 'bytedance', 'pinduoduo'],
    'RUSSIA': ['gazprom', 'rosneft', 'sberbank', 'lukoil', 'norilsk'],
    'INDIA': ['RELIANCE.NS', 'reliance', 'tata', 'infosys', 'hdfc', 'icici', 'wipro'],
    'TAIWAN': ['2330.TW', 'tsmc', 'foxconn', 'mediatek', 'asus', 'acer'],
    'CANADA': ['RY.TO', 'royal bank', 'td bank', 'shopify', 'enbridge', 'suncor']
}

def load_data():
    """Load all data sources"""
    print("Loading all data sources...")
    data = {}

    # Country indices
    try:
        data['indices'] = pd.read_csv(COUNTRY_DATA_DIR / "country_indices.csv")
        data['indices']['Date'] = pd.to_datetime(data['indices']['Date'])
        print(f"  Indices: {len(data['indices']):,} records")
    except Exception as e:
        print(f"  Indices: Failed - {e}")
        data['indices'] = pd.DataFrame()

    # Country stocks
    try:
        data['stocks'] = pd.read_csv(COUNTRY_DATA_DIR / "country_stocks.csv")
        data['stocks']['Date'] = pd.to_datetime(data['stocks']['Date'])
        print(f"  Stocks: {len(data['stocks']):,} records")
    except Exception as e:
        print(f"  Stocks: Failed - {e}")
        data['stocks'] = pd.DataFrame()

    # Exchange rates
    try:
        data['fx'] = pd.read_csv(COUNTRY_DATA_DIR / "exchange_rates.csv")
        data['fx']['Date'] = pd.to_datetime(data['fx']['Date'])
        print(f"  Exchange Rates: {len(data['fx']):,} records")
    except Exception as e:
        print(f"  Exchange Rates: Failed - {e}")
        data['fx'] = pd.DataFrame()

    # News
    try:
        data['news'] = pd.read_csv(COUNTRY_DATA_DIR / "country_news.csv")
        print(f"  News: {len(data['news']):,} articles")
    except Exception as e:
        print(f"  News: Failed - {e}")
        data['news'] = pd.DataFrame()

    # YouTube
    try:
        data['youtube_videos'] = pd.read_csv(COUNTRY_DATA_DIR / "youtube_videos.csv")
        data['youtube_comments'] = pd.read_csv(COUNTRY_DATA_DIR / "youtube_comments.csv")
        print(f"  YouTube: {len(data['youtube_videos']):,} videos, {len(data['youtube_comments']):,} comments")
    except Exception as e:
        print(f"  YouTube: Failed - {e}")
        data['youtube_videos'] = pd.DataFrame()
        data['youtube_comments'] = pd.DataFrame()

    # FRED
    try:
        data['fred'] = pd.read_csv(COUNTRY_DATA_DIR / "fred_extended.csv")
        print(f"  FRED: {len(data['fred']):,} records")
    except Exception as e:
        print(f"  FRED: Failed - {e}")
        data['fred'] = pd.DataFrame()

    # Word PageRank
    try:
        data['pagerank'] = pd.read_csv(SPARK_OUTPUT / "word_pagerank.csv")
        print(f"  PageRank: {len(data['pagerank']):,} words")
    except Exception as e:
        print(f"  PageRank: Failed - {e}")
        data['pagerank'] = pd.DataFrame()

    # Word Graph
    try:
        data['word_graph'] = pd.read_csv(SPARK_OUTPUT / "word_graph_edges.csv")
        print(f"  Word Graph: {len(data['word_graph']):,} edges")
    except Exception as e:
        print(f"  Word Graph: Failed - {e}")
        data['word_graph'] = pd.DataFrame()

    # Granger
    try:
        data['granger'] = pd.read_csv(SPARK_OUTPUT / "granger_correlations.csv")
        print(f"  Granger: {len(data['granger']):,} correlations")
    except Exception as e:
        print(f"  Granger: Failed - {e}")
        data['granger'] = pd.DataFrame()

    # TF-IDF
    try:
        data['tfidf'] = pd.read_csv(DEEP_OUTPUT / "tfidf_by_sentiment.csv")
        print(f"  TF-IDF: {len(data['tfidf']):,} terms")
    except Exception as e:
        print(f"  TF-IDF: Failed - {e}")
        data['tfidf'] = pd.DataFrame()

    # N-grams
    try:
        data['ngrams'] = pd.read_csv(DEEP_OUTPUT / "ngram_analysis.csv")
        print(f"  N-grams: {len(data['ngrams']):,} phrases")
    except Exception as e:
        print(f"  N-grams: Failed - {e}")
        data['ngrams'] = pd.DataFrame()

    return data

def calculate_economic_metrics(data):
    """Calculate ECONOMIC metrics by country"""
    print("\n=== Calculating ECONOMIC Metrics ===")

    results = {country: {} for country in COUNTRIES}

    # 1. Stock Market Returns (1 year)
    if not data['indices'].empty:
        cutoff = datetime.now() - timedelta(days=365)
        recent = data['indices'][data['indices']['Date'] >= cutoff]

        for country in COUNTRIES:
            country_data = recent[recent['country'] == country]
            if not country_data.empty:
                # Use the PRIMARY_INDEX for each country (e.g., KOSPI not KOSDAQ for Korea)
                primary_symbol = PRIMARY_INDEX.get(country)

                # Filter for primary index, fallback to first available if not found
                if primary_symbol and primary_symbol in country_data['symbol'].values:
                    sym_data = country_data[country_data['symbol'] == primary_symbol].sort_values('Date')
                else:
                    # Fallback: use the symbol with most data points
                    symbol_counts = country_data.groupby('symbol').size()
                    primary_symbol = symbol_counts.idxmax()
                    sym_data = country_data[country_data['symbol'] == primary_symbol].sort_values('Date')

                if len(sym_data) > 10:
                    start_price = sym_data['Close'].iloc[0]
                    end_price = sym_data['Close'].iloc[-1]
                    returns = ((end_price - start_price) / start_price) * 100
                    volatility = sym_data['Close'].pct_change().std() * np.sqrt(252) * 100
                    results[country]['stock_return_1y'] = round(returns, 2)
                    results[country]['stock_volatility'] = round(volatility, 2)
                    print(f"  {country} ({primary_symbol}): Return={returns:.1f}%, Vol={volatility:.1f}%")

    # 2. Company Performance (average return of top companies)
    if not data['stocks'].empty:
        cutoff = datetime.now() - timedelta(days=365)
        recent = data['stocks'][data['stocks']['Date'] >= cutoff]

        for country in COUNTRIES:
            country_stocks = recent[recent['country'] == country]
            if not country_stocks.empty:
                returns_list = []
                for symbol in country_stocks['symbol'].unique():
                    sym_data = country_stocks[country_stocks['symbol'] == symbol].sort_values('Date')
                    if len(sym_data) > 10:
                        start = sym_data['Close'].iloc[0]
                        end = sym_data['Close'].iloc[-1]
                        ret = ((end - start) / start) * 100
                        returns_list.append(ret)

                if returns_list:
                    avg_return = np.mean(returns_list)
                    best_return = max(returns_list)
                    results[country]['company_avg_return'] = round(avg_return, 2)
                    results[country]['company_best_return'] = round(best_return, 2)
                    results[country]['num_companies'] = len(returns_list)

    # 3. Currency Strength (vs USD)
    if not data['fx'].empty:
        fx_mapping = {
            'USA': 'DX-Y.NYB',  # Dollar index (inverse logic)
            'KOREA': 'USDKRW=X',
            'GERMANY': 'EURUSD=X',
            'FRANCE': 'EURUSD=X',
            'UK': 'GBPUSD=X',
            'JAPAN': 'USDJPY=X',
            'CHINA': 'USDCNY=X',
            'RUSSIA': 'USDRUB=X',
            'INDIA': 'USDINR=X',
            'TAIWAN': 'USDTWD=X',
            'CANADA': 'USDCAD=X'
        }

        cutoff = datetime.now() - timedelta(days=365)
        recent = data['fx'][data['fx']['Date'] >= cutoff]

        for country, symbol in fx_mapping.items():
            fx_data = recent[recent['symbol'] == symbol].sort_values('Date')
            if len(fx_data) > 10:
                start = fx_data['Close'].iloc[0]
                end = fx_data['Close'].iloc[-1]
                change = ((end - start) / start) * 100

                # For USD pairs, negative change means currency strengthened
                if symbol in ['EURUSD=X', 'GBPUSD=X']:
                    currency_strength = change  # Positive = strengthened
                elif symbol == 'DX-Y.NYB':
                    currency_strength = change  # Dollar index
                else:
                    currency_strength = -change  # Negative = strengthened vs USD

                results[country]['currency_change_1y'] = round(currency_strength, 2)

    return results

def calculate_social_metrics(data):
    """Calculate SOCIAL metrics by country"""
    print("\n=== Calculating SOCIAL Metrics ===")

    results = {country: {} for country in COUNTRIES}

    # 1. News Coverage
    if not data['news'].empty:
        for country in COUNTRIES:
            country_news = data['news'][data['news']['country'] == country]
            results[country]['news_articles'] = len(country_news)
            results[country]['news_sources'] = country_news['source'].nunique() if 'source' in country_news.columns else 0
        print(f"  News coverage calculated")

    # 2. YouTube Engagement
    if not data['youtube_videos'].empty:
        for country in COUNTRIES:
            country_yt = data['youtube_videos'][data['youtube_videos']['country'] == country]
            if not country_yt.empty:
                results[country]['youtube_videos'] = len(country_yt)
                results[country]['youtube_views'] = int(country_yt['view_count'].sum())
                results[country]['youtube_likes'] = int(country_yt['like_count'].sum())
                results[country]['youtube_comments'] = int(country_yt['comment_count'].sum())
                results[country]['youtube_engagement'] = int(country_yt['like_count'].sum() + country_yt['comment_count'].sum())
        print(f"  YouTube engagement calculated")

    # 3. YouTube Comments Sentiment (simple positive word count)
    if not data['youtube_comments'].empty:
        positive_words = ['good', 'great', 'excellent', 'amazing', 'best', 'love', 'wonderful', 'bullish', 'growth', 'profit']
        negative_words = ['bad', 'terrible', 'worst', 'hate', 'crash', 'crisis', 'bearish', 'loss', 'fail', 'recession']

        for country in COUNTRIES:
            country_comments = data['youtube_comments'][data['youtube_comments']['country'] == country]
            if not country_comments.empty and 'text' in country_comments.columns:
                all_text = ' '.join(country_comments['text'].astype(str).str.lower())
                pos_count = sum(all_text.count(w) for w in positive_words)
                neg_count = sum(all_text.count(w) for w in negative_words)
                total = pos_count + neg_count
                if total > 0:
                    sentiment_ratio = (pos_count - neg_count) / total
                    results[country]['social_sentiment'] = round(sentiment_ratio, 3)
        print(f"  Social sentiment calculated")

    return results

def calculate_cultural_metrics(data):
    """Calculate CULTURAL/WORD metrics by country"""
    print("\n=== Calculating CULTURAL Metrics ===")

    results = {country: {} for country in COUNTRIES}

    # 1. Country/Company Word Frequency from PageRank
    if not data['pagerank'].empty:
        word_col = data['pagerank'].columns[0]  # First column is usually the word
        score_col = data['pagerank'].columns[-1] if len(data['pagerank'].columns) > 1 else word_col

        for country in COUNTRIES:
            keywords = COUNTRY_KEYWORDS.get(country, []) + COUNTRY_COMPANIES.get(country, [])
            total_score = 0
            matches = 0

            for kw in keywords:
                kw_lower = kw.lower()
                matching = data['pagerank'][data['pagerank'][word_col].astype(str).str.lower().str.contains(kw_lower, na=False)]
                if not matching.empty:
                    if score_col != word_col:
                        total_score += matching[score_col].sum()
                    matches += len(matching)

            results[country]['word_pagerank_score'] = round(total_score, 4)
            results[country]['word_matches'] = matches
        print(f"  PageRank word scores calculated")

    # 2. Word Graph Centrality (how connected country keywords are)
    if not data['word_graph'].empty:
        cols = data['word_graph'].columns.tolist()
        src_col = cols[0]
        tgt_col = cols[1] if len(cols) > 1 else cols[0]

        for country in COUNTRIES:
            keywords = COUNTRY_KEYWORDS.get(country, [])
            edge_count = 0

            for kw in keywords:
                kw_lower = kw.lower()
                # Count edges involving this keyword
                src_matches = data['word_graph'][data['word_graph'][src_col].astype(str).str.lower().str.contains(kw_lower, na=False)]
                tgt_matches = data['word_graph'][data['word_graph'][tgt_col].astype(str).str.lower().str.contains(kw_lower, na=False)]
                edge_count += len(src_matches) + len(tgt_matches)

            results[country]['word_graph_edges'] = edge_count
        print(f"  Word graph centrality calculated")

    # 3. N-gram Phrase Frequency
    if not data['ngrams'].empty:
        cols = data['ngrams'].columns.tolist()
        phrase_col = cols[0]
        count_col = cols[1] if len(cols) > 1 else None

        for country in COUNTRIES:
            keywords = COUNTRY_KEYWORDS.get(country, [])
            phrase_score = 0

            for kw in keywords:
                kw_lower = kw.lower()
                matching = data['ngrams'][data['ngrams'][phrase_col].astype(str).str.lower().str.contains(kw_lower, na=False)]
                if not matching.empty and count_col:
                    phrase_score += matching[count_col].sum()
                else:
                    phrase_score += len(matching)

            results[country]['ngram_score'] = int(phrase_score)
        print(f"  N-gram scores calculated")

    # 4. Granger Causality (words that predict prices)
    if not data['granger'].empty:
        cols = data['granger'].columns.tolist()
        word_col = cols[0]

        for country in COUNTRIES:
            keywords = COUNTRY_KEYWORDS.get(country, []) + COUNTRY_COMPANIES.get(country, [])
            causal_count = 0

            for kw in keywords:
                kw_lower = kw.lower()
                matching = data['granger'][data['granger'][word_col].astype(str).str.lower().str.contains(kw_lower, na=False)]
                causal_count += len(matching)

            results[country]['granger_causal_words'] = causal_count
        print(f"  Granger causality calculated")

    return results

def calculate_trend_indicators(data):
    """Calculate TREND indicators (up/down momentum)"""
    print("\n=== Calculating TREND Indicators ===")

    results = {country: {} for country in COUNTRIES}

    if not data['indices'].empty:
        for country in COUNTRIES:
            country_data = data['indices'][data['indices']['country'] == country]
            if not country_data.empty:
                # Use PRIMARY_INDEX for each country
                primary_symbol = PRIMARY_INDEX.get(country)

                if primary_symbol and primary_symbol in country_data['symbol'].values:
                    sym_data = country_data[country_data['symbol'] == primary_symbol].sort_values('Date')
                else:
                    # Fallback: use the symbol with most data points
                    symbol = country_data.groupby('symbol').size().idxmax()
                    sym_data = country_data[country_data['symbol'] == symbol].sort_values('Date')

                if len(sym_data) > 30:
                    # Short-term trend (30 days)
                    recent_30 = sym_data.tail(30)
                    trend_30 = ((recent_30['Close'].iloc[-1] - recent_30['Close'].iloc[0]) / recent_30['Close'].iloc[0]) * 100

                    # Medium-term trend (90 days)
                    if len(sym_data) > 90:
                        recent_90 = sym_data.tail(90)
                        trend_90 = ((recent_90['Close'].iloc[-1] - recent_90['Close'].iloc[0]) / recent_90['Close'].iloc[0]) * 100
                    else:
                        trend_90 = trend_30

                    # Momentum (is it accelerating?)
                    momentum = trend_30 - (trend_90 / 3)  # Compare recent to average

                    results[country]['trend_30d'] = round(trend_30, 2)
                    results[country]['trend_90d'] = round(trend_90, 2)
                    results[country]['momentum'] = round(momentum, 2)

                    # Direction indicator
                    if trend_30 > 2 and momentum > 0:
                        results[country]['trend_direction'] = 'STRONG_UP'
                    elif trend_30 > 0:
                        results[country]['trend_direction'] = 'UP'
                    elif trend_30 < -2 and momentum < 0:
                        results[country]['trend_direction'] = 'STRONG_DOWN'
                    elif trend_30 < 0:
                        results[country]['trend_direction'] = 'DOWN'
                    else:
                        results[country]['trend_direction'] = 'NEUTRAL'

    print(f"  Trend indicators calculated for {len([c for c in results if results[c]])} countries")
    return results

def normalize_scores_zscore(df, columns, higher_is_better=True):
    """
    Normalize columns using Z-score based method.

    Instead of Min-Max (0-100 with hard limits), this method:
    1. Calculates Z-score (standard deviations from mean)
    2. Converts to percentile-like score centered at 50
    3. Allows scores to exceed 100 or go below 0 for outliers

    This preserves the actual distribution and shows true differences.
    A score of 50 = average, 70 = 2 std above average, etc.
    """
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()

            if std_val > 0:
                # Z-score: how many standard deviations from mean
                z_scores = (df[col] - mean_val) / std_val

                # Convert to score centered at 50, with ~10 points per standard deviation
                # This means: 50 = average, 60 = 1 std above, 70 = 2 std above
                if higher_is_better:
                    df[f'{col}_norm'] = 50 + (z_scores * 10)
                else:
                    df[f'{col}_norm'] = 50 - (z_scores * 10)  # Invert for "lower is better"
            else:
                df[f'{col}_norm'] = 50  # All values are the same
    return df


def normalize_scores_minmax(df, columns, higher_is_better=True):
    """
    Legacy Min-Max normalization to 0-100 scale.
    Kept for reference but not used by default.
    """
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                if higher_is_better:
                    df[f'{col}_norm'] = ((df[col] - min_val) / (max_val - min_val)) * 100
                else:
                    df[f'{col}_norm'] = ((max_val - df[col]) / (max_val - min_val)) * 100
            else:
                df[f'{col}_norm'] = 50
    return df

def calculate_composite_scores(df):
    """Calculate composite scores for each category"""
    print("\n=== Calculating Composite Scores ===")

    # Economic Score (40% weight)
    econ_cols = ['stock_return_1y_norm', 'company_avg_return_norm', 'currency_change_1y_norm']
    econ_cols = [c for c in econ_cols if c in df.columns]
    if econ_cols:
        df['economic_score'] = df[econ_cols].mean(axis=1)
    else:
        df['economic_score'] = 50

    # Social Score (30% weight)
    social_cols = ['youtube_engagement_norm', 'news_articles_norm', 'social_sentiment_norm']
    social_cols = [c for c in social_cols if c in df.columns]
    if social_cols:
        df['social_score'] = df[social_cols].mean(axis=1)
    else:
        df['social_score'] = 50

    # Cultural Score (20% weight)
    cultural_cols = ['word_pagerank_score_norm', 'word_graph_edges_norm', 'ngram_score_norm']
    cultural_cols = [c for c in cultural_cols if c in df.columns]
    if cultural_cols:
        df['cultural_score'] = df[cultural_cols].mean(axis=1)
    else:
        df['cultural_score'] = 50

    # Trend Score (10% weight)
    if 'momentum_norm' in df.columns:
        df['trend_score'] = df['momentum_norm']
    else:
        df['trend_score'] = 50

    # Final Composite Score
    df['composite_score'] = (
        df['economic_score'] * 0.40 +
        df['social_score'] * 0.30 +
        df['cultural_score'] * 0.20 +
        df['trend_score'] * 0.10
    )

    # Overall Rank
    df['overall_rank'] = df['composite_score'].rank(ascending=False).astype(int)

    return df

def main():
    print("="*70)
    print("INTEGRATED COUNTRY RANKING PROCESSOR")
    print("="*70)
    print(f"Processing {len(COUNTRIES)} countries...")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)

    # Load all data
    data = load_data()

    # Calculate all metrics
    economic = calculate_economic_metrics(data)
    social = calculate_social_metrics(data)
    cultural = calculate_cultural_metrics(data)
    trends = calculate_trend_indicators(data)

    # Combine into single DataFrame
    print("\n=== Combining All Metrics ===")

    rows = []
    for country in COUNTRIES:
        row = {'country': country}
        row.update(economic.get(country, {}))
        row.update(social.get(country, {}))
        row.update(cultural.get(country, {}))
        row.update(trends.get(country, {}))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Normalize scores
    normalize_cols_higher = [
        'stock_return_1y', 'company_avg_return', 'company_best_return',
        'news_articles', 'news_sources',
        'youtube_videos', 'youtube_views', 'youtube_likes', 'youtube_engagement',
        'social_sentiment',
        'word_pagerank_score', 'word_matches', 'word_graph_edges', 'ngram_score', 'granger_causal_words',
        'trend_30d', 'trend_90d', 'momentum'
    ]

    normalize_cols_lower = ['stock_volatility']  # Lower is better

    # Use Z-score normalization (no artificial 0-100 limits)
    # Score interpretation: 50 = average, 60 = 1 std above, 70 = 2 std above
    df = normalize_scores_zscore(df, normalize_cols_higher, higher_is_better=True)
    df = normalize_scores_zscore(df, normalize_cols_lower, higher_is_better=False)

    # Calculate composite scores
    df = calculate_composite_scores(df)

    # Sort by overall rank
    df = df.sort_values('overall_rank')

    # Save detailed results
    detailed_file = OUTPUT_DIR / "country_ranking_detailed.csv"
    df.to_csv(detailed_file, index=False)
    print(f"\nSaved detailed ranking to: {detailed_file}")

    # Create summary table
    summary_cols = [
        'overall_rank', 'country',
        'economic_score', 'social_score', 'cultural_score', 'trend_score', 'composite_score',
        'stock_return_1y', 'youtube_engagement', 'news_articles',
        'trend_direction', 'momentum'
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    summary = df[summary_cols].copy()

    summary_file = OUTPUT_DIR / "country_ranking_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary ranking to: {summary_file}")

    # Print results
    print("\n" + "="*70)
    print("FINAL COUNTRY RANKINGS")
    print("="*70)
    print(f"{'Rank':<6}{'Country':<12}{'Economic':<10}{'Social':<10}{'Cultural':<10}{'Trend':<10}{'TOTAL':<10}")
    print("-"*70)

    for _, row in df.iterrows():
        print(f"{int(row['overall_rank']):<6}{row['country']:<12}"
              f"{row['economic_score']:.1f}{'':>4}{row['social_score']:.1f}{'':>4}"
              f"{row['cultural_score']:.1f}{'':>4}{row['trend_score']:.1f}{'':>4}"
              f"{row['composite_score']:.1f}")

    print("="*70)
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Output files saved to: {OUTPUT_DIR}")

    return df

if __name__ == "__main__":
    result = main()
