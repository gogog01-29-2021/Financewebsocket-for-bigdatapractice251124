#!/usr/bin/env python3
"""
ENHANCED COUNTRY RANKING DASHBOARD v2.0
========================================
Comprehensive country-centered analytics with REAL DATA:
- Stock Market Indices (Yahoo Finance - 10 years)
- Top Company Stocks (Yahoo Finance - 5 years)
- Exchange Rates (Yahoo Finance - 10 years)
- Commodities & Crypto (Yahoo Finance)
- Country News (NewsAPI - real-time)
- Economic Data (FRED API, World Bank)
- NLP Analysis (PageRank, TF-IDF, LDA, Word2Vec)

Countries: USA, Korea, Germany, France, UK, Japan, China, Russia, India, Taiwan, Canada
"""

import streamlit as st

st.set_page_config(
    page_title="Country Analytics Dashboard v2.0",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
COUNTRY_DATA_DIR = BASE_DIR / "organized_data" / "raw_data" / "country_data"
SPARK_OUTPUT = BASE_DIR / "advanced_analytics" / "spark_output"
DEEP_OUTPUT = BASE_DIR / "advanced_analytics" / "deep_semantic_output"
BATCH_SPARK = BASE_DIR / "data" / "batch" / "spark_output"
DATA_DIR = BASE_DIR / "data" / "batch"

# =============================================================================
# COUNTRY CONFIGURATION
# =============================================================================
COUNTRIES = {
    'USA': {'name': 'United States', 'flag': '🇺🇸', 'currency': 'USD', 'index': '^GSPC', 'index_name': 'S&P 500'},
    'KOREA': {'name': 'South Korea', 'flag': '🇰🇷', 'currency': 'KRW', 'index': '^KS11', 'index_name': 'KOSPI'},
    'GERMANY': {'name': 'Germany', 'flag': '🇩🇪', 'currency': 'EUR', 'index': '^GDAXI', 'index_name': 'DAX'},
    'FRANCE': {'name': 'France', 'flag': '🇫🇷', 'currency': 'EUR', 'index': '^FCHI', 'index_name': 'CAC 40'},
    'UK': {'name': 'United Kingdom', 'flag': '🇬🇧', 'currency': 'GBP', 'index': '^FTSE', 'index_name': 'FTSE 100'},
    'JAPAN': {'name': 'Japan', 'flag': '🇯🇵', 'currency': 'JPY', 'index': '^N225', 'index_name': 'Nikkei 225'},
    'CHINA': {'name': 'China', 'flag': '🇨🇳', 'currency': 'CNY', 'index': '000001.SS', 'index_name': 'Shanghai'},
    'RUSSIA': {'name': 'Russia', 'flag': '🇷🇺', 'currency': 'RUB', 'index': 'IMOEX.ME', 'index_name': 'MOEX'},
    'INDIA': {'name': 'India', 'flag': '🇮🇳', 'currency': 'INR', 'index': '^BSESN', 'index_name': 'Sensex'},
    'TAIWAN': {'name': 'Taiwan', 'flag': '🇹🇼', 'currency': 'TWD', 'index': '^TWII', 'index_name': 'TAIEX'},
    'CANADA': {'name': 'Canada', 'flag': '🇨🇦', 'currency': 'CAD', 'index': '^GSPTSE', 'index_name': 'TSX'},
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def load_country_indices():
    """Load country stock indices data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "country_indices.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"Could not load indices: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_country_stocks():
    """Load country company stocks data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "country_stocks.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"Could not load stocks: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_exchange_rates():
    """Load exchange rates data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "exchange_rates.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"Could not load exchange rates: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_commodities():
    """Load commodities data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "commodities.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"Could not load commodities: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_crypto():
    """Load cryptocurrency data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "crypto.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.warning(f"Could not load crypto: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_country_news():
    """Load country news data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "country_news.csv")
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        return df
    except Exception as e:
        st.warning(f"Could not load news: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_fred_data():
    """Load FRED economic data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "fred_extended.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.warning(f"Could not load FRED data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_youtube_videos():
    """Load YouTube videos data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "youtube_videos.csv")
        df['published_at'] = pd.to_datetime(df['published_at'])
        return df
    except Exception as e:
        st.warning(f"Could not load YouTube videos: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_youtube_comments():
    """Load YouTube comments data"""
    try:
        df = pd.read_csv(COUNTRY_DATA_DIR / "youtube_comments.csv")
        df['published_at'] = pd.to_datetime(df['published_at'])
        return df
    except Exception as e:
        st.warning(f"Could not load YouTube comments: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_pagerank():
    """Load PageRank data"""
    try:
        return pd.read_csv(SPARK_OUTPUT / "word_pagerank.csv")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_word_graph():
    """Load word graph edges"""
    try:
        return pd.read_csv(SPARK_OUTPUT / "word_graph_edges.csv")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_granger():
    """Load Granger causality data"""
    try:
        return pd.read_csv(SPARK_OUTPUT / "granger_correlations.csv")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_tfidf():
    """Load TF-IDF data"""
    try:
        return pd.read_csv(DEEP_OUTPUT / "tfidf_by_sentiment.csv")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_lda():
    """Load LDA topics"""
    try:
        return pd.read_csv(DEEP_OUTPUT / "lda_topics.csv")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_entity_price():
    """Load entity-price correlations"""
    try:
        return pd.read_csv(DEEP_OUTPUT / "entity_price_correlation.csv")
    except:
        return pd.DataFrame()

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def calculate_market_performance(indices_df, period_days=365):
    """Calculate market performance ranking"""
    if indices_df.empty:
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=period_days)
    recent = indices_df[indices_df['Date'] >= cutoff]

    results = []
    for country in recent['country'].unique():
        country_data = recent[recent['country'] == country]
        main_idx = COUNTRIES.get(country, {}).get('index', '')

        idx_data = country_data[country_data['symbol'] == main_idx]
        if not idx_data.empty and len(idx_data) > 1:
            idx_data = idx_data.sort_values('Date')
            start_price = idx_data['Close'].iloc[0]
            end_price = idx_data['Close'].iloc[-1]
            returns = ((end_price - start_price) / start_price) * 100
            volatility = idx_data['Close'].pct_change().std() * np.sqrt(252) * 100

            results.append({
                'country': country,
                'country_name': COUNTRIES.get(country, {}).get('name', country),
                'flag': COUNTRIES.get(country, {}).get('flag', ''),
                'index_name': COUNTRIES.get(country, {}).get('index_name', ''),
                'return_pct': returns,
                'volatility': volatility,
                'latest_price': end_price,
                'data_points': len(idx_data)
            })

    if results:
        df = pd.DataFrame(results)
        df['rank'] = df['return_pct'].rank(ascending=False).astype(int)
        return df.sort_values('rank')
    return pd.DataFrame()

def calculate_news_ranking(news_df):
    """Calculate country news ranking"""
    if news_df.empty:
        return pd.DataFrame()

    results = []
    for country in news_df['country'].unique():
        country_news = news_df[news_df['country'] == country]

        results.append({
            'country': country,
            'country_name': COUNTRIES.get(country, {}).get('name', country),
            'flag': COUNTRIES.get(country, {}).get('flag', ''),
            'article_count': len(country_news),
            'sources': country_news['source'].nunique(),
            'latest_article': country_news['publishedAt'].max()
        })

    if results:
        df = pd.DataFrame(results)
        df['rank'] = df['article_count'].rank(ascending=False).astype(int)
        return df.sort_values('rank')
    return pd.DataFrame()

def calculate_company_ranking(stocks_df, period_days=365):
    """Calculate top companies by performance"""
    if stocks_df.empty:
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=period_days)
    recent = stocks_df[stocks_df['Date'] >= cutoff]

    results = []
    for symbol in recent['symbol'].unique():
        sym_data = recent[recent['symbol'] == symbol].sort_values('Date')
        if len(sym_data) > 10:
            start_price = sym_data['Close'].iloc[0]
            end_price = sym_data['Close'].iloc[-1]
            returns = ((end_price - start_price) / start_price) * 100
            avg_volume = sym_data['Volume'].mean()

            country = sym_data['country'].iloc[0] if 'country' in sym_data.columns else 'Unknown'

            results.append({
                'symbol': symbol,
                'country': country,
                'return_pct': returns,
                'latest_price': end_price,
                'avg_volume': avg_volume
            })

    if results:
        df = pd.DataFrame(results)
        df['rank'] = df['return_pct'].rank(ascending=False).astype(int)
        return df.sort_values('rank')
    return pd.DataFrame()

def calculate_currency_ranking(fx_df, period_days=365):
    """Calculate currency performance ranking"""
    if fx_df.empty:
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=period_days)
    recent = fx_df[fx_df['Date'] >= cutoff]

    results = []
    for symbol in recent['symbol'].unique():
        sym_data = recent[recent['symbol'] == symbol].sort_values('Date')
        if len(sym_data) > 10:
            start_price = sym_data['Close'].iloc[0]
            end_price = sym_data['Close'].iloc[-1]
            change = ((end_price - start_price) / start_price) * 100

            results.append({
                'symbol': symbol,
                'change_pct': change,
                'latest_rate': end_price,
                'volatility': sym_data['Close'].pct_change().std() * 100
            })

    if results:
        df = pd.DataFrame(results)
        return df.sort_values('change_pct', ascending=False)
    return pd.DataFrame()

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_market_comparison(indices_df, selected_countries):
    """Plot market indices comparison"""
    if indices_df.empty:
        return None

    fig = go.Figure()

    for country in selected_countries:
        config = COUNTRIES.get(country, {})
        main_idx = config.get('index', '')

        data = indices_df[(indices_df['country'] == country) & (indices_df['symbol'] == main_idx)]
        if not data.empty:
            data = data.sort_values('Date')
            # Normalize to 100
            normalized = (data['Close'] / data['Close'].iloc[0]) * 100

            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=normalized,
                mode='lines',
                name=f"{config.get('flag', '')} {config.get('index_name', country)}",
                hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title="Stock Market Indices Comparison (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        hovermode='x unified',
        height=500
    )

    return fig

def plot_country_heatmap(performance_df):
    """Plot country performance heatmap"""
    if performance_df.empty:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=[performance_df['return_pct'].values],
        x=performance_df['country'].values,
        y=['Return %'],
        colorscale='RdYlGn',
        text=[[f"{v:.1f}%" for v in performance_df['return_pct'].values]],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))

    fig.update_layout(
        title="Country Market Returns Heatmap",
        height=200
    )

    return fig

def plot_company_treemap(stocks_df, selected_country):
    """Plot company market cap treemap"""
    if stocks_df.empty:
        return None

    country_stocks = stocks_df[stocks_df['country'] == selected_country]
    if country_stocks.empty:
        return None

    # Get latest data for each company
    latest = country_stocks.groupby('symbol').last().reset_index()
    latest['market_value'] = latest['Close'] * latest['Volume']

    if latest.empty:
        return None

    fig = px.treemap(
        latest.nlargest(20, 'market_value'),
        path=['symbol'],
        values='market_value',
        color='Close',
        color_continuous_scale='Blues',
        title=f"Top Companies in {COUNTRIES.get(selected_country, {}).get('name', selected_country)}"
    )

    return fig

def plot_exchange_rates(fx_df):
    """Plot exchange rate trends"""
    if fx_df.empty:
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'USD/EUR', 'USD/JPY', 'USD/KRW', 'USD/CNY'
    ])

    symbols = ['EURUSD=X', 'USDJPY=X', 'USDKRW=X', 'USDCNY=X']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for symbol, (row, col) in zip(symbols, positions):
        data = fx_df[fx_df['symbol'] == symbol].sort_values('Date')
        if not data.empty:
            fig.add_trace(
                go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=symbol),
                row=row, col=col
            )

    fig.update_layout(height=500, showlegend=False, title="Major Exchange Rates")
    return fig

def plot_commodity_prices(comm_df):
    """Plot commodity price trends"""
    if comm_df.empty:
        return None

    fig = go.Figure()

    for name in comm_df['commodity_name'].unique():
        data = comm_df[comm_df['commodity_name'] == name].sort_values('Date')
        if not data.empty:
            # Normalize
            normalized = (data['Close'] / data['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=normalized,
                mode='lines',
                name=name
            ))

    fig.update_layout(
        title="Commodity Prices (Normalized to 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        height=500
    )

    return fig

def plot_crypto_prices(crypto_df):
    """Plot cryptocurrency prices"""
    if crypto_df.empty:
        return None

    fig = go.Figure()

    for name in crypto_df['crypto_name'].unique():
        data = crypto_df[crypto_df['crypto_name'] == name].sort_values('Date')
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Close'],
                mode='lines',
                name=name
            ))

    fig.update_layout(
        title="Cryptocurrency Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        yaxis_type="log"
    )

    return fig

# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    st.title("🌍 Country Analytics Dashboard v2.0")
    st.markdown("**Comprehensive country-centered financial analytics with REAL DATA**")

    # Sidebar
    st.sidebar.title("Navigation")

    # Data stats
    st.sidebar.markdown("### Data Sources")

    # Load all data
    indices_df = load_country_indices()
    stocks_df = load_country_stocks()
    fx_df = load_exchange_rates()
    comm_df = load_commodities()
    crypto_df = load_crypto()
    news_df = load_country_news()
    fred_df = load_fred_data()
    youtube_videos_df = load_youtube_videos()
    youtube_comments_df = load_youtube_comments()
    pagerank_df = load_pagerank()
    granger_df = load_granger()
    tfidf_df = load_tfidf()
    entity_df = load_entity_price()

    # Show data stats in sidebar
    st.sidebar.markdown(f"""
    - Indices: **{len(indices_df):,}** records
    - Stocks: **{len(stocks_df):,}** records
    - Exchange Rates: **{len(fx_df):,}** records
    - Commodities: **{len(comm_df):,}** records
    - Crypto: **{len(crypto_df):,}** records
    - News: **{len(news_df):,}** articles
    - YouTube Videos: **{len(youtube_videos_df):,}** videos
    - YouTube Comments: **{len(youtube_comments_df):,}** comments
    - FRED: **{len(fred_df):,}** records
    """)

    # Country selection
    st.sidebar.markdown("### Country Filter")
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=list(COUNTRIES.keys()),
        default=list(COUNTRIES.keys()),
        format_func=lambda x: f"{COUNTRIES[x]['flag']} {COUNTRIES[x]['name']}"
    )

    # Time period
    period = st.sidebar.selectbox(
        "Time Period",
        options=[30, 90, 180, 365, 730, 1825],
        index=3,
        format_func=lambda x: f"{x} days" if x < 365 else f"{x//365} year(s)"
    )

    # Main content tabs
    tabs = st.tabs([
        "📊 Market Overview",
        "🏢 Company Rankings",
        "💱 Currency & Commodities",
        "📰 News Analysis",
        "📺 YouTube Social",
        "🔤 Word Analytics",
        "📈 Economic Data"
    ])

    # TAB 1: Market Overview
    with tabs[0]:
        st.header("📊 Country Market Performance")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Market comparison chart
            fig = plot_market_comparison(indices_df, selected_countries)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Performance ranking
            perf_df = calculate_market_performance(indices_df, period)
            if not perf_df.empty:
                st.markdown("### Market Performance Ranking")
                for _, row in perf_df.head(11).iterrows():
                    color = "green" if row['return_pct'] > 0 else "red"
                    st.markdown(
                        f"**{row['rank']}. {row['flag']} {row['country']}** "
                        f"<span style='color:{color}'>{row['return_pct']:.1f}%</span>",
                        unsafe_allow_html=True
                    )

        # Heatmap
        if not perf_df.empty:
            fig = plot_country_heatmap(perf_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Company Rankings
    with tabs[1]:
        st.header("🏢 Top Companies by Country")

        col1, col2 = st.columns(2)

        with col1:
            selected_country = st.selectbox(
                "Select Country",
                options=selected_countries,
                format_func=lambda x: f"{COUNTRIES[x]['flag']} {COUNTRIES[x]['name']}"
            )

        with col2:
            top_n = st.slider("Show Top N Companies", 5, 50, 20)

        # Company performance
        company_df = calculate_company_ranking(stocks_df, period)
        if not company_df.empty:
            country_companies = company_df[company_df['country'] == selected_country].head(top_n)

            if not country_companies.empty:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown(f"### Top Performers in {COUNTRIES[selected_country]['name']}")
                    st.dataframe(
                        country_companies[['rank', 'symbol', 'return_pct', 'latest_price']].rename(columns={
                            'rank': 'Rank',
                            'symbol': 'Symbol',
                            'return_pct': 'Return %',
                            'latest_price': 'Price'
                        }),
                        hide_index=True
                    )

                with col2:
                    fig = plot_company_treemap(stocks_df, selected_country)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

        # Global top companies
        st.markdown("### Global Top 20 Companies (All Countries)")
        if not company_df.empty:
            global_top = company_df.head(20)

            fig = px.bar(
                global_top,
                x='symbol',
                y='return_pct',
                color='country',
                title="Top 20 Companies by Returns",
                labels={'return_pct': 'Return %', 'symbol': 'Company'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Currency & Commodities
    with tabs[2]:
        st.header("💱 Exchange Rates & Commodities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Exchange Rates")
            fig = plot_exchange_rates(fx_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Currency ranking
            fx_ranking = calculate_currency_ranking(fx_df, period)
            if not fx_ranking.empty:
                st.markdown("### Currency Changes")
                st.dataframe(fx_ranking, hide_index=True)

        with col2:
            st.markdown("### Commodities")
            fig = plot_commodity_prices(comm_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Cryptocurrencies")
            fig = plot_crypto_prices(crypto_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # TAB 4: News Analysis
    with tabs[3]:
        st.header("📰 Country News Analysis")

        news_ranking = calculate_news_ranking(news_df)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### News Coverage Ranking")
            if not news_ranking.empty:
                for _, row in news_ranking.iterrows():
                    st.markdown(
                        f"**{row['rank']}. {row['flag']} {row['country']}**: "
                        f"{row['article_count']} articles from {row['sources']} sources"
                    )

        with col2:
            if not news_df.empty:
                # News by country bar chart
                fig = px.bar(
                    news_ranking,
                    x='country',
                    y='article_count',
                    color='sources',
                    title="News Articles by Country",
                    labels={'article_count': 'Articles', 'sources': 'Unique Sources'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Latest news
        st.markdown("### Latest Headlines by Country")
        news_country = st.selectbox(
            "Select Country for News",
            options=selected_countries,
            format_func=lambda x: f"{COUNTRIES[x]['flag']} {COUNTRIES[x]['name']}",
            key="news_country"
        )

        if not news_df.empty:
            country_news = news_df[news_df['country'] == news_country].sort_values('publishedAt', ascending=False)
            for _, article in country_news.head(10).iterrows():
                st.markdown(f"**{article['title']}**")
                st.markdown(f"*{article['source']}* - {article['publishedAt']}")
                if pd.notna(article.get('description')):
                    st.markdown(article['description'][:200] + "...")
                st.markdown("---")

    # TAB 5: YouTube Social Media
    with tabs[4]:
        st.header("📺 YouTube Social Media Analysis (REAL DATA)")

        if not youtube_videos_df.empty:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Country Video Rankings")
                video_stats = youtube_videos_df.groupby('country').agg({
                    'video_id': 'count',
                    'view_count': 'sum',
                    'like_count': 'sum',
                    'comment_count': 'sum'
                }).rename(columns={'video_id': 'videos'}).reset_index()
                video_stats['engagement'] = video_stats['like_count'] + video_stats['comment_count']
                video_stats = video_stats.sort_values('view_count', ascending=False)

                for _, row in video_stats.iterrows():
                    flag = COUNTRIES.get(row['country'], {}).get('flag', '')
                    st.markdown(
                        f"**{flag} {row['country']}**: "
                        f"{row['videos']} videos, {row['view_count']:,} views"
                    )

            with col2:
                # Views by country bar chart
                fig = px.bar(
                    video_stats,
                    x='country',
                    y='view_count',
                    color='engagement',
                    title="Total Views by Country (YouTube Financial Videos)",
                    labels={'view_count': 'Total Views', 'engagement': 'Engagement'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top videos
            st.markdown("### Top Videos by Views")
            top_videos = youtube_videos_df.nlargest(10, 'view_count')
            for _, video in top_videos.iterrows():
                flag = COUNTRIES.get(video['country'], {}).get('flag', '')
                st.markdown(f"**{flag} {video['title'][:80]}...**")
                st.markdown(f"Channel: *{video['channel']}* | Views: {video['view_count']:,} | Likes: {video['like_count']:,}")
                st.markdown("---")

            # Comments section
            if not youtube_comments_df.empty:
                st.markdown("### Sample Comments by Country")
                yt_country = st.selectbox(
                    "Select Country for Comments",
                    options=selected_countries,
                    format_func=lambda x: f"{COUNTRIES[x]['flag']} {COUNTRIES[x]['name']}",
                    key="yt_country"
                )

                country_comments = youtube_comments_df[youtube_comments_df['country'] == yt_country]
                if not country_comments.empty:
                    for _, comment in country_comments.head(5).iterrows():
                        st.markdown(f"**{comment['author']}** ({comment['like_count']} likes)")
                        st.markdown(f">{comment['text'][:300]}...")
                        st.markdown("---")
        else:
            st.info("YouTube data not available. Run download_youtube_data.py to fetch real data.")

    # TAB 6: Word Analytics
    with tabs[5]:
        st.header("🔤 NLP Word Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### PageRank Word Importance")
            if not pagerank_df.empty:
                st.dataframe(pagerank_df.head(30), hide_index=True)
            else:
                st.info("PageRank data not available")

        with col2:
            st.markdown("### Granger Causality (Word -> Price)")
            if not granger_df.empty:
                st.dataframe(granger_df.head(30), hide_index=True)
            else:
                st.info("Granger causality data not available")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### TF-IDF by Sentiment")
            if not tfidf_df.empty:
                st.dataframe(tfidf_df.head(30), hide_index=True)
            else:
                st.info("TF-IDF data not available")

        with col2:
            st.markdown("### Entity-Price Correlations")
            if not entity_df.empty:
                st.dataframe(entity_df.head(30), hide_index=True)
            else:
                st.info("Entity-price data not available")

    # TAB 7: Economic Data
    with tabs[6]:
        st.header("📈 Economic Indicators (FRED)")

        if not fred_df.empty:
            # Select series
            series_options = fred_df['series_name'].unique()
            selected_series = st.multiselect(
                "Select Economic Indicators",
                options=series_options,
                default=list(series_options[:5])
            )

            if selected_series:
                fig = go.Figure()

                for series in selected_series:
                    data = fred_df[fred_df['series_name'] == series].sort_values('date')
                    if not data.empty:
                        fig.add_trace(go.Scatter(
                            x=data['date'],
                            y=data['value'],
                            mode='lines',
                            name=series
                        ))

                fig.update_layout(
                    title="FRED Economic Indicators",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("FRED data not available")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Sources:**
    - Stock Data: Yahoo Finance (Real)
    - News: NewsAPI (Real)
    - Economic Data: FRED API (Real)
    - World Bank: World Bank API (Real)

    **Last Updated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
