#!/usr/bin/env python3
"""
COUNTRY & COMPANY RANKING DASHBOARD
Comprehensive analysis of countries and their top companies across:
- GDP Rankings (World Bank)
- News Mentions & Sentiment
- Word Graph (Co-occurrence Network)
- Word-Price Predictions
- Social Media Presence
- Entity-Price Correlations
- Company Performance
"""

import streamlit as st

st.set_page_config(
    page_title="Country & Company Ranking",
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

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
SPARK_OUTPUT = BASE_DIR / "advanced_analytics" / "spark_output"
DEEP_OUTPUT = BASE_DIR / "advanced_analytics" / "deep_semantic_output"
BATCH_SPARK = BASE_DIR / "data" / "batch" / "spark_output"
DATA_DIR = BASE_DIR / "data" / "batch"

# =============================================================================
# COMPREHENSIVE COUNTRY-COMPANY MAPPINGS
# =============================================================================
COUNTRY_DATA = {
    'USA': {
        'name': 'United States',
        'code': 'US',
        'keywords': ['usa', 'us', 'america', 'american', 'united states', 'wall street', 'nasdaq', 'nyse', 'dow jones', 's&p'],
        'companies': {
            # Tech Giants
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Alphabet/Google',
            'AMZN': 'Amazon',
            'META': 'Meta/Facebook',
            'NVDA': 'NVIDIA',
            'TSLA': 'Tesla',
            'NFLX': 'Netflix',
            'INTC': 'Intel',
            'AMD': 'AMD',
            'CRM': 'Salesforce',
            'ORCL': 'Oracle',
            'CSCO': 'Cisco',
            'IBM': 'IBM',
            'ADBE': 'Adobe',
            # Finance
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
            'WFC': 'Wells Fargo',
            'C': 'Citigroup',
            'BRK.B': 'Berkshire Hathaway',
            'V': 'Visa',
            'MA': 'Mastercard',
            # Consumer
            'WMT': 'Walmart',
            'KO': 'Coca-Cola',
            'PEP': 'PepsiCo',
            'MCD': 'McDonald\'s',
            'NKE': 'Nike',
            'SBUX': 'Starbucks',
            'HD': 'Home Depot',
            # Healthcare
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer',
            'UNH': 'UnitedHealth',
            'MRK': 'Merck',
            'ABBV': 'AbbVie',
            # Energy
            'XOM': 'ExxonMobil',
            'CVX': 'Chevron',
        },
        'index': 'S&P 500, NASDAQ, Dow Jones'
    },
    'KOREA': {
        'name': 'South Korea',
        'code': 'KR',
        'keywords': ['korea', 'korean', 'seoul', 'kospi', 'samsung', 'hyundai', 'lg', 'sk'],
        'companies': {
            '005930.KS': 'Samsung Electronics',
            '000660.KS': 'SK Hynix',
            '005380.KS': 'Hyundai Motor',
            '051910.KS': 'LG Chem',
            '035420.KS': 'NAVER',
            '035720.KS': 'Kakao',
            '006400.KS': 'Samsung SDI',
            '003550.KS': 'LG',
            '066570.KS': 'LG Electronics',
            '055550.KS': 'Shinhan Financial',
        },
        'index': 'KOSPI'
    },
    'CHINA': {
        'name': 'China',
        'code': 'CN',
        'keywords': ['china', 'chinese', 'beijing', 'shanghai', 'shenzhen', 'alibaba', 'tencent', 'baidu'],
        'companies': {
            'BABA': 'Alibaba',
            'JD': 'JD.com',
            'PDD': 'Pinduoduo',
            'BIDU': 'Baidu',
            'NIO': 'NIO',
            'XPEV': 'XPeng',
            'LI': 'Li Auto',
            'TCEHY': 'Tencent',
            'NTES': 'NetEase',
            'TME': 'Tencent Music',
        },
        'index': 'Shanghai Composite, Hang Seng'
    },
    'JAPAN': {
        'name': 'Japan',
        'code': 'JP',
        'keywords': ['japan', 'japanese', 'tokyo', 'nikkei', 'toyota', 'sony', 'honda'],
        'companies': {
            'TM': 'Toyota',
            'SONY': 'Sony',
            'HMC': 'Honda',
            'NTDOY': 'Nintendo',
            'SNE': 'Sony',
            '7203.T': 'Toyota',
            '6758.T': 'Sony Group',
            '7267.T': 'Honda',
            '6861.T': 'Keyence',
            '9984.T': 'SoftBank',
        },
        'index': 'Nikkei 225'
    },
    'GERMANY': {
        'name': 'Germany',
        'code': 'DE',
        'keywords': ['germany', 'german', 'berlin', 'frankfurt', 'dax', 'volkswagen', 'mercedes', 'bmw', 'siemens'],
        'companies': {
            'SAP': 'SAP',
            'DB': 'Deutsche Bank',
            'VWAGY': 'Volkswagen',
            'BMWYY': 'BMW',
            'DDAIF': 'Mercedes-Benz',
            'SIEGY': 'Siemens',
            'ADDYY': 'Adidas',
            'BAYRY': 'Bayer',
            'BASFY': 'BASF',
            'DTEKY': 'Deutsche Telekom',
        },
        'index': 'DAX'
    },
    'UK': {
        'name': 'United Kingdom',
        'code': 'GB',
        'keywords': ['uk', 'britain', 'british', 'london', 'ftse', 'england'],
        'companies': {
            'BP': 'BP',
            'HSBC': 'HSBC',
            'SHEL': 'Shell',
            'GSK': 'GlaxoSmithKline',
            'AZN': 'AstraZeneca',
            'RIO': 'Rio Tinto',
            'UL': 'Unilever',
            'VOD': 'Vodafone',
            'BTI': 'British American Tobacco',
            'LRLCY': 'L\'Oreal',
        },
        'index': 'FTSE 100'
    },
    'FRANCE': {
        'name': 'France',
        'code': 'FR',
        'keywords': ['france', 'french', 'paris', 'cac', 'lvmh', 'total'],
        'companies': {
            'TTE': 'TotalEnergies',
            'LVMUY': 'LVMH',
            'SNY': 'Sanofi',
            'BNPQY': 'BNP Paribas',
            'OR': 'L\'Oreal',
            'DANOY': 'Danone',
            'AI.PA': 'Air Liquide',
            'CAP': 'Capgemini',
            'ENGI': 'Engie',
            'RNO.PA': 'Renault',
        },
        'index': 'CAC 40'
    },
    'INDIA': {
        'name': 'India',
        'code': 'IN',
        'keywords': ['india', 'indian', 'mumbai', 'sensex', 'nifty', 'infosys', 'tata', 'reliance'],
        'companies': {
            'INFY': 'Infosys',
            'WIT': 'Wipro',
            'HDB': 'HDFC Bank',
            'IBN': 'ICICI Bank',
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy',
            'HDFCBANK.NS': 'HDFC Bank',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'ITC.NS': 'ITC',
            'KOTAKBANK.NS': 'Kotak Mahindra',
        },
        'index': 'SENSEX, NIFTY 50'
    },
    'TAIWAN': {
        'name': 'Taiwan',
        'code': 'TW',
        'keywords': ['taiwan', 'taiwanese', 'taipei', 'tsmc', 'foxconn'],
        'companies': {
            'TSM': 'TSMC',
            '2330.TW': 'TSMC',
            '2317.TW': 'Hon Hai/Foxconn',
            '2454.TW': 'MediaTek',
            '2412.TW': 'Chunghwa Telecom',
            '1301.TW': 'Formosa Plastics',
            '2882.TW': 'Cathay Financial',
            '3008.TW': 'Largan Precision',
            '2308.TW': 'Delta Electronics',
            '2881.TW': 'Fubon Financial',
        },
        'index': 'TAIEX'
    },
    'CANADA': {
        'name': 'Canada',
        'code': 'CA',
        'keywords': ['canada', 'canadian', 'toronto', 'tsx'],
        'companies': {
            'TD': 'Toronto-Dominion Bank',
            'RY': 'Royal Bank of Canada',
            'SHOP': 'Shopify',
            'ENB': 'Enbridge',
            'CNR': 'Canadian National Railway',
            'BNS': 'Bank of Nova Scotia',
            'BCE': 'BCE Inc',
            'TRP': 'TC Energy',
            'SU': 'Suncor Energy',
            'BMO': 'Bank of Montreal',
        },
        'index': 'TSX'
    },
    'RUSSIA': {
        'name': 'Russia',
        'code': 'RU',
        'keywords': ['russia', 'russian', 'moscow', 'gazprom', 'lukoil', 'sberbank'],
        'companies': {
            'LUKOY': 'Lukoil',
            'OGZPY': 'Gazprom',
            'SBRCY': 'Sberbank',
            'ROSN.ME': 'Rosneft',
            'GMKN.ME': 'Norilsk Nickel',
            'NVTK.ME': 'Novatek',
            'TATN.ME': 'Tatneft',
            'YNDX': 'Yandex',
            'MTSS.ME': 'MTS',
            'PLZL.ME': 'Polyus',
        },
        'index': 'MOEX'
    },
}

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=300)
def load_csv_safe(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except:
        pass
    return None


def load_all_data():
    data = {}
    data['gdp_rankings'] = load_csv_safe(BATCH_SPARK / "gdp_rankings.csv")
    data['korea_usa'] = load_csv_safe(DEEP_OUTPUT / "korea_usa_gdp_comparison.csv")
    data['pagerank'] = load_csv_safe(SPARK_OUTPUT / "word_pagerank.csv")
    data['word_price'] = load_csv_safe(SPARK_OUTPUT / "word_before_price_lag1.csv")
    data['word_graph'] = load_csv_safe(SPARK_OUTPUT / "word_graph_edges.csv")
    data['word_comparison'] = load_csv_safe(SPARK_OUTPUT / "word_comparison_sources.csv")
    data['entity_price'] = load_csv_safe(DEEP_OUTPUT / "entity_price_correlation.csv")
    data['news'] = load_csv_safe(DATA_DIR / "news" / "financial_news_headlines.csv")
    data['ngrams'] = load_csv_safe(DEEP_OUTPUT / "ngram_analysis.csv")
    data['reddit'] = load_csv_safe(SPARK_OUTPUT / "reddit_comments.csv")
    data['youtube'] = load_csv_safe(SPARK_OUTPUT / "youtube_comments.csv")
    data['stock_summary'] = load_csv_safe(BATCH_SPARK / "stock_summary_stats.csv")
    data['stock_corr'] = load_csv_safe(BATCH_SPARK / "stock_correlations.csv")
    return data


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_country_news_mentions(news_df):
    """Analyze country mentions in news headlines"""
    if news_df is None:
        return None

    results = []
    for country, info in COUNTRY_DATA.items():
        mention_count = 0
        sentiment_scores = []
        company_mentions = {}

        for _, row in news_df.iterrows():
            headline = str(row.get('headline', '')).lower()

            # Check country keywords
            for kw in info['keywords']:
                if kw in headline:
                    mention_count += 1
                    if pd.notna(row.get('sentiment_score')):
                        try:
                            sentiment_scores.append(float(row['sentiment_score']))
                        except:
                            pass
                    break

            # Check company mentions
            for symbol, name in info['companies'].items():
                if symbol.lower() in headline or name.lower() in headline:
                    company_mentions[symbol] = company_mentions.get(symbol, 0) + 1

        results.append({
            'country': country,
            'country_name': info['name'],
            'news_mentions': mention_count,
            'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
            'company_mentions': sum(company_mentions.values()),
            'top_company': max(company_mentions, key=company_mentions.get) if company_mentions else '-',
            'num_companies_mentioned': len(company_mentions)
        })

    return pd.DataFrame(results).sort_values('news_mentions', ascending=False)


def analyze_country_word_graph(word_graph_df):
    """Analyze country presence in word graph"""
    if word_graph_df is None:
        return None

    results = []
    for country, info in COUNTRY_DATA.items():
        total_weight = 0
        connections = 0
        connected_words = set()

        for kw in info['keywords']:
            mask = (word_graph_df['source'].str.lower() == kw) | (word_graph_df['target'].str.lower() == kw)
            edges = word_graph_df[mask]
            total_weight += edges['weight'].sum()
            connections += len(edges)
            connected_words.update(edges['source'].tolist() + edges['target'].tolist())

        # Check company names too
        for name in info['companies'].values():
            name_lower = name.lower().split()[0]  # First word of company name
            mask = (word_graph_df['source'].str.lower() == name_lower) | (word_graph_df['target'].str.lower() == name_lower)
            edges = word_graph_df[mask]
            total_weight += edges['weight'].sum()
            connections += len(edges)

        results.append({
            'country': country,
            'country_name': info['name'],
            'graph_weight': total_weight,
            'graph_connections': connections,
            'unique_words': len(connected_words)
        })

    return pd.DataFrame(results).sort_values('graph_weight', ascending=False)


def analyze_country_word_price(word_price_df):
    """Analyze country keywords in word-price predictions"""
    if word_price_df is None:
        return None

    results = []
    for country, info in COUNTRY_DATA.items():
        country_rows = []

        for kw in info['keywords']:
            match = word_price_df[word_price_df['word'].str.lower() == kw]
            if len(match) > 0:
                country_rows.append(match.iloc[0])

        # Check company names
        for name in info['companies'].values():
            name_lower = name.lower().split()[0]
            match = word_price_df[word_price_df['word'].str.lower() == name_lower]
            if len(match) > 0:
                country_rows.append(match.iloc[0])

        if country_rows:
            avg_bias = np.mean([r.get('predictive_bias', 0) for r in country_rows])
            total_count = sum([r.get('total', 0) for r in country_rows])
            results.append({
                'country': country,
                'country_name': info['name'],
                'keywords_found': len(country_rows),
                'total_occurrences': total_count,
                'avg_predictive_bias': avg_bias,
                'prediction': 'UP' if avg_bias > 0.05 else ('DOWN' if avg_bias < -0.05 else 'NEUTRAL')
            })

    return pd.DataFrame(results).sort_values('total_occurrences', ascending=False) if results else None


def analyze_country_social_media(reddit_df, youtube_df):
    """Analyze country/company mentions in social media"""
    if reddit_df is None and youtube_df is None:
        return None

    results = []
    for country, info in COUNTRY_DATA.items():
        reddit_mentions = 0
        youtube_mentions = 0
        reddit_sentiment = []
        youtube_sentiment = []

        # Check Reddit
        if reddit_df is not None:
            for _, row in reddit_df.iterrows():
                text = str(row.get('text', '')).lower()
                for kw in info['keywords'] + list(info['companies'].values()):
                    kw_lower = kw.lower() if isinstance(kw, str) else str(kw).lower()
                    if kw_lower.split()[0] in text:  # Check first word
                        reddit_mentions += 1
                        if pd.notna(row.get('sentiment_score')):
                            reddit_sentiment.append(row['sentiment_score'])
                        break

        # Check YouTube
        if youtube_df is not None:
            for _, row in youtube_df.iterrows():
                text = str(row.get('text', '')).lower()
                for kw in info['keywords'] + list(info['companies'].values()):
                    kw_lower = kw.lower() if isinstance(kw, str) else str(kw).lower()
                    if kw_lower.split()[0] in text:
                        youtube_mentions += 1
                        if pd.notna(row.get('sentiment_score')):
                            youtube_sentiment.append(row['sentiment_score'])
                        break

        results.append({
            'country': country,
            'country_name': info['name'],
            'reddit_mentions': reddit_mentions,
            'youtube_mentions': youtube_mentions,
            'total_social': reddit_mentions + youtube_mentions,
            'reddit_sentiment': np.mean(reddit_sentiment) if reddit_sentiment else 0,
            'youtube_sentiment': np.mean(youtube_sentiment) if youtube_sentiment else 0
        })

    return pd.DataFrame(results).sort_values('total_social', ascending=False)


def analyze_country_entity_price(entity_price_df):
    """Analyze company-price correlations by country"""
    if entity_price_df is None:
        return None

    results = []
    for country, info in COUNTRY_DATA.items():
        country_companies = []
        total_mentions = 0
        same_day_corrs = []
        next_day_corrs = []

        for _, row in entity_price_df.iterrows():
            symbol = row.get('symbol', '')
            if symbol in info['companies']:
                country_companies.append(symbol)
                total_mentions += row.get('total_mentions', 0)
                if pd.notna(row.get('same_day_corr')):
                    same_day_corrs.append(row['same_day_corr'])
                if pd.notna(row.get('next_day_corr')):
                    next_day_corrs.append(row['next_day_corr'])

        if country_companies:
            results.append({
                'country': country,
                'country_name': info['name'],
                'companies_tracked': len(country_companies),
                'company_list': ', '.join(country_companies),
                'total_mentions': total_mentions,
                'avg_same_day_corr': np.mean(same_day_corrs) if same_day_corrs else 0,
                'avg_next_day_corr': np.mean(next_day_corrs) if next_day_corrs else 0,
                'price_impact': 'HIGH' if abs(np.mean(same_day_corrs) if same_day_corrs else 0) > 0.03 else 'LOW'
            })

    return pd.DataFrame(results).sort_values('total_mentions', ascending=False) if results else None


def create_unified_country_ranking(data):
    """Create comprehensive unified ranking"""

    # Get all analyses
    news_analysis = analyze_country_news_mentions(data.get('news'))
    graph_analysis = analyze_country_word_graph(data.get('word_graph'))
    word_price_analysis = analyze_country_word_price(data.get('word_price'))
    social_analysis = analyze_country_social_media(data.get('reddit'), data.get('youtube'))
    entity_analysis = analyze_country_entity_price(data.get('entity_price'))
    gdp_df = data.get('gdp_rankings')

    # Start with base country list
    base = pd.DataFrame([
        {'country': k, 'country_name': v['name'], 'num_top_companies': len(v['companies']), 'index': v['index']}
        for k, v in COUNTRY_DATA.items()
    ])

    # Merge all analyses
    if news_analysis is not None:
        news_analysis['news_rank'] = news_analysis['news_mentions'].rank(ascending=False)
        base = base.merge(news_analysis[['country', 'news_mentions', 'avg_sentiment', 'news_rank']], on='country', how='left')

    if graph_analysis is not None:
        graph_analysis['graph_rank'] = graph_analysis['graph_weight'].rank(ascending=False)
        base = base.merge(graph_analysis[['country', 'graph_weight', 'graph_connections', 'graph_rank']], on='country', how='left')

    if social_analysis is not None:
        social_analysis['social_rank'] = social_analysis['total_social'].rank(ascending=False)
        base = base.merge(social_analysis[['country', 'total_social', 'reddit_mentions', 'youtube_mentions', 'social_rank']], on='country', how='left')

    if entity_analysis is not None:
        entity_analysis['entity_rank'] = entity_analysis['total_mentions'].rank(ascending=False)
        base = base.merge(entity_analysis[['country', 'total_mentions', 'avg_same_day_corr', 'entity_rank']], on='country', how='left')

    # GDP data
    if gdp_df is not None:
        code_map = {v['code']: k for k, v in COUNTRY_DATA.items()}
        gdp_df['country'] = gdp_df['country_code'].map(code_map)
        gdp_df = gdp_df[gdp_df['country'].notna()]
        gdp_df['gdp_rank'] = gdp_df['rank']
        base = base.merge(gdp_df[['country', 'value', 'gdp_growth', 'gdp_rank']], on='country', how='left')

    # Calculate composite rank
    rank_cols = [c for c in base.columns if c.endswith('_rank')]
    for col in rank_cols:
        base[col] = base[col].fillna(50)  # Default rank for missing data

    if rank_cols:
        base['composite_score'] = base[rank_cols].mean(axis=1)
        base['final_rank'] = base['composite_score'].rank().astype(int)
        base = base.sort_values('final_rank')

    return base


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    st.title("🌍 Country & Company Ranking Dashboard")
    st.markdown("""
    **국가 및 기업 종합 순위** - Comprehensive ranking of countries and their top companies
    across all data sources: News, Word Graph, Social Media, Entity-Price, GDP
    """)

    # Load data
    with st.spinner("Loading all data..."):
        data = load_all_data()

    loaded = sum(1 for v in data.values() if v is not None)
    st.success(f"✅ Loaded {loaded}/{len(data)} data sources")

    # Sidebar
    st.sidebar.title("🌍 Analysis Selection")

    section = st.sidebar.radio(
        "Select View",
        [
            "🏆 Unified Country Ranking",
            "📰 News Mentions Analysis",
            "🔗 Word Graph Analysis",
            "💬 Social Media Analysis",
            "💹 Entity-Price Analysis",
            "📈 GDP & Economic Ranking",
            "🏢 Company Details",
            "📊 Korea vs USA Deep Dive"
        ]
    )

    # Show country list
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Countries Analyzed")
    for c, info in COUNTRY_DATA.items():
        st.sidebar.markdown(f"**{c}**: {len(info['companies'])} companies")

    # ==========================================================================
    # UNIFIED RANKING
    # ==========================================================================
    if "Unified" in section:
        st.header("🏆 Unified Country Ranking")

        unified = create_unified_country_ranking(data)

        if unified is not None:
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                top = unified.iloc[0]
                st.metric("🥇 #1 Country", top['country'], f"Score: {top.get('composite_score', 0):.1f}")
            with col2:
                st.metric("Total Countries", len(unified))
            with col3:
                st.metric("Total Companies", sum(len(COUNTRY_DATA[c]['companies']) for c in COUNTRY_DATA))
            with col4:
                if 'news_mentions' in unified.columns:
                    st.metric("Total News Mentions", int(unified['news_mentions'].sum()))

            # Main ranking chart
            fig = px.bar(
                unified,
                x='country',
                y='final_rank',
                color='composite_score',
                color_continuous_scale='RdYlGn_r',
                title="🏆 Country Rankings (Lower = Better)",
                hover_data=['country_name', 'num_top_companies']
            )
            fig.update_layout(yaxis={'autorange': 'reversed'})
            st.plotly_chart(fig, use_container_width=True)

            # Breakdown by category
            st.subheader("📊 Ranking Breakdown by Category")

            rank_cols = [c for c in unified.columns if c.endswith('_rank') and c != 'final_rank']
            if rank_cols:
                fig = go.Figure()
                for _, row in unified.head(8).iterrows():
                    values = [100 - row[col] for col in rank_cols]
                    values.append(values[0])

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=[c.replace('_rank', '').upper() for c in rank_cols] + [rank_cols[0].replace('_rank', '').upper()],
                        name=row['country'],
                        fill='toself',
                        opacity=0.6
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Multi-Dimensional Country Comparison (Higher = Better)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            # Full table
            st.subheader("📋 Complete Rankings Table")
            st.dataframe(unified, use_container_width=True)

            csv = unified.to_csv(index=False)
            st.download_button("📥 Download Rankings", csv, "country_rankings.csv", "text/csv")

    # ==========================================================================
    # NEWS MENTIONS
    # ==========================================================================
    elif "News Mentions" in section:
        st.header("📰 News Mentions Analysis")

        news_analysis = analyze_country_news_mentions(data.get('news'))

        if news_analysis is not None:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    news_analysis,
                    x='country',
                    y='news_mentions',
                    color='avg_sentiment',
                    color_continuous_scale='RdYlGn',
                    title="Country Mentions in Financial News"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    news_analysis,
                    x='news_mentions',
                    y='company_mentions',
                    size='num_companies_mentioned',
                    color='country',
                    title="Country vs Company Mentions",
                    hover_data=['top_company']
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(news_analysis, use_container_width=True)

    # ==========================================================================
    # WORD GRAPH
    # ==========================================================================
    elif "Word Graph" in section:
        st.header("🔗 Word Graph Analysis")

        graph_analysis = analyze_country_word_graph(data.get('word_graph'))

        if graph_analysis is not None:
            filtered = graph_analysis[graph_analysis['graph_weight'] > 0]

            if len(filtered) > 0:
                fig = px.bar(
                    filtered,
                    x='country',
                    y='graph_weight',
                    color='graph_connections',
                    title="Country Presence in Word Co-occurrence Network"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(graph_analysis, use_container_width=True)

    # ==========================================================================
    # SOCIAL MEDIA
    # ==========================================================================
    elif "Social Media" in section:
        st.header("💬 Social Media Analysis (Reddit & YouTube)")

        social_analysis = analyze_country_social_media(data.get('reddit'), data.get('youtube'))

        if social_analysis is not None:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    social_analysis,
                    x='country',
                    y=['reddit_mentions', 'youtube_mentions'],
                    barmode='group',
                    title="Social Media Mentions by Country"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    social_analysis,
                    x='reddit_sentiment',
                    y='youtube_sentiment',
                    size='total_social',
                    color='country',
                    title="Sentiment: Reddit vs YouTube"
                )
                fig.add_hline(y=0, line_dash="dash")
                fig.add_vline(x=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(social_analysis, use_container_width=True)

    # ==========================================================================
    # ENTITY-PRICE
    # ==========================================================================
    elif "Entity-Price" in section:
        st.header("💹 Entity-Price Correlation Analysis")

        entity_analysis = analyze_country_entity_price(data.get('entity_price'))

        if entity_analysis is not None:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    entity_analysis,
                    x='country',
                    y='total_mentions',
                    color='avg_same_day_corr',
                    color_continuous_scale='RdBu',
                    title="Company Mentions by Country"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    entity_analysis,
                    x='avg_same_day_corr',
                    y='avg_next_day_corr',
                    size='total_mentions',
                    color='country',
                    text='country',
                    title="Same-Day vs Next-Day Price Impact"
                )
                fig.add_hline(y=0, line_dash="dash")
                fig.add_vline(x=0, line_dash="dash")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(entity_analysis, use_container_width=True)

        # Show raw entity data
        if data.get('entity_price') is not None:
            st.subheader("📊 Individual Company Data")
            st.dataframe(data['entity_price'], use_container_width=True)

    # ==========================================================================
    # GDP RANKING
    # ==========================================================================
    elif "GDP" in section:
        st.header("📈 GDP & Economic Ranking")

        gdp_df = data.get('gdp_rankings')

        if gdp_df is not None:
            # Filter for our countries
            code_map = {v['code']: k for k, v in COUNTRY_DATA.items()}
            gdp_filtered = gdp_df[gdp_df['country_code'].isin(code_map.keys())].copy()
            gdp_filtered['country'] = gdp_filtered['country_code'].map(code_map)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    gdp_filtered,
                    x='country',
                    y='value',
                    color='gdp_growth',
                    color_continuous_scale='RdYlGn',
                    title="GDP by Country (USD)"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    gdp_filtered.sort_values('gdp_growth', ascending=False),
                    x='country',
                    y='gdp_growth',
                    color='gdp_growth',
                    color_continuous_scale='RdYlGn',
                    title="GDP Growth Rate (%)"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(gdp_filtered, use_container_width=True)

    # ==========================================================================
    # COMPANY DETAILS
    # ==========================================================================
    elif "Company Details" in section:
        st.header("🏢 Company Details by Country")

        selected_country = st.selectbox("Select Country", list(COUNTRY_DATA.keys()))

        if selected_country:
            info = COUNTRY_DATA[selected_country]

            st.subheader(f"🌍 {info['name']} ({selected_country})")
            st.markdown(f"**Stock Index:** {info['index']}")
            st.markdown(f"**Keywords:** {', '.join(info['keywords'])}")

            # Company table
            st.subheader(f"🏢 Top Companies ({len(info['companies'])})")

            company_df = pd.DataFrame([
                {'Symbol': k, 'Company': v}
                for k, v in info['companies'].items()
            ])
            st.dataframe(company_df, use_container_width=True)

    # ==========================================================================
    # KOREA VS USA
    # ==========================================================================
    elif "Korea vs USA" in section:
        st.header("📊 Korea vs USA Deep Dive")

        kr_usa = data.get('korea_usa')

        if kr_usa is not None:
            fig = make_subplots(rows=2, cols=2, subplot_titles=(
                "GDP Over Time (Trillion USD)",
                "GDP Growth Rate (%)",
                "Korea/USA GDP Ratio (%)",
                "Growth Difference (Korea - USA)"
            ))

            # GDP
            fig.add_trace(go.Scatter(x=kr_usa['year'], y=kr_usa['usa_gdp']/1e12, name='USA', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=kr_usa['year'], y=kr_usa['korea_gdp']/1e12, name='Korea', line=dict(color='red')), row=1, col=1)

            # Growth
            fig.add_trace(go.Scatter(x=kr_usa['year'], y=kr_usa['usa_growth'], name='USA Growth', line=dict(color='blue')), row=1, col=2)
            fig.add_trace(go.Scatter(x=kr_usa['year'], y=kr_usa['korea_growth'], name='Korea Growth', line=dict(color='red')), row=1, col=2)

            # Ratio
            fig.add_trace(go.Scatter(x=kr_usa['year'], y=kr_usa['ratio_korea_usa']*100, fill='tozeroy', name='Ratio'), row=2, col=1)

            # Difference
            colors = ['green' if x > 0 else 'red' for x in kr_usa['growth_diff']]
            fig.add_trace(go.Bar(x=kr_usa['year'], y=kr_usa['growth_diff'], name='Diff', marker_color=colors), row=2, col=2)

            fig.update_layout(height=700, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Key metrics
            latest = kr_usa.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("USA GDP", f"${latest['usa_gdp']/1e12:.1f}T", f"{latest['usa_growth']:.1f}%")
            with col2:
                st.metric("Korea GDP", f"${latest['korea_gdp']/1e12:.2f}T", f"{latest['korea_growth']:.1f}%")
            with col3:
                st.metric("Korea/USA", f"{latest['ratio_korea_usa']*100:.1f}%")
            with col4:
                st.metric("Correlation", f"{kr_usa['korea_growth'].corr(kr_usa['usa_growth']):.3f}")

            st.dataframe(kr_usa.tail(15), use_container_width=True)


if __name__ == "__main__":
    main()
