#!/usr/bin/env python3
"""
INTEGRATED RANKING DASHBOARD
Combines all processed data files and provides unified rankings:
- Word Rankings (PageRank, TF-IDF, Frequency)
- Stock/Asset Rankings (Correlation, Performance, Volatility)
- Topic Rankings (LDA)
- Entity Rankings (Mention frequency, Price impact)
- Economic Rankings (GDP, Growth)
"""

import streamlit as st

st.set_page_config(
    page_title="Integrated Ranking Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# =============================================================================
# PATHS TO ALL OUTPUT FILES
# =============================================================================
BASE_DIR = Path(__file__).parent
SPARK_OUTPUT = BASE_DIR / "advanced_analytics" / "spark_output"
DEEP_OUTPUT = BASE_DIR / "advanced_analytics" / "deep_semantic_output"
BATCH_SPARK = BASE_DIR / "data" / "batch" / "spark_output"
ECON_OUTPUT = BASE_DIR / "economics_social_cultural" / "output"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data(ttl=300)
def load_csv_safe(path):
    """Safely load CSV file"""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Error loading {path.name}: {e}")
    return None


def load_all_data():
    """Load all processed data files"""
    data = {}

    # Spark Output (Advanced Analytics)
    data['pagerank'] = load_csv_safe(SPARK_OUTPUT / "word_pagerank.csv")
    data['word_price'] = load_csv_safe(SPARK_OUTPUT / "word_before_price_lag1.csv")
    data['granger'] = load_csv_safe(SPARK_OUTPUT / "granger_correlations.csv")
    data['word_graph'] = load_csv_safe(SPARK_OUTPUT / "word_graph_edges.csv")
    data['word_comparison'] = load_csv_safe(SPARK_OUTPUT / "word_comparison_sources.csv")
    data['sentiment_source'] = load_csv_safe(SPARK_OUTPUT / "sentiment_by_source.csv")

    # Deep Semantic Output
    data['tfidf'] = load_csv_safe(DEEP_OUTPUT / "tfidf_by_sentiment.csv")
    data['lda_topics'] = load_csv_safe(DEEP_OUTPUT / "lda_topics.csv")
    data['lda_dist'] = load_csv_safe(DEEP_OUTPUT / "lda_topic_distribution.csv")
    data['ngrams'] = load_csv_safe(DEEP_OUTPUT / "ngram_analysis.csv")
    data['word2vec'] = load_csv_safe(DEEP_OUTPUT / "word2vec_similarities.csv")
    data['entity_price'] = load_csv_safe(DEEP_OUTPUT / "entity_price_correlation.csv")
    data['korea_usa'] = load_csv_safe(DEEP_OUTPUT / "korea_usa_gdp_comparison.csv")
    data['kospi'] = load_csv_safe(DEEP_OUTPUT / "kospi_index.csv")
    data['usa_market'] = load_csv_safe(DEEP_OUTPUT / "usa_market_performance.csv")

    # Batch Spark Output
    data['stock_corr'] = load_csv_safe(BATCH_SPARK / "stock_correlations.csv")
    data['stock_summary'] = load_csv_safe(BATCH_SPARK / "stock_summary_stats.csv")
    data['gdp_rankings'] = load_csv_safe(BATCH_SPARK / "gdp_rankings.csv")
    data['word_freq'] = load_csv_safe(BATCH_SPARK / "word_frequencies.csv")
    data['company_sentiment'] = load_csv_safe(BATCH_SPARK / "company_sentiment_analysis.csv")

    # Economics Output
    data['correlation_matrix'] = load_csv_safe(ECON_OUTPUT / "correlation_matrix.csv")
    data['regression'] = load_csv_safe(ECON_OUTPUT / "linear_regression_results.csv")
    data['decade'] = load_csv_safe(ECON_OUTPUT / "decade_analysis.csv")
    data['merged_assets'] = load_csv_safe(ECON_OUTPUT / "merged_multi_asset_data.csv")

    return data


# =============================================================================
# RANKING CREATION FUNCTIONS
# =============================================================================
def create_word_master_ranking(data):
    """Create unified word ranking combining PageRank, TF-IDF, Frequency"""
    rankings = []

    # PageRank scores
    if data['pagerank'] is not None:
        pr = data['pagerank'].copy()
        pr['pagerank_rank'] = pr['rank'] if 'rank' in pr.columns else range(1, len(pr)+1)
        pr['pagerank_score'] = pr['pagerank']
        rankings.append(pr[['word', 'pagerank_rank', 'pagerank_score', 'frequency']])

    # TF-IDF scores (average across sentiments)
    if data['tfidf'] is not None:
        tfidf = data['tfidf'].groupby('word').agg({
            'tfidf_score': 'mean'
        }).reset_index()
        tfidf['tfidf_rank'] = tfidf['tfidf_score'].rank(ascending=False).astype(int)
        rankings.append(tfidf[['word', 'tfidf_rank', 'tfidf_score']])

    # Word frequency
    if data['word_freq'] is not None:
        wf = data['word_freq'].copy()
        if 'count' in wf.columns:
            wf['freq_rank'] = wf['count'].rank(ascending=False).astype(int)
            rankings.append(wf[['word', 'freq_rank', 'count']])

    # Combine all rankings
    if rankings:
        master = rankings[0]
        for r in rankings[1:]:
            master = master.merge(r, on='word', how='outer')

        # Calculate composite score (lower rank = better)
        rank_cols = [c for c in master.columns if c.endswith('_rank')]
        master['avg_rank'] = master[rank_cols].mean(axis=1)
        master['composite_rank'] = master['avg_rank'].rank().astype(int)
        master = master.sort_values('composite_rank')

        return master
    return None


def create_asset_ranking(data):
    """Create asset/stock rankings based on performance metrics"""
    rankings = []

    # Stock summary stats
    if data['stock_summary'] is not None:
        ss = data['stock_summary'].copy()
        if 'symbol' in ss.columns:
            # Rank by various metrics
            if 'avg_return' in ss.columns:
                ss['return_rank'] = ss['avg_return'].rank(ascending=False).astype(int)
            if 'volatility' in ss.columns:
                ss['volatility_rank'] = ss['volatility'].rank(ascending=True).astype(int)  # Lower is better
            if 'sharpe_ratio' in ss.columns:
                ss['sharpe_rank'] = ss['sharpe_ratio'].rank(ascending=False).astype(int)
            rankings.append(ss)

    # Entity price correlation
    if data['entity_price'] is not None:
        ep = data['entity_price'].copy()
        if 'symbol' in ep.columns:
            if 'same_day_corr' in ep.columns:
                ep['impact_rank'] = ep['same_day_corr'].abs().rank(ascending=False).astype(int)
            rankings.append(ep)

    # Combine
    if rankings:
        master = rankings[0]
        for r in rankings[1:]:
            if 'symbol' in r.columns and 'symbol' in master.columns:
                master = master.merge(r, on='symbol', how='outer', suffixes=('', '_dup'))

        # Calculate composite
        rank_cols = [c for c in master.columns if c.endswith('_rank')]
        if rank_cols:
            master['avg_rank'] = master[rank_cols].mean(axis=1)
            master['composite_rank'] = master['avg_rank'].rank().astype(int)
            master = master.sort_values('composite_rank')

        return master
    return None


def create_topic_ranking(data):
    """Create topic rankings from LDA analysis"""
    if data['lda_dist'] is not None and data['lda_topics'] is not None:
        # Topic distribution
        dist = data['lda_dist'].copy()
        if 'dominant_topic' in dist.columns:
            dist['topic_rank'] = dist['count'].rank(ascending=False).astype(int)

        # Get top words per topic
        topics = data['lda_topics']
        topic_names = topics.groupby('topic_id').apply(
            lambda x: ', '.join(x.nsmallest(3, 'weight' if 'weight' in x.columns else 'word')['word'].tolist())
            if 'word' in x.columns else f"Topic {x['topic_id'].iloc[0]}"
        ).to_dict()

        if 'dominant_topic' in dist.columns:
            dist['topic_keywords'] = dist['dominant_topic'].map(topic_names)

        return dist
    return None


def create_correlation_ranking(data):
    """Create correlation-based rankings"""
    rankings = []

    # Stock correlations
    if data['stock_corr'] is not None:
        sc = data['stock_corr'].copy()
        if 'correlation' in sc.columns:
            sc['corr_rank'] = sc['correlation'].abs().rank(ascending=False).astype(int)
            rankings.append(sc)

    # Multi-asset correlation matrix
    if data['correlation_matrix'] is not None:
        cm = data['correlation_matrix'].copy()
        if cm.shape[0] > 0:
            # Melt to long format
            cm_long = cm.set_index(cm.columns[0]).stack().reset_index()
            cm_long.columns = ['asset1', 'asset2', 'correlation']
            cm_long = cm_long[cm_long['asset1'] != cm_long['asset2']]
            cm_long['abs_corr'] = cm_long['correlation'].abs()
            cm_long['corr_rank'] = cm_long['abs_corr'].rank(ascending=False).astype(int)
            rankings.append(cm_long)

    if rankings:
        return rankings[0]  # Return first available
    return None


def create_country_ranking(data):
    """Create country/economy rankings"""
    if data['gdp_rankings'] is not None:
        return data['gdp_rankings'].copy()

    if data['korea_usa'] is not None:
        kr_usa = data['korea_usa'].copy()
        if 'year' in kr_usa.columns:
            latest = kr_usa.iloc[-1]
            ranking = pd.DataFrame([
                {'country': 'USA', 'gdp': latest.get('usa_gdp', 0), 'growth': latest.get('usa_growth', 0)},
                {'country': 'Korea', 'gdp': latest.get('korea_gdp', 0), 'growth': latest.get('korea_growth', 0)}
            ])
            ranking['gdp_rank'] = ranking['gdp'].rank(ascending=False).astype(int)
            ranking['growth_rank'] = ranking['growth'].rank(ascending=False).astype(int)
            return ranking
    return None


def create_ngram_ranking(data):
    """Create N-gram phrase rankings"""
    if data['ngrams'] is not None:
        ng = data['ngrams'].copy()
        if 'total_count' in ng.columns:
            ng['ngram_rank'] = ng['total_count'].rank(ascending=False).astype(int)
            ng = ng.sort_values('ngram_rank')
        return ng
    return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_ranking_bar(df, x_col, y_col, title, color_col=None, top_n=20):
    """Create a horizontal bar chart for rankings"""
    if df is None or len(df) == 0:
        return None

    df_plot = df.head(top_n).copy()

    fig = px.bar(
        df_plot,
        y=x_col,
        x=y_col,
        orientation='h',
        color=color_col if color_col and color_col in df_plot.columns else None,
        title=title,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    return fig


def plot_ranking_treemap(df, path_col, value_col, title, top_n=30):
    """Create treemap for rankings"""
    if df is None or len(df) == 0:
        return None

    df_plot = df.head(top_n).copy()

    fig = px.treemap(
        df_plot,
        path=[path_col],
        values=value_col,
        title=title,
        color=value_col,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    return fig


def plot_composite_ranking(df, name_col, rank_cols, title):
    """Create radar/spider chart for composite rankings"""
    if df is None or len(df) == 0:
        return None

    df_plot = df.head(10).copy()

    # Normalize ranks to 0-1 scale (inverted so higher = better)
    max_rank = df_plot[rank_cols].max().max()

    fig = go.Figure()

    for idx, row in df_plot.iterrows():
        values = [1 - (row[col] / max_rank) if pd.notna(row[col]) else 0 for col in rank_cols]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=rank_cols + [rank_cols[0]],
            name=str(row[name_col])[:15],
            fill='toself',
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        height=500
    )
    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    st.title("🏆 Integrated Ranking Dashboard")
    st.markdown("""
    **통합 순위 대시보드** - All processed data combined into unified rankings
    """)

    # Load all data
    with st.spinner("Loading all processed data..."):
        data = load_all_data()

    # Count loaded files
    loaded = sum(1 for v in data.values() if v is not None)
    total = len(data)
    st.success(f"✅ Loaded {loaded}/{total} data files")

    # Sidebar navigation
    st.sidebar.title("🏆 Ranking Categories")

    category = st.sidebar.radio(
        "Select Category",
        [
            "📊 Overview (All Rankings)",
            "📝 Word Master Ranking",
            "💹 Asset/Stock Ranking",
            "🎯 Topic Ranking (LDA)",
            "🔗 Correlation Ranking",
            "🌍 Country/Economy Ranking",
            "📚 N-gram Phrase Ranking",
            "📁 Raw Data Explorer"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Data Sources")
    st.sidebar.markdown(f"""
    - **Spark Output:** {sum(1 for k in ['pagerank','word_price','granger','word_graph'] if data.get(k) is not None)}/4
    - **Deep Semantic:** {sum(1 for k in ['tfidf','lda_topics','ngrams','word2vec','entity_price'] if data.get(k) is not None)}/5
    - **Batch Analysis:** {sum(1 for k in ['stock_corr','stock_summary','gdp_rankings','word_freq'] if data.get(k) is not None)}/4
    - **Economics:** {sum(1 for k in ['correlation_matrix','regression','decade','merged_assets'] if data.get(k) is not None)}/4
    """)

    # ==========================================================================
    # OVERVIEW PAGE
    # ==========================================================================
    if "Overview" in category:
        st.header("📊 All Rankings Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 Top Words (Composite)")
            word_ranking = create_word_master_ranking(data)
            if word_ranking is not None:
                st.dataframe(
                    word_ranking[['word', 'composite_rank', 'pagerank_score', 'frequency']].head(15),
                    use_container_width=True
                )
            else:
                st.info("Word ranking data not available")

        with col2:
            st.subheader("💹 Top Assets")
            asset_ranking = create_asset_ranking(data)
            if asset_ranking is not None:
                cols_to_show = ['symbol'] + [c for c in asset_ranking.columns if 'rank' in c.lower() or 'return' in c.lower()][:4]
                cols_to_show = [c for c in cols_to_show if c in asset_ranking.columns]
                st.dataframe(
                    asset_ranking[cols_to_show].head(15),
                    use_container_width=True
                )
            else:
                st.info("Asset ranking data not available")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🎯 Top Topics")
            topic_ranking = create_topic_ranking(data)
            if topic_ranking is not None:
                st.dataframe(topic_ranking.head(10), use_container_width=True)
            else:
                st.info("Topic ranking data not available")

        with col4:
            st.subheader("📚 Top N-grams")
            ngram_ranking = create_ngram_ranking(data)
            if ngram_ranking is not None:
                st.dataframe(
                    ngram_ranking[['ngram', 'total_count', 'n']].head(15),
                    use_container_width=True
                )
            else:
                st.info("N-gram ranking data not available")

        # Summary metrics
        st.markdown("---")
        st.subheader("📈 Key Metrics Summary")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            if word_ranking is not None:
                st.metric("Total Words Ranked", len(word_ranking))
        with m2:
            if asset_ranking is not None:
                st.metric("Assets Analyzed", len(asset_ranking))
        with m3:
            if topic_ranking is not None:
                st.metric("Topics Discovered", len(topic_ranking))
        with m4:
            if ngram_ranking is not None:
                st.metric("Phrases Found", len(ngram_ranking))

    # ==========================================================================
    # WORD MASTER RANKING
    # ==========================================================================
    elif "Word Master" in category:
        st.header("📝 Word Master Ranking")
        st.markdown("""
        **Unified word ranking** combining:
        - PageRank (influence in co-occurrence network)
        - TF-IDF (importance in documents)
        - Frequency (raw occurrence count)
        """)

        word_ranking = create_word_master_ranking(data)

        if word_ranking is not None:
            # Top N selector
            top_n = st.slider("Show Top N Words", 10, 100, 30)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Bar chart
                if 'pagerank_score' in word_ranking.columns:
                    fig = plot_ranking_bar(
                        word_ranking, 'word', 'pagerank_score',
                        f"Top {top_n} Words by PageRank", top_n=top_n
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Treemap
                if 'frequency' in word_ranking.columns:
                    fig = plot_ranking_treemap(
                        word_ranking, 'word', 'frequency',
                        "Word Frequency Treemap", top_n=top_n
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Full table
            st.subheader("📋 Complete Word Rankings Table")
            st.dataframe(word_ranking.head(top_n), use_container_width=True)

            # Download button
            csv = word_ranking.to_csv(index=False)
            st.download_button(
                "📥 Download Word Rankings CSV",
                csv,
                "word_master_ranking.csv",
                "text/csv"
            )
        else:
            st.warning("No word ranking data available")

    # ==========================================================================
    # ASSET/STOCK RANKING
    # ==========================================================================
    elif "Asset" in category:
        st.header("💹 Asset/Stock Ranking")

        asset_ranking = create_asset_ranking(data)

        if asset_ranking is not None:
            # Metrics
            if 'symbol' in asset_ranking.columns:
                st.subheader("🏆 Top Performing Assets")

                col1, col2 = st.columns(2)

                with col1:
                    # Return ranking
                    if 'avg_return' in asset_ranking.columns:
                        fig = px.bar(
                            asset_ranking.head(15),
                            x='symbol', y='avg_return',
                            title="Average Return by Asset",
                            color='avg_return',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Volatility ranking
                    if 'volatility' in asset_ranking.columns:
                        fig = px.bar(
                            asset_ranking.head(15),
                            x='symbol', y='volatility',
                            title="Volatility by Asset (Lower = Safer)",
                            color='volatility',
                            color_continuous_scale='Reds_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Entity-price impact
                if 'same_day_corr' in asset_ranking.columns:
                    st.subheader("📰 News Mention Impact on Price")
                    fig = px.bar(
                        asset_ranking,
                        x='symbol', y='same_day_corr',
                        title="Correlation: News Mentions → Same-Day Price Change",
                        color='same_day_corr',
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Full table
            st.subheader("📋 Complete Asset Rankings")
            st.dataframe(asset_ranking, use_container_width=True)
        else:
            st.warning("No asset ranking data available")

    # ==========================================================================
    # TOPIC RANKING
    # ==========================================================================
    elif "Topic" in category:
        st.header("🎯 Topic Ranking (LDA)")

        topic_ranking = create_topic_ranking(data)

        if topic_ranking is not None:
            col1, col2 = st.columns(2)

            with col1:
                if 'count' in topic_ranking.columns:
                    fig = px.pie(
                        topic_ranking,
                        values='count',
                        names='dominant_topic' if 'dominant_topic' in topic_ranking.columns else topic_ranking.index,
                        title="Topic Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'count' in topic_ranking.columns:
                    fig = px.bar(
                        topic_ranking,
                        x='dominant_topic' if 'dominant_topic' in topic_ranking.columns else topic_ranking.index,
                        y='count',
                        title="Documents per Topic",
                        color='count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Topic details
            if data['lda_topics'] is not None:
                st.subheader("📝 Top Words per Topic")
                topics = data['lda_topics']

                for topic_id in topics['topic_id'].unique():
                    topic_words = topics[topics['topic_id'] == topic_id].head(10)
                    with st.expander(f"Topic {topic_id}", expanded=False):
                        st.dataframe(topic_words[['word', 'weight']], use_container_width=True)
        else:
            st.warning("No topic ranking data available")

    # ==========================================================================
    # CORRELATION RANKING
    # ==========================================================================
    elif "Correlation" in category:
        st.header("🔗 Correlation Ranking")

        corr_ranking = create_correlation_ranking(data)

        if corr_ranking is not None:
            st.subheader("🔝 Strongest Correlations")

            # Filter for strong correlations
            if 'correlation' in corr_ranking.columns:
                strong = corr_ranking[corr_ranking['correlation'].abs() > 0.3].copy()
                strong = strong.sort_values('correlation', key=abs, ascending=False)

                fig = px.bar(
                    strong.head(20),
                    y=strong.columns[0] if 'asset1' not in strong.columns else 'asset1',
                    x='correlation',
                    orientation='h',
                    title="Top 20 Correlations",
                    color='correlation',
                    color_continuous_scale='RdBu'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Correlation matrix heatmap
            if data['correlation_matrix'] is not None:
                st.subheader("🗺️ Full Correlation Matrix")
                cm = data['correlation_matrix']
                cm_values = cm.set_index(cm.columns[0])

                fig = px.imshow(
                    cm_values,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title="Multi-Asset Correlation Matrix"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(corr_ranking.head(30), use_container_width=True)
        else:
            st.warning("No correlation data available")

    # ==========================================================================
    # COUNTRY/ECONOMY RANKING
    # ==========================================================================
    elif "Country" in category:
        st.header("🌍 Country/Economy Ranking")

        country_ranking = create_country_ranking(data)

        if country_ranking is not None:
            col1, col2 = st.columns(2)

            with col1:
                if 'gdp' in country_ranking.columns:
                    fig = px.bar(
                        country_ranking,
                        x='country',
                        y='gdp',
                        title="GDP Comparison",
                        color='gdp',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'growth' in country_ranking.columns:
                    fig = px.bar(
                        country_ranking,
                        x='country',
                        y='growth',
                        title="GDP Growth Rate (%)",
                        color='growth',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(country_ranking, use_container_width=True)

        # Korea-USA time series
        if data['korea_usa'] is not None:
            st.subheader("📈 Korea vs USA GDP Over Time")
            kr_usa = data['korea_usa']

            if 'year' in kr_usa.columns:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("GDP (USD)", "Growth Rate (%)"))

                # GDP
                if 'usa_gdp' in kr_usa.columns:
                    fig.add_trace(
                        go.Scatter(x=kr_usa['year'], y=kr_usa['usa_gdp'], name='USA GDP', line=dict(color='blue')),
                        row=1, col=1
                    )
                if 'korea_gdp' in kr_usa.columns:
                    fig.add_trace(
                        go.Scatter(x=kr_usa['year'], y=kr_usa['korea_gdp'], name='Korea GDP', line=dict(color='red')),
                        row=1, col=1
                    )

                # Growth
                if 'usa_growth' in kr_usa.columns:
                    fig.add_trace(
                        go.Scatter(x=kr_usa['year'], y=kr_usa['usa_growth'], name='USA Growth', line=dict(color='blue', dash='dash')),
                        row=1, col=2
                    )
                if 'korea_growth' in kr_usa.columns:
                    fig.add_trace(
                        go.Scatter(x=kr_usa['year'], y=kr_usa['korea_growth'], name='Korea Growth', line=dict(color='red', dash='dash')),
                        row=1, col=2
                    )

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No country ranking data available")

    # ==========================================================================
    # N-GRAM RANKING
    # ==========================================================================
    elif "N-gram" in category:
        st.header("📚 N-gram Phrase Ranking")

        ngram_ranking = create_ngram_ranking(data)

        if ngram_ranking is not None:
            # Filter by N
            n_values = ngram_ranking['n'].unique() if 'n' in ngram_ranking.columns else [2]
            selected_n = st.multiselect("Filter by N-gram size", sorted(n_values), default=list(n_values))

            filtered = ngram_ranking[ngram_ranking['n'].isin(selected_n)] if 'n' in ngram_ranking.columns else ngram_ranking

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Top Phrases")
                fig = plot_ranking_bar(
                    filtered, 'ngram', 'total_count',
                    "Top N-grams by Frequency", top_n=25
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Phrase Cloud")
                fig = plot_ranking_treemap(
                    filtered, 'ngram', 'total_count',
                    "N-gram Treemap", top_n=40
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(filtered.head(50), use_container_width=True)
        else:
            st.warning("No N-gram data available")

    # ==========================================================================
    # RAW DATA EXPLORER
    # ==========================================================================
    elif "Raw Data" in category:
        st.header("📁 Raw Data Explorer")

        st.markdown("Browse all loaded data files:")

        available_data = {k: v for k, v in data.items() if v is not None}

        selected_file = st.selectbox(
            "Select Data File",
            list(available_data.keys())
        )

        if selected_file and available_data.get(selected_file) is not None:
            df = available_data[selected_file]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

            st.subheader(f"📋 {selected_file}")
            st.dataframe(df, use_container_width=True)

            # Column stats
            with st.expander("📊 Column Statistics"):
                st.write(df.describe())

            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                f"📥 Download {selected_file}.csv",
                csv,
                f"{selected_file}.csv",
                "text/csv"
            )


if __name__ == "__main__":
    main()
