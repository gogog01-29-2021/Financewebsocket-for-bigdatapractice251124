#!/usr/bin/env python3
"""
ADVANCED CROSS-SOURCE ANALYTICS DASHBOARD
Visualizes deep analysis results:
- PageRank Word Importance
- Word Co-occurrence Network
- Formal vs Informal Language
- Word-Price Causality
- Social Media vs News Comparison
"""

import streamlit as st

st.set_page_config(
    page_title="Advanced Cross-Source Analytics",
    page_icon="🔬",
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
import networkx as nx

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "batch"
OUTPUT_DIR = Path(__file__).parent / "spark_output"


@st.cache_data(ttl=300)
def load_output(filename):
    """Load analysis output file"""
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        if filename.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return pd.read_csv(file_path)
    return None


def main():
    st.title("🔬 Advanced Cross-Source Analytics Dashboard")

    st.markdown("""
    **Deep Analysis Features:**
    - **PageRank Algorithm** - Google-style word importance ranking
    - **Word Co-occurrence Network** - Graph visualization of word relationships
    - **Formal vs Informal** - News headlines vs Social media comparison
    - **Causality Analysis** - Word → Price and Price → Word relationships
    """)

    # Sidebar
    st.sidebar.title("Analysis Sections")

    st.sidebar.markdown("**Basic Analysis**")
    section = st.sidebar.radio(
        "Select Analysis",
        ["📊 Overview",
         "📁 Raw Data Explorer",
         "🏆 PageRank Word Importance",
         "🕸️ Word Network Graph",
         "📝 Formal vs Informal",
         "📈 Word-Price Causality",
         "💬 Social Media Analysis",
         "🔄 Granger Causality",
         "---Deep Semantic---",
         "📚 N-gram Phrases",
         "🎯 TF-IDF Analysis",
         "🏷️ Topic Modeling (LDA)",
         "🧠 Word2Vec Embeddings",
         "🏢 Entity-Price Link",
         "🇰🇷🇺🇸 Korea vs USA"]
    )

    if section == "📊 Overview":
        show_overview()
    elif section == "📁 Raw Data Explorer":
        show_raw_data()
    elif section == "🏆 PageRank Word Importance":
        show_pagerank()
    elif section == "🕸️ Word Network Graph":
        show_word_network()
    elif section == "📝 Formal vs Informal":
        show_formal_informal()
    elif section == "📈 Word-Price Causality":
        show_word_price()
    elif section == "💬 Social Media Analysis":
        show_social_media()
    elif section == "🔄 Granger Causality":
        show_granger()
    elif section == "📚 N-gram Phrases":
        show_ngrams()
    elif section == "🎯 TF-IDF Analysis":
        show_tfidf()
    elif section == "🏷️ Topic Modeling (LDA)":
        show_lda()
    elif section == "🧠 Word2Vec Embeddings":
        show_word2vec()
    elif section == "🏢 Entity-Price Link":
        show_entity_price()
    elif section == "🇰🇷🇺🇸 Korea vs USA":
        show_korea_usa()


def show_raw_data():
    """Show raw data explorer"""
    st.header("📁 Raw Data Explorer")

    st.markdown("""
    Explore all source data and analysis outputs.
    """)

    tabs = st.tabs(["📈 Stock Data", "📰 News Headlines", "🌍 Economic Data", "📊 Analysis Outputs"])

    with tabs[0]:
        st.subheader("Stock Data (Yahoo Finance)")
        stock_file = DATA_DIR / "stock_data.csv"
        if stock_file.exists():
            stock_df = pd.read_csv(stock_file)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(stock_df):,}")
            with col2:
                st.metric("Symbols", stock_df['symbol'].nunique() if 'symbol' in stock_df.columns else 'N/A')
            with col3:
                if 'date' in stock_df.columns:
                    st.metric("Date Range", f"{stock_df['date'].min()} to {stock_df['date'].max()}")

            # Filter by symbol
            if 'symbol' in stock_df.columns:
                symbols = ['All'] + sorted(stock_df['symbol'].unique().tolist())
                selected_symbol = st.selectbox("Filter by Symbol", symbols)
                if selected_symbol != 'All':
                    stock_df = stock_df[stock_df['symbol'] == selected_symbol]

            st.dataframe(stock_df.head(500), use_container_width=True)
            st.caption(f"Showing first 500 rows of {len(stock_df):,} total")
        else:
            st.warning("Stock data not found.")

    with tabs[1]:
        st.subheader("News Headlines")
        news_file = DATA_DIR / "news_headlines.csv"
        if news_file.exists():
            news_df = pd.read_csv(news_file)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Headlines", f"{len(news_df):,}")
            with col2:
                if 'sentiment' in news_df.columns:
                    st.metric("Positive %", f"{(news_df['sentiment'] == 'positive').mean()*100:.1f}%")
            with col3:
                if 'sentiment' in news_df.columns:
                    st.metric("Negative %", f"{(news_df['sentiment'] == 'negative').mean()*100:.1f}%")

            # Filter by sentiment
            if 'sentiment' in news_df.columns:
                sentiments = ['All'] + sorted(news_df['sentiment'].unique().tolist())
                selected_sentiment = st.selectbox("Filter by Sentiment", sentiments)
                if selected_sentiment != 'All':
                    news_df = news_df[news_df['sentiment'] == selected_sentiment]

            # Search headlines
            search_term = st.text_input("Search Headlines", "")
            if search_term:
                news_df = news_df[news_df['headline'].str.contains(search_term, case=False, na=False)]

            st.dataframe(news_df.head(500), use_container_width=True)
            st.caption(f"Showing first 500 rows of {len(news_df):,} total")
        else:
            st.warning("News data not found.")

    with tabs[2]:
        st.subheader("Economic Indicators")

        # World Bank GDP
        gdp_file = DATA_DIR / "worldbank_gdp.csv"
        if gdp_file.exists():
            gdp_df = pd.read_csv(gdp_file)
            st.write("**World Bank GDP Data**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", f"{len(gdp_df):,}")
            with col2:
                if 'country' in gdp_df.columns:
                    st.metric("Countries", gdp_df['country'].nunique())

            st.dataframe(gdp_df.head(200), use_container_width=True)

        # Economic indicators
        econ_file = DATA_DIR / "economic_indicators.csv"
        if econ_file.exists():
            econ_df = pd.read_csv(econ_file)
            st.write("**Economic Indicators (FRED-style)**")
            st.dataframe(econ_df.head(200), use_container_width=True)

    with tabs[3]:
        st.subheader("Analysis Output Files")

        output_files = list(OUTPUT_DIR.glob("*.csv")) + list(OUTPUT_DIR.glob("*.json"))

        if not output_files:
            st.warning("No analysis outputs found. Run advanced_spark_cross_analysis.py first.")
            return

        file_info = []
        for f in output_files:
            size_kb = f.stat().st_size / 1024
            file_info.append({
                'File': f.name,
                'Size (KB)': f"{size_kb:.1f}",
                'Type': f.suffix
            })

        st.dataframe(pd.DataFrame(file_info), use_container_width=True)

        # Select file to view
        file_names = [f.name for f in output_files]
        selected_file = st.selectbox("Select file to preview", file_names)

        if selected_file:
            file_path = OUTPUT_DIR / selected_file
            if selected_file.endswith('.csv'):
                df = pd.read_csv(file_path)
                st.write(f"**{selected_file}** ({len(df)} rows)")
                st.dataframe(df, use_container_width=True)
            elif selected_file.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                st.write(f"**{selected_file}**")
                st.json(data)


def show_overview():
    """Show dashboard overview"""
    st.header("Analysis Overview")

    # Check available files
    available_files = list(OUTPUT_DIR.glob("*.csv")) + list(OUTPUT_DIR.glob("*.json"))

    if not available_files:
        st.warning("No analysis output found. Run `python advanced_spark_cross_analysis.py` first.")
        st.code("cd advanced_analytics && python advanced_spark_cross_analysis.py")
        return

    st.success(f"✓ {len(available_files)} analysis files available")

    # Load report
    report = load_output("advanced_analysis_report.json")
    if report:
        st.subheader("Analysis Report")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Generated at:**", report.get('generated_at', 'N/A'))
            st.write("**Engine:**", report.get('engine', 'N/A'))

        with col2:
            st.write("**Analyses Performed:**")
            for analysis in report.get('analyses_performed', []):
                st.write(f"  - {analysis}")

    # Quick stats
    st.subheader("Quick Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        pagerank = load_output("word_pagerank.csv")
        if pagerank is not None:
            st.metric("PageRank Words", len(pagerank))
            st.write("Top 5 by PageRank:")
            for _, row in pagerank.head(5).iterrows():
                st.write(f"  {row['word']}: {row['pagerank']:.4f}")

    with col2:
        edges = load_output("word_graph_edges.csv")
        if edges is not None:
            st.metric("Word Connections", len(edges))
            st.write("Strongest connections:")
            for _, row in edges.head(5).iterrows():
                st.write(f"  {row['source']}↔{row['target']}: {row['weight']}")

    with col3:
        sentiment = load_output("sentiment_by_source.csv")
        if sentiment is not None:
            st.metric("Sources Analyzed", len(sentiment))
            for _, row in sentiment.iterrows():
                st.write(f"  {row['source']}: {row['avg_sentiment']:.3f}")


def show_pagerank():
    """Show PageRank analysis"""
    st.header("🏆 PageRank Word Importance")

    st.markdown("""
    **PageRank Algorithm** (Google's original search ranking algorithm) applied to words:
    - Words are nodes in a graph
    - Co-occurrence in headlines creates edges
    - PageRank finds the most "influential" words based on connections
    """)

    pagerank = load_output("word_pagerank.csv")

    if pagerank is None:
        st.warning("PageRank data not found.")
        return

    # Top words bar chart
    fig = px.bar(
        pagerank.head(30),
        x='pagerank',
        y='word',
        orientation='h',
        title='Top 30 Words by PageRank Score',
        color='pagerank',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # PageRank vs Frequency scatter
    st.subheader("PageRank vs Raw Frequency")

    fig = px.scatter(
        pagerank.head(100),
        x='frequency',
        y='pagerank',
        text='word',
        title='PageRank Score vs Word Frequency',
        labels={'frequency': 'Raw Frequency', 'pagerank': 'PageRank Score'}
    )
    fig.update_traces(textposition='top center', textfont_size=8)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Interpretation:**
    - Words above the trend line have high influence relative to their frequency
    - These words are "hubs" that connect different topics
    - High PageRank + Low Frequency = Specialized but important term
    """)

    # Full table
    st.subheader("Full PageRank Results")
    st.dataframe(pagerank, use_container_width=True)


def show_word_network():
    """Show word co-occurrence network"""
    st.header("🕸️ Word Co-occurrence Network")

    edges = load_output("word_graph_edges.csv")

    if edges is None:
        st.warning("Word network data not found.")
        return

    # Filter controls
    min_weight = st.slider("Minimum connection strength", 1, int(edges['weight'].max()), 10)
    filtered_edges = edges[edges['weight'] >= min_weight]

    st.write(f"Showing {len(filtered_edges)} connections (min weight = {min_weight})")

    # Build network graph
    G = nx.Graph()
    for _, row in filtered_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    if len(G.nodes()) == 0:
        st.warning("No connections at this threshold. Try lowering the minimum weight.")
        return

    # Calculate positions
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create Plotly figure
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_degrees = [G.degree(node) for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[min(d * 3 + 10, 50) for d in node_degrees],
            color=node_degrees,
            colorbar=dict(title='Connections'),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Word Co-occurrence Network',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700
                    ))

    st.plotly_chart(fig, use_container_width=True)

    # Network statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes (Words)", len(G.nodes()))
    with col2:
        st.metric("Edges (Connections)", len(G.edges()))
    with col3:
        if len(G.nodes()) > 0:
            density = nx.density(G)
            st.metric("Network Density", f"{density:.4f}")

    # Top connections table
    st.subheader("Strongest Word Connections")
    st.dataframe(filtered_edges.head(20), use_container_width=True)


def show_formal_informal():
    """Show formal vs informal language comparison"""
    st.header("📝 Formal vs Informal Language Analysis")

    comparison = load_output("word_comparison_sources.csv")
    sentiment = load_output("sentiment_by_source.csv")

    if comparison is None:
        st.warning("Comparison data not found.")
        return

    st.markdown("""
    **Sources Compared:**
    - **News Headlines** - Formal, professional language
    - **YouTube Comments** - Casual, emoji-heavy
    - **Reddit (WSB-style)** - Very informal, meme language
    """)

    # Sentiment comparison
    if sentiment is not None:
        st.subheader("Sentiment by Source")

        fig = go.Figure()

        for _, row in sentiment.iterrows():
            fig.add_trace(go.Bar(
                name=row['source'].title(),
                x=['Positive', 'Negative', 'Neutral'],
                y=[row['positive_pct'], row['negative_pct'], row['neutral_pct']]
            ))

        fig.update_layout(
            barmode='group',
            title='Sentiment Distribution by Source',
            yaxis_title='Percentage (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Word comparison
    st.subheader("Top Words by Source")

    # Filter to show meaningful comparisons
    comparison = comparison.sort_values('total_count', ascending=False).head(50)

    fig = go.Figure()

    fig.add_trace(go.Bar(name='News', x=comparison['word'], y=comparison['news_count'],
                         marker_color='blue'))
    fig.add_trace(go.Bar(name='YouTube', x=comparison['word'], y=comparison['youtube_count'],
                         marker_color='red'))
    fig.add_trace(go.Bar(name='Reddit', x=comparison['word'], y=comparison['reddit_count'],
                         marker_color='orange'))

    fig.update_layout(
        barmode='group',
        title='Word Frequency Comparison Across Sources',
        xaxis_tickangle=-45,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Source-specific words
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("News-Specific Words (Formal)")
        news_specific = comparison[
            (comparison['news_rank'] > 0) &
            (comparison['youtube_rank'] == -1) &
            (comparison['reddit_rank'] == -1)
        ].head(15)
        if not news_specific.empty:
            for _, row in news_specific.iterrows():
                st.write(f"  • {row['word']}: {row['news_count']}")
        else:
            st.write("  No news-specific words found")

    with col2:
        st.subheader("Social Media-Specific Words (Informal)")
        social_specific = comparison[
            (comparison['news_rank'] == -1) &
            ((comparison['youtube_rank'] > 0) | (comparison['reddit_rank'] > 0))
        ].head(15)
        if not social_specific.empty:
            for _, row in social_specific.iterrows():
                st.write(f"  • {row['word']}: YT={row['youtube_count']}, Reddit={row['reddit_count']}")
        else:
            st.write("  No social-specific words found")

    st.dataframe(comparison, use_container_width=True)


def show_word_price():
    """Show word-price causality analysis"""
    st.header("📈 Word-Price Causality Analysis")

    word_before = load_output("word_before_price_lag1.csv")

    if word_before is None:
        st.warning("Word-price data not found.")
        return

    st.markdown("""
    **Analysis:** Which words predict price movements?
    - **Positive bias** = Word appears more before UP days
    - **Negative bias** = Word appears more before DOWN days
    """)

    # Top predictive words
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Words Predicting UP")
        up_words = word_before.nlargest(15, 'predictive_bias')
        fig = px.bar(
            up_words,
            x='predictive_bias',
            y='word',
            orientation='h',
            color='predictive_bias',
            color_continuous_scale='Greens',
            title='Top Words Before UP Days'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔴 Words Predicting DOWN")
        down_words = word_before.nsmallest(15, 'predictive_bias')
        fig = px.bar(
            down_words,
            x='predictive_bias',
            y='word',
            orientation='h',
            color='predictive_bias',
            color_continuous_scale='Reds_r',
            title='Top Words Before DOWN Days'
        )
        fig.update_layout(yaxis={'categoryorder': 'total descending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Distribution of predictive bias
    st.subheader("Distribution of Predictive Bias")
    fig = px.histogram(
        word_before,
        x='predictive_bias',
        nbins=50,
        title='Distribution of Word Predictive Bias',
        labels={'predictive_bias': 'Predictive Bias (positive = predicts UP)'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.subheader("All Word-Price Predictions")
    st.dataframe(word_before.sort_values('predictive_bias', key=abs, ascending=False),
                 use_container_width=True)


def show_social_media():
    """Show social media analysis"""
    st.header("💬 Social Media Analysis")

    youtube = load_output("youtube_comments.csv")
    reddit = load_output("reddit_comments.csv")

    if youtube is None and reddit is None:
        st.warning("Social media data not found.")
        return

    st.markdown("""
    **Synthetic Social Media Data** for analysis comparison:
    - YouTube finance video comments
    - Reddit (WSB-style) posts
    """)

    tabs = st.tabs(["YouTube", "Reddit"])

    with tabs[0]:
        if youtube is not None:
            st.subheader("YouTube Comments Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Comments", len(youtube))
            with col2:
                st.metric("Avg Sentiment", f"{youtube['sentiment_score'].mean():.3f}")

            # Sentiment distribution
            sentiment_counts = youtube['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='YouTube Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stock mentions
            stock_mentions = youtube['stock_mentioned'].value_counts()
            fig = px.bar(
                x=stock_mentions.index,
                y=stock_mentions.values,
                title='Stock Mentions in YouTube Comments'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sample comments
            st.subheader("Sample Comments")
            sample = youtube.sample(min(10, len(youtube)))
            for _, row in sample.iterrows():
                if row['sentiment'] == 'positive':
                    st.success(row['text'])
                elif row['sentiment'] == 'negative':
                    st.error(row['text'])
                else:
                    st.info(row['text'])

    with tabs[1]:
        if reddit is not None:
            st.subheader("Reddit (WSB-style) Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Posts", len(reddit))
            with col2:
                st.metric("Avg Sentiment", f"{reddit['sentiment_score'].mean():.3f}")

            # Sentiment distribution
            sentiment_counts = reddit['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Reddit Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stock mentions
            stock_mentions = reddit['stock_mentioned'].value_counts()
            fig = px.bar(
                x=stock_mentions.index,
                y=stock_mentions.values,
                title='Stock Mentions in Reddit Posts'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Sample posts
            st.subheader("Sample Posts")
            sample = reddit.sample(min(10, len(reddit)))
            for _, row in sample.iterrows():
                if row['sentiment'] == 'positive':
                    st.success(row['text'])
                elif row['sentiment'] == 'negative':
                    st.error(row['text'])
                else:
                    st.info(row['text'])


def show_granger():
    """Show Granger causality analysis"""
    st.header("🔄 Granger Causality Analysis")

    granger = load_output("granger_correlations.csv")

    if granger is None:
        st.warning("Granger causality data not found.")
        return

    st.markdown("""
    **Granger Causality** tests whether past values of one variable help predict another:
    - **Sentiment → Return**: Does yesterday's sentiment predict today's return?
    - **Return → Sentiment**: Does yesterday's return predict today's sentiment?
    """)

    # Split by direction
    s2r = granger[granger['direction'] == 'sentiment_to_return']
    r2s = granger[granger['direction'] == 'return_to_sentiment']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment → Return")
        fig = px.bar(
            s2r,
            x='lag',
            y='correlation',
            title='Correlation: Past Sentiment vs Future Return',
            labels={'lag': 'Lag (days)', 'correlation': 'Correlation'},
            color='correlation',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        max_corr = s2r.loc[s2r['correlation'].abs().idxmax()]
        st.info(f"Strongest signal at lag {int(max_corr['lag'])}: r = {max_corr['correlation']:.4f}")

    with col2:
        st.subheader("Return → Sentiment")
        fig = px.bar(
            r2s,
            x='lag',
            y='correlation',
            title='Correlation: Past Return vs Future Sentiment',
            labels={'lag': 'Lag (days)', 'correlation': 'Correlation'},
            color='correlation',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        max_corr = r2s.loc[r2s['correlation'].abs().idxmax()]
        st.info(f"Strongest signal at lag {int(max_corr['lag'])}: r = {max_corr['correlation']:.4f}")

    # Interpretation
    st.subheader("Interpretation")

    st.markdown("""
    **Key Questions Answered:**

    1. **Does news sentiment predict stock returns?**
       - Look at "Sentiment → Return" chart
       - Positive correlation at lag 1 = bullish news predicts up days

    2. **Do stock returns influence news sentiment?**
       - Look at "Return → Sentiment" chart
       - Positive correlation = good market days lead to more positive news

    3. **Which direction is stronger?**
       - Compare the magnitude of correlations
       - Tells us about market efficiency and news reactivity
    """)

    # Full table
    st.subheader("Full Correlation Data")
    st.dataframe(granger, use_container_width=True)


# ============================================================
# DEEP SEMANTIC ANALYSIS SECTIONS
# ============================================================

DEEP_OUTPUT_DIR = Path(__file__).parent / "deep_semantic_output"


@st.cache_data(ttl=300)
def load_deep_output(filename):
    """Load deep semantic analysis output"""
    file_path = DEEP_OUTPUT_DIR / filename
    if file_path.exists():
        if filename.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return pd.read_csv(file_path)
    return None


def show_ngrams():
    """Show N-gram phrase analysis"""
    st.header("📚 N-gram (Phrase) Analysis")

    st.markdown("""
    **N-grams** are sequences of N consecutive words:
    - **Bigrams (2-grams)**: "interest rate", "stock market"
    - **Trigrams (3-grams)**: "federal reserve bank", "quarterly earnings report"

    These reveal meaningful phrases that single words miss.
    """)

    ngrams = load_deep_output("ngram_analysis.csv")

    if ngrams is None:
        st.warning("N-gram data not found. Run `python deep_semantic_analysis.py` first.")
        st.code("cd advanced_analytics && python deep_semantic_analysis.py")
        return

    # Filter by n-gram type
    n_values = ngrams['n'].unique()
    selected_n = st.selectbox("N-gram Type", sorted(n_values), format_func=lambda x: f"{x}-gram")

    filtered = ngrams[ngrams['n'] == selected_n].head(50)

    # Bar chart
    fig = px.bar(
        filtered,
        x='total_count',
        y='ngram',
        orientation='h',
        title=f'Top {selected_n}-grams by Frequency',
        color='total_count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment breakdown
    st.subheader("Sentiment Breakdown")

    if 'positive' in filtered.columns and 'negative' in filtered.columns:
        fig = go.Figure()

        top_ngrams = filtered.head(20)

        fig.add_trace(go.Bar(name='Positive', x=top_ngrams['ngram'], y=top_ngrams['positive'].fillna(0),
                             marker_color='green'))
        fig.add_trace(go.Bar(name='Negative', x=top_ngrams['ngram'], y=top_ngrams['negative'].fillna(0),
                             marker_color='red'))
        fig.add_trace(go.Bar(name='Neutral', x=top_ngrams['ngram'], y=top_ngrams['neutral'].fillna(0),
                             marker_color='gray'))

        fig.update_layout(barmode='stack', xaxis_tickangle=-45, height=500,
                          title='N-gram Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered, use_container_width=True)


def show_tfidf():
    """Show TF-IDF analysis"""
    st.header("🎯 TF-IDF Analysis")

    st.markdown("""
    **TF-IDF (Term Frequency - Inverse Document Frequency)** finds words that are:
    - Common in specific categories (high TF)
    - Rare across all categories (high IDF)

    High TF-IDF = distinctive word for that category.
    """)

    tfidf = load_deep_output("tfidf_by_sentiment.csv")

    if tfidf is None:
        st.warning("TF-IDF data not found. Run `python deep_semantic_analysis.py` first.")
        return

    # Filter by sentiment
    sentiments = tfidf['sentiment'].unique()

    col1, col2, col3 = st.columns(3)

    for i, sentiment in enumerate(sentiments[:3]):
        with [col1, col2, col3][i]:
            st.subheader(f"{sentiment.title()}")

            sent_data = tfidf[tfidf['sentiment'] == sentiment].head(15)

            fig = px.bar(
                sent_data,
                x='tfidf_score',
                y='word',
                orientation='h',
                color='tfidf_score',
                color_continuous_scale='Blues' if sentiment == 'neutral' else ('Greens' if sentiment == 'positive' else 'Reds')
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full TF-IDF Results")
    st.dataframe(tfidf, use_container_width=True)


def show_lda():
    """Show LDA topic modeling"""
    st.header("🏷️ Topic Modeling (LDA)")

    st.markdown("""
    **Latent Dirichlet Allocation (LDA)** automatically discovers hidden topics in text:
    - Each topic is a distribution over words
    - Each document belongs to multiple topics
    - Reveals underlying themes in news headlines
    """)

    topics = load_deep_output("lda_topics.csv")
    topic_dist = load_deep_output("lda_topic_distribution.csv")
    topic_sentiment = load_deep_output("lda_topic_sentiment.csv")

    if topics is None:
        st.warning("LDA data not found. Run `python deep_semantic_analysis.py` first.")
        return

    # Topic distribution pie chart
    if topic_dist is not None:
        st.subheader("Topic Distribution")

        fig = px.pie(
            topic_dist,
            values='count',
            names='dominant_topic',
            title='Document Distribution Across Topics'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Topics detail
    st.subheader("Discovered Topics")

    topic_ids = topics['topic_id'].unique()

    for topic_id in sorted(topic_ids):
        topic_data = topics[topics['topic_id'] == topic_id]
        topic_name = topic_data['topic_name'].iloc[0] if 'topic_name' in topic_data.columns else f"Topic {topic_id}"

        with st.expander(f"📌 {topic_name}"):
            fig = px.bar(
                topic_data,
                x='weight',
                y='word',
                orientation='h',
                color='weight',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Topic-sentiment relationship
    if topic_sentiment is not None:
        st.subheader("Topic-Sentiment Relationship")

        pivot = topic_sentiment.pivot_table(
            index='dominant_topic', columns='sentiment', values='count', fill_value=0
        ).reset_index()

        fig = go.Figure()
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in pivot.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=pivot['dominant_topic'],
                    y=pivot[sentiment],
                    marker_color={'positive': 'green', 'negative': 'red', 'neutral': 'gray'}[sentiment]
                ))

        fig.update_layout(barmode='stack', title='Sentiment Distribution per Topic',
                          xaxis_title='Topic', yaxis_title='Document Count')
        st.plotly_chart(fig, use_container_width=True)


def show_word2vec():
    """Show Word2Vec embeddings"""
    st.header("🧠 Word2Vec Embeddings")

    st.markdown("""
    **Word2Vec** learns word meanings from context:
    - Words with similar contexts have similar vectors
    - Enables semantic similarity calculations
    - Example: "stock" is similar to "equity", "share"
    """)

    similarities = load_deep_output("word2vec_similarities.csv")

    if similarities is None:
        st.warning("Word2Vec data not found. Run `python deep_semantic_analysis.py` first.")
        return

    # Similar words for each term
    st.subheader("Similar Words for Key Financial Terms")

    terms = similarities['term'].unique()

    cols = st.columns(3)
    for i, term in enumerate(terms):
        with cols[i % 3]:
            st.write(f"**'{term}'** similar to:")
            term_data = similarities[similarities['term'] == term]
            for _, row in term_data.iterrows():
                similarity_pct = row['similarity'] * 100
                st.write(f"  • {row['similar_word']}: {similarity_pct:.1f}%")
            st.write("")

    # Similarity heatmap
    st.subheader("Word Similarity Network")

    fig = px.scatter(
        similarities,
        x='term',
        y='similar_word',
        size='similarity',
        color='similarity',
        color_continuous_scale='Viridis',
        title='Word Similarity Map'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(similarities, use_container_width=True)


def show_entity_price():
    """Show entity-price correlation"""
    st.header("🏢 Entity-Specific Price Correlation")

    st.markdown("""
    **Entity-Price Analysis** measures how mentions of specific companies affect their stock prices:
    - Same-day correlation: Do mentions coincide with price moves?
    - Next-day correlation: Do mentions predict future prices?
    """)

    entity_corr = load_deep_output("entity_price_correlation.csv")

    if entity_corr is None:
        st.warning("Entity-price data not found. Run `python deep_semantic_analysis.py` first.")
        return

    # Key metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Same-Day Correlation")
        fig = px.bar(
            entity_corr,
            x='same_day_corr',
            y='symbol',
            orientation='h',
            color='same_day_corr',
            color_continuous_scale='RdYlGn',
            title='Mention-Price Same-Day Correlation'
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Next-Day (Predictive) Correlation")
        fig = px.bar(
            entity_corr,
            x='next_day_corr',
            y='symbol',
            orientation='h',
            color='next_day_corr',
            color_continuous_scale='RdYlGn',
            title='Mention Today → Price Tomorrow'
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Mention counts
    st.subheader("Total Mentions by Stock")
    fig = px.bar(
        entity_corr,
        x='symbol',
        y='total_mentions',
        color='total_mentions',
        title='News Mentions per Stock'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(entity_corr, use_container_width=True)


def show_korea_usa():
    """Show Korea vs USA comparison"""
    st.header("🇰🇷🇺🇸 Korea vs USA Economic Comparison")

    st.markdown("""
    **Comparing Two Economies:**
    - GDP growth trends over 50+ years
    - Economic correlation
    - Market performance comparison
    """)

    gdp_comp = load_deep_output("korea_usa_gdp_comparison.csv")
    comparison = load_deep_output("korea_usa_comparison.json")
    usa_market = load_deep_output("usa_market_performance.csv")

    if gdp_comp is None and comparison is None:
        st.warning("Korea-USA comparison data not found. Run `python deep_semantic_analysis.py` first.")
        return

    # Key metrics
    if comparison:
        st.subheader("Key Metrics")

        metrics = comparison.get('key_metrics', {})
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'korea_avg_growth_10yr' in metrics:
                st.metric("Korea Avg Growth (10yr)", f"{metrics['korea_avg_growth_10yr']:.2f}%")
        with col2:
            if 'usa_avg_growth_10yr' in metrics:
                st.metric("USA Avg Growth (10yr)", f"{metrics['usa_avg_growth_10yr']:.2f}%")
        with col3:
            if 'growth_correlation' in metrics:
                st.metric("Growth Correlation", f"{metrics['growth_correlation']:.3f}")

    # GDP over time
    if gdp_comp is not None:
        st.subheader("GDP Over Time")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=gdp_comp['year'], y=gdp_comp['usa_gdp'],
                       name='USA GDP', line=dict(color='blue')),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=gdp_comp['year'], y=gdp_comp['korea_gdp'],
                       name='Korea GDP', line=dict(color='red')),
            secondary_y=True
        )

        fig.update_layout(title='GDP Comparison: USA vs Korea',
                          xaxis_title='Year', height=500)
        fig.update_yaxes(title_text="USA GDP", secondary_y=False)
        fig.update_yaxes(title_text="Korea GDP", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        # GDP Growth comparison
        st.subheader("GDP Growth Rate Comparison")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gdp_comp['year'], y=gdp_comp['usa_growth'],
            name='USA Growth', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=gdp_comp['year'], y=gdp_comp['korea_growth'],
            name='Korea Growth', line=dict(color='red')
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title='GDP Growth Rate (%)', xaxis_title='Year',
                          yaxis_title='Growth Rate (%)', height=400)

        st.plotly_chart(fig, use_container_width=True)

        # Korea/USA ratio over time
        st.subheader("Korea GDP as % of USA GDP")

        fig = px.line(
            gdp_comp,
            x='year',
            y='ratio_korea_usa',
            title='Korea GDP / USA GDP Ratio Over Time'
        )
        fig.update_layout(yaxis_title='Ratio', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # US Market Performance
    if usa_market is not None and comparison and 'us_market' in comparison:
        st.subheader("US Market Performance (SPY)")

        us_metrics = comparison['us_market']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Daily Return", f"{us_metrics.get('avg_daily_return', 0):.4f}%")
        with col2:
            st.metric("Volatility", f"{us_metrics.get('volatility', 0):.4f}%")
        with col3:
            st.metric("Total Return", f"{us_metrics.get('total_return', 0):.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{us_metrics.get('sharpe_approx', 0):.2f}")

    # Summary insights
    st.subheader("Key Insights")

    st.markdown("""
    **Economic Comparison Insights:**

    1. **Size Difference**: USA GDP is significantly larger, but Korea has shown rapid growth
    2. **Growth Patterns**: Both economies show correlated business cycles
    3. **Development Stage**: Korea transitioned from developing to developed economy
    4. **Trade Relationship**: Strong economic ties between the two nations

    **Note**: For deeper Korea-specific analysis, consider adding:
    - KOSPI index data
    - Korean Won (KRW) exchange rates
    - Korea-specific news sentiment
    """)

    if gdp_comp is not None:
        st.dataframe(gdp_comp, use_container_width=True)


if __name__ == "__main__":
    main()
