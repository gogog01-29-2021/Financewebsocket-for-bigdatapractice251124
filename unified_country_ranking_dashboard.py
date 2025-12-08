#!/usr/bin/env python3
"""
UNIFIED COUNTRY RANKING DASHBOARD
==================================
Single integrated view of all country rankings combining:
- Economic metrics (GDP, Stock returns, Currency)
- Social metrics (YouTube, News, Sentiment)
- Cultural metrics (Word frequency, Company mentions)
- Trend indicators (Momentum, Direction)

Country = Primary Key
All metrics aggregated by country
"""

import streamlit as st

st.set_page_config(
    page_title="Unified Country Ranking",
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
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
RANKING_DIR = BASE_DIR / "organized_data" / "processed_data" / "integrated_ranking"

# Country flags
FLAGS = {
    'USA': '🇺🇸', 'KOREA': '🇰🇷', 'GERMANY': '🇩🇪', 'FRANCE': '🇫🇷',
    'UK': '🇬🇧', 'JAPAN': '🇯🇵', 'CHINA': '🇨🇳', 'RUSSIA': '🇷🇺',
    'INDIA': '🇮🇳', 'TAIWAN': '🇹🇼', 'CANADA': '🇨🇦'
}

@st.cache_data(ttl=300)
def load_ranking_data():
    """Load integrated ranking data"""
    try:
        detailed = pd.read_csv(RANKING_DIR / "country_ranking_detailed.csv")
        summary = pd.read_csv(RANKING_DIR / "country_ranking_summary.csv")
        return detailed, summary
    except Exception as e:
        st.error(f"Could not load ranking data: {e}")
        st.info("Run `python integrated_country_ranking_processor.py` first to generate rankings.")
        return pd.DataFrame(), pd.DataFrame()

def create_ranking_table(df):
    """Create formatted ranking table"""
    if df.empty:
        return None

    display_df = df.copy()
    display_df['flag'] = display_df['country'].map(FLAGS)
    display_df['Country'] = display_df['flag'] + ' ' + display_df['country']

    # Select and rename columns
    cols = {
        'overall_rank': 'Rank',
        'Country': 'Country',
        'composite_score': 'Total Score',
        'economic_score': 'Economic',
        'social_score': 'Social',
        'cultural_score': 'Cultural',
        'trend_score': 'Trend'
    }

    available_cols = [c for c in cols.keys() if c in display_df.columns]
    result = display_df[available_cols].rename(columns=cols)

    return result

def create_radar_chart(df, country):
    """Create radar chart for a specific country"""
    if df.empty:
        return None

    country_data = df[df['country'] == country]
    if country_data.empty:
        return None

    categories = ['Economic', 'Social', 'Cultural', 'Trend']
    values = [
        country_data['economic_score'].iloc[0],
        country_data['social_score'].iloc[0],
        country_data['cultural_score'].iloc[0],
        country_data['trend_score'].iloc[0]
    ]
    values.append(values[0])  # Close the radar

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name=country
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=f"{FLAGS.get(country, '')} {country} Performance Profile"
    )

    return fig

def create_score_comparison(df):
    """Create bar chart comparing all scores"""
    if df.empty:
        return None

    df_sorted = df.sort_values('composite_score', ascending=True)

    fig = go.Figure()

    # Add bars for each category
    fig.add_trace(go.Bar(
        name='Economic',
        y=df_sorted['country'],
        x=df_sorted['economic_score'],
        orientation='h',
        marker_color='#2ecc71'
    ))

    fig.add_trace(go.Bar(
        name='Social',
        y=df_sorted['country'],
        x=df_sorted['social_score'],
        orientation='h',
        marker_color='#3498db'
    ))

    fig.add_trace(go.Bar(
        name='Cultural',
        y=df_sorted['country'],
        x=df_sorted['cultural_score'],
        orientation='h',
        marker_color='#9b59b6'
    ))

    fig.add_trace(go.Bar(
        name='Trend',
        y=df_sorted['country'],
        x=df_sorted['trend_score'],
        orientation='h',
        marker_color='#e74c3c'
    ))

    fig.update_layout(
        barmode='group',
        title="Country Score Comparison by Category",
        xaxis_title="Score (0-100)",
        yaxis_title="Country",
        height=500
    )

    return fig

def create_heatmap(df):
    """Create heatmap of all metrics"""
    if df.empty:
        return None

    # Select numeric columns for heatmap
    metric_cols = [
        'stock_return_1y', 'company_avg_return', 'currency_change_1y',
        'news_articles', 'youtube_engagement', 'social_sentiment',
        'word_pagerank_score', 'word_graph_edges',
        'trend_30d', 'momentum'
    ]

    available_cols = [c for c in metric_cols if c in df.columns]
    if not available_cols:
        return None

    heatmap_data = df.set_index('country')[available_cols]

    # Normalize for display
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    heatmap_normalized = heatmap_normalized.fillna(0)

    fig = px.imshow(
        heatmap_normalized.T,
        labels=dict(x="Country", y="Metric", color="Normalized Value"),
        x=heatmap_normalized.index,
        y=available_cols,
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )

    fig.update_layout(
        title="Country Metrics Heatmap (Normalized)",
        height=400
    )

    return fig

def create_trend_gauge(trend_direction, momentum):
    """Create gauge chart for trend"""
    color = 'green' if momentum > 0 else 'red' if momentum < 0 else 'gray'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=momentum,
        title={'text': f"Momentum ({trend_direction})"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-50, 50]},
            'bar': {'color': color},
            'steps': [
                {'range': [-50, -10], 'color': '#ffcccc'},
                {'range': [-10, 10], 'color': '#ffffcc'},
                {'range': [10, 50], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': momentum
            }
        }
    ))

    fig.update_layout(height=250)
    return fig

def main():
    st.title("🏆 Unified Country Ranking Dashboard")
    st.markdown("**Integrated ranking combining Economic, Social, Cultural, and Trend metrics**")

    # Load data
    detailed_df, summary_df = load_ranking_data()

    if detailed_df.empty:
        st.warning("No ranking data available. Please run the processor first.")
        st.code("python integrated_country_ranking_processor.py")
        return

    # Sidebar
    st.sidebar.title("Ranking Controls")

    # Weight adjustment
    st.sidebar.markdown("### Score Weights")
    w_econ = st.sidebar.slider("Economic Weight", 0, 100, 40)
    w_social = st.sidebar.slider("Social Weight", 0, 100, 30)
    w_cultural = st.sidebar.slider("Cultural Weight", 0, 100, 20)
    w_trend = st.sidebar.slider("Trend Weight", 0, 100, 10)

    total_weight = w_econ + w_social + w_cultural + w_trend
    if total_weight > 0:
        # Recalculate composite score with new weights
        detailed_df['custom_composite'] = (
            detailed_df['economic_score'] * (w_econ / total_weight) +
            detailed_df['social_score'] * (w_social / total_weight) +
            detailed_df['cultural_score'] * (w_cultural / total_weight) +
            detailed_df['trend_score'] * (w_trend / total_weight)
        )
        detailed_df['custom_rank'] = detailed_df['custom_composite'].rank(ascending=False).astype(int)
        detailed_df = detailed_df.sort_values('custom_rank')

    # Main content
    tabs = st.tabs(["🏆 Rankings", "📊 Detailed Analysis", "🔍 Country Deep Dive", "📈 Raw Data"])

    # TAB 1: Rankings
    with tabs[0]:
        st.header("Country Rankings")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Top 3 podium
            st.markdown("### 🥇🥈🥉 Top 3 Countries")
            top3 = detailed_df.head(3)
            cols = st.columns(3)
            medals = ['🥇', '🥈', '🥉']

            for i, (_, row) in enumerate(top3.iterrows()):
                with cols[i]:
                    flag = FLAGS.get(row['country'], '')
                    score = row.get('custom_composite', row['composite_score'])
                    st.metric(
                        f"{medals[i]} {flag} {row['country']}",
                        f"{score:.1f}",
                        f"Rank #{i+1}"
                    )

        with col2:
            # Quick stats
            st.markdown("### Quick Stats")
            st.metric("Countries Ranked", len(detailed_df))
            st.metric("Top Economic", detailed_df.loc[detailed_df['economic_score'].idxmax(), 'country'])
            st.metric("Top Social", detailed_df.loc[detailed_df['social_score'].idxmax(), 'country'])

        # Full ranking table
        st.markdown("### Full Rankings")
        ranking_table = create_ranking_table(detailed_df)
        if ranking_table is not None:
            st.dataframe(ranking_table, hide_index=True, use_container_width=True)

        # Score comparison chart
        fig = create_score_comparison(detailed_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Detailed Analysis
    with tabs[1]:
        st.header("Detailed Score Analysis")

        # Heatmap
        fig = create_heatmap(detailed_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Individual category rankings
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Economic Rankings")
            econ_rank = detailed_df[['country', 'economic_score', 'stock_return_1y', 'company_avg_return']].sort_values('economic_score', ascending=False)
            for i, (_, row) in enumerate(econ_rank.iterrows()):
                flag = FLAGS.get(row['country'], '')
                st.markdown(f"**{i+1}. {flag} {row['country']}**: {row['economic_score']:.1f}")

        with col2:
            st.markdown("### Social Rankings")
            social_rank = detailed_df[['country', 'social_score', 'youtube_engagement', 'news_articles']].sort_values('social_score', ascending=False)
            for i, (_, row) in enumerate(social_rank.iterrows()):
                flag = FLAGS.get(row['country'], '')
                st.markdown(f"**{i+1}. {flag} {row['country']}**: {row['social_score']:.1f}")

        # Trend analysis
        st.markdown("### Trend Direction by Country")
        if 'trend_direction' in detailed_df.columns:
            trend_cols = st.columns(4)
            trend_groups = detailed_df.groupby('trend_direction')['country'].apply(list).to_dict()

            trend_icons = {'STRONG_UP': '📈⬆️', 'UP': '📈', 'NEUTRAL': '➡️', 'DOWN': '📉', 'STRONG_DOWN': '📉⬇️'}

            for i, (trend, icon) in enumerate(trend_icons.items()):
                with trend_cols[i % 4]:
                    countries = trend_groups.get(trend, [])
                    st.markdown(f"**{icon} {trend}**")
                    for c in countries:
                        st.markdown(f"- {FLAGS.get(c, '')} {c}")

    # TAB 3: Country Deep Dive
    with tabs[2]:
        st.header("Country Deep Dive")

        selected_country = st.selectbox(
            "Select Country",
            options=detailed_df['country'].tolist(),
            format_func=lambda x: f"{FLAGS.get(x, '')} {x}"
        )

        country_data = detailed_df[detailed_df['country'] == selected_country].iloc[0]

        col1, col2 = st.columns([1, 1])

        with col1:
            # Radar chart
            fig = create_radar_chart(detailed_df, selected_country)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Key metrics
            st.markdown(f"### {FLAGS.get(selected_country, '')} {selected_country}")

            rank = int(country_data.get('custom_rank', country_data.get('overall_rank', 0)))
            score = country_data.get('custom_composite', country_data.get('composite_score', 0))

            st.metric("Overall Rank", f"#{rank}")
            st.metric("Composite Score", f"{score:.1f}")

            st.markdown("---")

            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Economic", f"{country_data['economic_score']:.1f}")
                st.metric("Social", f"{country_data['social_score']:.1f}")
            with col2b:
                st.metric("Cultural", f"{country_data['cultural_score']:.1f}")
                st.metric("Trend", f"{country_data['trend_score']:.1f}")

        # Trend gauge
        if 'trend_direction' in country_data and 'momentum' in country_data:
            st.markdown("### Market Momentum")
            fig = create_trend_gauge(
                country_data.get('trend_direction', 'NEUTRAL'),
                country_data.get('momentum', 0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics
        st.markdown("### All Metrics")
        metrics_df = pd.DataFrame({
            'Metric': country_data.index,
            'Value': country_data.values
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # TAB 4: Raw Data
    with tabs[3]:
        st.header("Raw Ranking Data")

        st.markdown("### Detailed Data (49 columns)")
        st.dataframe(detailed_df, use_container_width=True)

        st.markdown("### Download Data")
        col1, col2 = st.columns(2)

        with col1:
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                "Download Detailed CSV",
                csv,
                "country_ranking_detailed.csv",
                "text/csv"
            )

        with col2:
            if not summary_df.empty:
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Summary CSV",
                    csv_summary,
                    "country_ranking_summary.csv",
                    "text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown(f"""
    **Data Sources:** Stock indices, Company stocks, Exchange rates, News, YouTube, Word analytics

    **Weights:** Economic ({w_econ}%), Social ({w_social}%), Cultural ({w_cultural}%), Trend ({w_trend}%)

    **Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """)

if __name__ == "__main__":
    main()
