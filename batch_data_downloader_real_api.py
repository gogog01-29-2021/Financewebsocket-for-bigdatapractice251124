#!/usr/bin/env python3
"""
BATCH DATA DOWNLOADER (REAL API VERSION)
Downloads multiple datasets using REAL APIs:
1. World Bank GDP data (50+ years, multiple countries) - REAL API
2. FRED Economic indicators - REAL API (requires API key)
3. Yahoo Finance historical stock data - REAL API
4. News headlines - REAL API (NewsAPI, requires API key)

API Keys required in .env file:
- FRED_API_KEY: Get from https://fred.stlouisfed.org/docs/api/api_key.html
- NEWS_API_KEY: Get from https://newsapi.org/
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
import time
from dotenv import load_dotenv
from textblob import TextBlob

# Load environment variables
load_dotenv()

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Data directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "batch"
WORLDBANK_DIR = DATA_DIR / "worldbank"
FRED_DIR = DATA_DIR / "fred"
STOCKS_DIR = DATA_DIR / "stocks"
NEWS_DIR = DATA_DIR / "news"

# Create directories
for d in [WORLDBANK_DIR, FRED_DIR, STOCKS_DIR, NEWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class WorldBankDownloader:
    """
    Download World Bank economic indicators

    API Documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
    Base URL: https://api.worldbank.org/v2
    No API key required
    """

    BASE_URL = "https://api.worldbank.org/v2"

    INDICATORS = {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
        "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
        "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
        "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
        "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
        "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
        "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment (% of GDP)",
    }

    COUNTRIES = ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "ITA", "BRA", "CAN",
                 "RUS", "KOR", "AUS", "ESP", "MEX", "IDN", "NLD", "SAU", "TUR", "CHE"]

    def download_all(self):
        """Download all World Bank indicators"""
        print("\n" + "="*60)
        print("DOWNLOADING WORLD BANK DATA (REAL API)")
        print("="*60)
        print(f"API: {self.BASE_URL}")
        print(f"Indicators: {len(self.INDICATORS)}")

        all_data = []

        for indicator_code, indicator_name in self.INDICATORS.items():
            print(f"\nDownloading: {indicator_name}...")

            try:
                url = f"{self.BASE_URL}/country/all/indicator/{indicator_code}"
                params = {
                    "format": "json",
                    "per_page": 20000,
                    "date": "1970:2024"
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if len(data) > 1 and data[1]:
                        for record in data[1]:
                            if record.get("value") is not None:
                                all_data.append({
                                    "country_code": record.get("country", {}).get("id"),
                                    "country_name": record.get("country", {}).get("value"),
                                    "indicator_code": indicator_code,
                                    "indicator_name": indicator_name,
                                    "year": int(record.get("date", 0)),
                                    "value": float(record.get("value", 0))
                                })

                        print(f"  Downloaded {len([r for r in data[1] if r.get('value')])} records")
                    else:
                        print(f"  No data returned")
                else:
                    print(f"  HTTP {response.status_code}")

            except Exception as e:
                print(f"  Error: {e}")

            time.sleep(0.5)

        if all_data:
            df = pd.DataFrame(all_data)
            output_file = WORLDBANK_DIR / "world_bank_indicators.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved {len(df)} records to {output_file}")

            summary = df.groupby(['indicator_name', 'year']).agg({
                'value': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            summary.columns = ['indicator_name', 'year', 'mean', 'std', 'min', 'max', 'count']
            summary.to_csv(WORLDBANK_DIR / "world_bank_summary.csv", index=False)

            return df

        return None


class FREDDownloader:
    """
    Download Federal Reserve Economic Data using REAL FRED API

    API Documentation: https://fred.stlouisfed.org/docs/api/fred/
    Base URL: https://api.stlouisfed.org/fred
    API Key Required: Yes (free registration)
    Get API Key: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    SERIES = {
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real Gross Domestic Product",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
        "FEDFUNDS": "Federal Funds Effective Rate",
        "DGS10": "10-Year Treasury Constant Maturity Rate",
        "SP500": "S&P 500",
        "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
        "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI)",
        "GOLDAMGBD228NLBM": "Gold Fixing Price in London Bullion Market",
        "M2SL": "M2 Money Stock",
        "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average",
        "HOUST": "Housing Starts: Total",
        "INDPRO": "Industrial Production Index",
        "UMCSENT": "University of Michigan: Consumer Sentiment",
    }

    def __init__(self):
        self.api_key = FRED_API_KEY
        if not self.api_key:
            print("WARNING: FRED_API_KEY not found in .env file")
            print("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")

    def download_all(self):
        """Download all FRED series using real API"""
        print("\n" + "="*60)
        print("DOWNLOADING FRED DATA (REAL API)")
        print("="*60)
        print(f"API: {self.BASE_URL}")
        print(f"API Key: {'*' * 20 + self.api_key[-8:] if self.api_key else 'NOT SET'}")
        print(f"Series: {len(self.SERIES)}")

        if not self.api_key:
            print("\nERROR: FRED_API_KEY required. Set it in .env file.")
            print("Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            return None

        all_data = []

        for series_id, series_name in self.SERIES.items():
            print(f"\nDownloading: {series_id} ({series_name})...")

            try:
                # FRED API endpoint for series observations
                url = f"{self.BASE_URL}/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": "1970-01-01",
                    "observation_end": "2024-12-31"
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    observations = data.get("observations", [])

                    for obs in observations:
                        value = obs.get("value")
                        if value and value != ".":  # FRED uses "." for missing values
                            try:
                                all_data.append({
                                    "series_id": series_id,
                                    "series_name": series_name,
                                    "date": obs.get("date"),
                                    "year": int(obs.get("date", "1970")[:4]),
                                    "value": float(value)
                                })
                            except ValueError:
                                continue

                    print(f"  Downloaded {len(observations)} observations")
                else:
                    error_msg = response.json().get("error_message", f"HTTP {response.status_code}")
                    print(f"  Error: {error_msg}")

            except Exception as e:
                print(f"  Error: {e}")

            time.sleep(0.3)  # Rate limiting

        if all_data:
            df = pd.DataFrame(all_data)
            output_file = FRED_DIR / "economic_indicators.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved {len(df)} records to {output_file}")

            # Summary by series
            summary = df.groupby('series_id').agg({
                'value': ['mean', 'std', 'min', 'max', 'count'],
                'date': ['min', 'max']
            }).reset_index()
            summary.columns = ['series_id', 'mean', 'std', 'min', 'max', 'count', 'start_date', 'end_date']
            summary.to_csv(FRED_DIR / "fred_summary.csv", index=False)

            return df

        return None


class StockDataDownloader:
    """
    Download historical stock data using Yahoo Finance (yfinance)

    Library: yfinance (pip install yfinance)
    Documentation: https://pypi.org/project/yfinance/
    No API key required
    """

    SYMBOLS = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "GS": "Goldman Sachs",
        "JNJ": "Johnson & Johnson",
        "XOM": "Exxon Mobil",
        "WMT": "Walmart",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola",
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "DIA": "Dow Jones ETF",
    }

    def download_all(self):
        """Download stock data using yfinance"""
        print("\n" + "="*60)
        print("DOWNLOADING STOCK DATA (YFINANCE)")
        print("="*60)

        try:
            import yfinance as yf
            print("Using yfinance library")

            all_data = []

            for symbol, name in self.SYMBOLS.items():
                print(f"Downloading {symbol} ({name})...")

                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="max")

                    if not hist.empty:
                        hist = hist.reset_index()
                        hist['symbol'] = symbol
                        hist['company_name'] = name

                        hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]

                        all_data.append(hist)
                        print(f"  {len(hist)} records from {hist['date'].min()} to {hist['date'].max()}")
                    else:
                        print(f"  No data")

                except Exception as e:
                    print(f"  Error: {e}")

                time.sleep(0.5)

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                output_file = STOCKS_DIR / "stock_prices.csv"
                df.to_csv(output_file, index=False)
                print(f"\nSaved {len(df)} stock records to {output_file}")
                return df

        except ImportError:
            print("yfinance not installed. Run: pip install yfinance")
            return None

        return None


class NewsAPIDownloader:
    """
    Download real news headlines using NewsAPI

    API Documentation: https://newsapi.org/docs
    Base URL: https://newsapi.org/v2
    API Key Required: Yes (free tier: 100 requests/day)
    Get API Key: https://newsapi.org/register

    Note: Free tier only provides articles from last 30 days
    """

    BASE_URL = "https://newsapi.org/v2"

    # Financial/Business news sources
    SOURCES = [
        "bloomberg", "business-insider", "financial-times",
        "the-wall-street-journal", "fortune", "cnbc"
    ]

    # Keywords to search for financial news
    KEYWORDS = [
        "stock market", "earnings", "federal reserve", "interest rates",
        "inflation", "GDP", "tech stocks", "cryptocurrency", "bitcoin",
        "Apple", "Microsoft", "Tesla", "NVIDIA", "Amazon"
    ]

    def __init__(self):
        self.api_key = NEWS_API_KEY
        if not self.api_key:
            print("WARNING: NEWS_API_KEY not found in .env file")
            print("Get your free API key at: https://newsapi.org/register")

    def download_all(self):
        """Download news articles from NewsAPI"""
        print("\n" + "="*60)
        print("DOWNLOADING NEWS HEADLINES (NEWSAPI - REAL)")
        print("="*60)
        print(f"API: {self.BASE_URL}")
        print(f"API Key: {'*' * 20 + self.api_key[-8:] if self.api_key else 'NOT SET'}")

        if not self.api_key:
            print("\nERROR: NEWS_API_KEY required. Set it in .env file.")
            print("Get your free key at: https://newsapi.org/register")
            return None

        all_articles = []

        # Method 1: Top headlines from business category
        print("\nFetching business headlines...")
        try:
            url = f"{self.BASE_URL}/top-headlines"
            params = {
                "apiKey": self.api_key,
                "category": "business",
                "language": "en",
                "pageSize": 100
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                print(f"  Got {len(articles)} business headlines")

                for article in articles:
                    all_articles.append(self._process_article(article))
            else:
                print(f"  Error: {response.status_code}")

        except Exception as e:
            print(f"  Error: {e}")

        # Method 2: Search for each keyword
        for keyword in self.KEYWORDS[:5]:  # Limit to avoid rate limits
            print(f"\nSearching for: '{keyword}'...")
            try:
                url = f"{self.BASE_URL}/everything"
                params = {
                    "apiKey": self.api_key,
                    "q": keyword,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    print(f"  Got {len(articles)} articles")

                    for article in articles:
                        processed = self._process_article(article)
                        processed['search_keyword'] = keyword
                        all_articles.append(processed)
                elif response.status_code == 426:
                    print("  Upgrade required (free tier limitation)")
                else:
                    error = response.json().get("message", f"HTTP {response.status_code}")
                    print(f"  Error: {error}")

            except Exception as e:
                print(f"  Error: {e}")

            time.sleep(1)  # Rate limiting

        if all_articles:
            df = pd.DataFrame(all_articles)
            df = df.drop_duplicates(subset=['headline'])

            output_file = NEWS_DIR / "financial_news_headlines.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved {len(df)} unique articles to {output_file}")

            # Daily sentiment summary
            if 'date' in df.columns:
                daily = df.groupby('date').agg({
                    'sentiment_score': ['mean', 'std', 'count']
                }).reset_index()
                daily.columns = ['date', 'avg_sentiment', 'sentiment_std', 'headline_count']
                daily.to_csv(NEWS_DIR / "daily_sentiment_summary.csv", index=False)

            return df

        return None

    def _process_article(self, article):
        """Process a single article and add sentiment analysis"""
        title = article.get("title", "")
        description = article.get("description", "")

        # Sentiment analysis using TextBlob
        text_for_analysis = f"{title} {description}"
        try:
            blob = TextBlob(text_for_analysis)
            sentiment_score = blob.sentiment.polarity
            sentiment_label = "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral")
        except:
            sentiment_score = 0.0
            sentiment_label = "neutral"

        # Extract date
        published_at = article.get("publishedAt", "")
        date = published_at[:10] if published_at else ""

        return {
            "date": date,
            "timestamp": published_at,
            "headline": title,
            "description": description,
            "source": article.get("source", {}).get("name", ""),
            "url": article.get("url", ""),
            "author": article.get("author", ""),
            "sentiment": sentiment_label,
            "sentiment_score": round(sentiment_score, 3),
            "category": "Business"
        }


def main():
    """Download all datasets using REAL APIs"""
    print("="*60)
    print("BATCH DATA DOWNLOADER (REAL API VERSION)")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAPI Keys:")
    print(f"  FRED: {'SET' if FRED_API_KEY else 'NOT SET'}")
    print(f"  NewsAPI: {'SET' if NEWS_API_KEY else 'NOT SET'}")

    # 1. Download World Bank data (no API key needed)
    wb = WorldBankDownloader()
    wb_data = wb.download_all()

    # 2. Download FRED data (requires API key)
    fred = FREDDownloader()
    fred_data = fred.download_all()

    # 3. Download Stock data (no API key needed)
    stocks = StockDataDownloader()
    stock_data = stocks.download_all()

    # 4. Download News headlines (requires API key)
    news = NewsAPIDownloader()
    news_data = news.download_all()

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"\nData saved to: {DATA_DIR}")
    print("\nFiles created:")
    for f in DATA_DIR.rglob("*.csv"):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.relative_to(DATA_DIR)}: {size:.2f} MB")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
