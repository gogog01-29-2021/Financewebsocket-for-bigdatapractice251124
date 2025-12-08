"""
Comprehensive Country-Focused Data Downloader
Downloads REAL data for 11 countries: USA, Korea, Germany, France, UK, Japan, China, Russia, India, Taiwan, Canada

Data Sources:
- Yahoo Finance: Stock indices, company stocks, exchange rates, commodities
- NewsAPI: Country-specific news
- FRED: Economic indicators
- World Bank: GDP, population, economic data
- FMP API: Company financials
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import time

load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
FMP_API_KEY = os.getenv('FMP_API_KEY')

# Output directory
OUTPUT_DIR = 'organized_data/raw_data/country_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# COUNTRY CONFIGURATION - 11 Countries
# ============================================================================

COUNTRIES = {
    'USA': {
        'name': 'United States',
        'currency': 'USD',
        'index': '^GSPC',  # S&P 500
        'index_name': 'S&P 500',
        'other_indices': ['^DJI', '^IXIC', '^RUT'],  # Dow, NASDAQ, Russell
        'companies': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ',
                      'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'XOM', 'PFE', 'KO',
                      'CSCO', 'PEP', 'ABBV', 'TMO', 'COST', 'AVGO', 'MRK', 'CVX', 'ACN', 'LLY'],
        'fx': 'DX-Y.NYB',  # US Dollar Index
        'news_keywords': ['usa', 'america', 'us economy', 'federal reserve', 'wall street'],
        'fred_series': ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'SP500']
    },
    'KOREA': {
        'name': 'South Korea',
        'currency': 'KRW',
        'index': '^KS11',  # KOSPI
        'index_name': 'KOSPI',
        'other_indices': ['^KQ11'],  # KOSDAQ
        'companies': ['005930.KS', '000660.KS', '005380.KS', '035420.KS', '035720.KS',
                      '051910.KS', '006400.KS', '068270.KS', '028260.KS', '003670.KS'],
        'fx': 'USDKRW=X',
        'news_keywords': ['korea', 'korean economy', 'samsung', 'kospi', 'seoul'],
        'fred_series': []
    },
    'GERMANY': {
        'name': 'Germany',
        'currency': 'EUR',
        'index': '^GDAXI',  # DAX
        'index_name': 'DAX',
        'other_indices': [],
        'companies': ['SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAS.DE',
                      'BAYN.DE', 'BMW.DE', 'MBG.DE', 'VOW3.DE', 'ADS.DE'],
        'fx': 'EURUSD=X',
        'news_keywords': ['germany', 'german economy', 'dax', 'berlin', 'bundesbank'],
        'fred_series': []
    },
    'FRANCE': {
        'name': 'France',
        'currency': 'EUR',
        'index': '^FCHI',  # CAC 40
        'index_name': 'CAC 40',
        'other_indices': [],
        'companies': ['OR.PA', 'MC.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA',
                      'BNP.PA', 'AI.PA', 'CS.PA', 'DG.PA', 'SU.PA'],
        'fx': 'EURUSD=X',
        'news_keywords': ['france', 'french economy', 'cac 40', 'paris', 'ecb'],
        'fred_series': []
    },
    'UK': {
        'name': 'United Kingdom',
        'currency': 'GBP',
        'index': '^FTSE',  # FTSE 100
        'index_name': 'FTSE 100',
        'other_indices': [],
        'companies': ['SHEL.L', 'AZN.L', 'HSBA.L', 'BP.L', 'GSK.L',
                      'ULVR.L', 'RIO.L', 'DGE.L', 'BATS.L', 'LLOY.L'],
        'fx': 'GBPUSD=X',
        'news_keywords': ['uk', 'britain', 'british economy', 'ftse', 'london', 'bank of england'],
        'fred_series': []
    },
    'JAPAN': {
        'name': 'Japan',
        'currency': 'JPY',
        'index': '^N225',  # Nikkei 225
        'index_name': 'Nikkei 225',
        'other_indices': ['^TOPX'],  # TOPIX
        'companies': ['7203.T', '6758.T', '9984.T', '6861.T', '8306.T',
                      '9432.T', '6902.T', '4502.T', '6501.T', '8035.T'],
        'fx': 'USDJPY=X',
        'news_keywords': ['japan', 'japanese economy', 'nikkei', 'tokyo', 'bank of japan', 'yen'],
        'fred_series': []
    },
    'CHINA': {
        'name': 'China',
        'currency': 'CNY',
        'index': '000001.SS',  # Shanghai Composite
        'index_name': 'Shanghai Composite',
        'other_indices': ['399001.SZ', '^HSI'],  # Shenzhen, Hang Seng
        'companies': ['601318.SS', '600519.SS', '601398.SS', '600036.SS', '601988.SS',
                      '600276.SS', '601166.SS', '600900.SS', '601668.SS', '600028.SS'],
        'fx': 'USDCNY=X',
        'news_keywords': ['china', 'chinese economy', 'shanghai', 'beijing', 'pboc', 'yuan'],
        'fred_series': []
    },
    'RUSSIA': {
        'name': 'Russia',
        'currency': 'RUB',
        'index': 'IMOEX.ME',  # MOEX Russia Index
        'index_name': 'MOEX',
        'other_indices': [],
        'companies': ['SBER.ME', 'GAZP.ME', 'LKOH.ME', 'GMKN.ME', 'NVTK.ME'],
        'fx': 'USDRUB=X',
        'news_keywords': ['russia', 'russian economy', 'moscow', 'ruble', 'central bank of russia'],
        'fred_series': []
    },
    'INDIA': {
        'name': 'India',
        'currency': 'INR',
        'index': '^BSESN',  # BSE Sensex
        'index_name': 'BSE Sensex',
        'other_indices': ['^NSEI'],  # Nifty 50
        'companies': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                      'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
        'fx': 'USDINR=X',
        'news_keywords': ['india', 'indian economy', 'sensex', 'mumbai', 'rbi', 'rupee'],
        'fred_series': []
    },
    'TAIWAN': {
        'name': 'Taiwan',
        'currency': 'TWD',
        'index': '^TWII',  # Taiwan Weighted Index
        'index_name': 'TAIEX',
        'other_indices': [],
        'companies': ['2330.TW', '2317.TW', '2454.TW', '2412.TW', '2308.TW',
                      '2881.TW', '2882.TW', '2303.TW', '1301.TW', '2891.TW'],
        'fx': 'USDTWD=X',
        'news_keywords': ['taiwan', 'taiwanese economy', 'tsmc', 'taipei'],
        'fred_series': []
    },
    'CANADA': {
        'name': 'Canada',
        'currency': 'CAD',
        'index': '^GSPTSE',  # TSX Composite
        'index_name': 'TSX',
        'other_indices': [],
        'companies': ['RY.TO', 'TD.TO', 'ENB.TO', 'CNR.TO', 'BNS.TO',
                      'BMO.TO', 'CP.TO', 'TRP.TO', 'BCE.TO', 'SU.TO'],
        'fx': 'USDCAD=X',
        'news_keywords': ['canada', 'canadian economy', 'tsx', 'toronto', 'bank of canada'],
        'fred_series': []
    }
}

# Commodities (Global)
COMMODITIES = {
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'CL=F': 'Crude Oil WTI',
    'BZ=F': 'Brent Crude',
    'NG=F': 'Natural Gas',
    'HG=F': 'Copper',
    'ZC=F': 'Corn',
    'ZW=F': 'Wheat',
    'ZS=F': 'Soybeans'
}

# Cryptocurrencies (Global)
CRYPTO = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'Ripple',
    'SOL-USD': 'Solana',
    'ADA-USD': 'Cardano',
    'DOGE-USD': 'Dogecoin'
}

def download_stock_data(symbols, period='5y'):
    """Download stock/index data from Yahoo Finance"""
    all_data = []

    for symbol in symbols:
        try:
            print(f"  Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if not hist.empty:
                hist = hist.reset_index()
                hist['symbol'] = symbol
                hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
                all_data.append(hist)
                print(f"    Got {len(hist)} rows")
            else:
                print(f"    No data for {symbol}")
        except Exception as e:
            print(f"    Error for {symbol}: {e}")

        time.sleep(0.2)  # Rate limiting

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def download_country_indices():
    """Download main indices for all countries"""
    print("\n=== Downloading Country Stock Indices ===")

    all_indices = []
    for country, config in COUNTRIES.items():
        print(f"\n{country} ({config['name']}):")

        # Main index
        symbols = [config['index']] + config.get('other_indices', [])
        df = download_stock_data(symbols, period='10y')

        if not df.empty:
            df['country'] = country
            df['country_name'] = config['name']
            all_indices.append(df)

    if all_indices:
        result = pd.concat(all_indices, ignore_index=True)
        output_file = f"{OUTPUT_DIR}/country_indices.csv"
        result.to_csv(output_file, index=False)
        print(f"\nSaved {len(result)} index records to {output_file}")
        return result
    return pd.DataFrame()

def download_country_stocks():
    """Download top company stocks for all countries"""
    print("\n=== Downloading Country Top Stocks ===")

    all_stocks = []
    for country, config in COUNTRIES.items():
        print(f"\n{country} ({config['name']}):")

        df = download_stock_data(config['companies'], period='5y')

        if not df.empty:
            df['country'] = country
            df['country_name'] = config['name']
            all_stocks.append(df)

    if all_stocks:
        result = pd.concat(all_stocks, ignore_index=True)
        output_file = f"{OUTPUT_DIR}/country_stocks.csv"
        result.to_csv(output_file, index=False)
        print(f"\nSaved {len(result)} stock records to {output_file}")
        return result
    return pd.DataFrame()

def download_exchange_rates():
    """Download exchange rates for all countries"""
    print("\n=== Downloading Exchange Rates ===")

    fx_symbols = [config['fx'] for config in COUNTRIES.values()]
    fx_symbols = list(set(fx_symbols))  # Remove duplicates

    df = download_stock_data(fx_symbols, period='10y')

    if not df.empty:
        # Map symbols to country
        fx_to_country = {config['fx']: country for country, config in COUNTRIES.items()}
        df['primary_country'] = df['symbol'].map(fx_to_country)

        output_file = f"{OUTPUT_DIR}/exchange_rates.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} exchange rate records to {output_file}")
        return df
    return pd.DataFrame()

def download_commodities():
    """Download commodity prices"""
    print("\n=== Downloading Commodities ===")

    symbols = list(COMMODITIES.keys())
    df = download_stock_data(symbols, period='10y')

    if not df.empty:
        df['commodity_name'] = df['symbol'].map(COMMODITIES)
        output_file = f"{OUTPUT_DIR}/commodities.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} commodity records to {output_file}")
        return df
    return pd.DataFrame()

def download_crypto():
    """Download cryptocurrency prices"""
    print("\n=== Downloading Cryptocurrencies ===")

    symbols = list(CRYPTO.keys())
    df = download_stock_data(symbols, period='5y')

    if not df.empty:
        df['crypto_name'] = df['symbol'].map(CRYPTO)
        output_file = f"{OUTPUT_DIR}/crypto.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} crypto records to {output_file}")
        return df
    return pd.DataFrame()

def download_country_news():
    """Download news for each country using NewsAPI"""
    print("\n=== Downloading Country-Specific News ===")

    if not NEWS_API_KEY:
        print("No NEWS_API_KEY found!")
        return pd.DataFrame()

    all_news = []

    for country, config in COUNTRIES.items():
        print(f"\n{country} ({config['name']}):")

        for keyword in config['news_keywords'][:2]:  # Limit to 2 keywords per country
            try:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': NEWS_API_KEY,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 50
                }

                response = requests.get(url, params=params)
                data = response.json()

                if data.get('status') == 'ok':
                    articles = data.get('articles', [])
                    print(f"  '{keyword}': {len(articles)} articles")

                    for article in articles:
                        all_news.append({
                            'country': country,
                            'country_name': config['name'],
                            'keyword': keyword,
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'url': article.get('url', '')
                        })
                else:
                    print(f"  '{keyword}': Error - {data.get('message', 'Unknown')}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"  Error for {keyword}: {e}")

    if all_news:
        df = pd.DataFrame(all_news)
        output_file = f"{OUTPUT_DIR}/country_news.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} news articles to {output_file}")
        return df
    return pd.DataFrame()

def download_world_bank_data():
    """Download more World Bank indicators for all countries"""
    print("\n=== Downloading World Bank Data ===")

    # Country codes for World Bank API
    wb_countries = {
        'USA': 'US', 'KOREA': 'KR', 'GERMANY': 'DE', 'FRANCE': 'FR',
        'UK': 'GB', 'JAPAN': 'JP', 'CHINA': 'CN', 'RUSSIA': 'RU',
        'INDIA': 'IN', 'TAIWAN': 'TW', 'CANADA': 'CA'
    }

    indicators = {
        'NY.GDP.MKTP.CD': 'GDP (current US$)',
        'NY.GDP.PCAP.CD': 'GDP per capita (current US$)',
        'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
        'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
        'SL.UEM.TOTL.ZS': 'Unemployment (% of total labor force)',
        'NE.EXP.GNFS.ZS': 'Exports (% of GDP)',
        'NE.IMP.GNFS.ZS': 'Imports (% of GDP)',
        'BX.KLT.DINV.CD.WD': 'Foreign direct investment (BoP, current US$)',
        'PA.NUS.FCRF': 'Official exchange rate (LCU per US$)',
        'SP.POP.TOTL': 'Population, total'
    }

    all_data = []
    country_codes = ','.join(wb_countries.values())

    for indicator_code, indicator_name in indicators.items():
        try:
            print(f"  Downloading {indicator_name}...")
            url = f"https://api.worldbank.org/v2/country/{country_codes}/indicator/{indicator_code}"
            params = {
                'format': 'json',
                'per_page': 1000,
                'date': '2000:2024'
            }

            response = requests.get(url, params=params)
            data = response.json()

            if len(data) > 1 and data[1]:
                for item in data[1]:
                    if item.get('value') is not None:
                        # Reverse map country code to our country key
                        country_code = item.get('country', {}).get('id', '')
                        country_key = next((k for k, v in wb_countries.items() if v == country_code), country_code)

                        all_data.append({
                            'country': country_key,
                            'country_name': item.get('country', {}).get('value', ''),
                            'indicator_code': indicator_code,
                            'indicator_name': indicator_name,
                            'year': item.get('date', ''),
                            'value': item.get('value')
                        })
                print(f"    Got {len([d for d in data[1] if d.get('value')])} records")

            time.sleep(0.3)

        except Exception as e:
            print(f"    Error for {indicator_code}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        output_file = f"{OUTPUT_DIR}/country_worldbank.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} World Bank records to {output_file}")
        return df
    return pd.DataFrame()

def download_fred_data():
    """Download FRED economic data"""
    print("\n=== Downloading FRED Data ===")

    if not FRED_API_KEY:
        print("No FRED_API_KEY found!")
        return pd.DataFrame()

    # Key FRED series
    series = {
        # US Economy
        'GDP': 'US GDP',
        'GDPC1': 'US Real GDP',
        'UNRATE': 'US Unemployment Rate',
        'CPIAUCSL': 'US CPI',
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Rate',
        'T10YIE': '10-Year Breakeven Inflation',
        'DEXUSEU': 'USD/EUR Exchange Rate',
        'DEXJPUS': 'JPY/USD Exchange Rate',
        'DEXKOUS': 'KRW/USD Exchange Rate',
        'DEXCHUS': 'CNY/USD Exchange Rate',
        # Global
        'MCOILWTICO': 'WTI Oil Price',
        'GOLDAMGBD228NLBM': 'Gold Price',
        'VIXCLS': 'VIX Volatility Index',
        'BAMLH0A0HYM2': 'High Yield Bond Spread',
        # Trade
        'IMPGS': 'US Imports of Goods and Services',
        'EXPGS': 'US Exports of Goods and Services'
    }

    all_data = []

    for series_id, name in series.items():
        try:
            print(f"  Downloading {name}...")
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'observation_start': '2000-01-01'
            }

            response = requests.get(url, params=params)
            data = response.json()

            if 'observations' in data:
                for obs in data['observations']:
                    if obs.get('value') != '.':
                        all_data.append({
                            'series_id': series_id,
                            'series_name': name,
                            'date': obs.get('date'),
                            'value': float(obs.get('value'))
                        })
                print(f"    Got {len(data['observations'])} records")

            time.sleep(0.2)

        except Exception as e:
            print(f"    Error for {series_id}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        output_file = f"{OUTPUT_DIR}/fred_extended.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} FRED records to {output_file}")
        return df
    return pd.DataFrame()

def create_data_summary():
    """Create summary of all downloaded data"""
    print("\n=== Creating Data Summary ===")

    summary = []

    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(OUTPUT_DIR, filename)
            df = pd.read_csv(filepath)

            summary.append({
                'file': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'size_kb': os.path.getsize(filepath) / 1024,
                'column_list': ', '.join(df.columns[:5]) + '...'
            })

    summary_df = pd.DataFrame(summary)
    summary_file = f"{OUTPUT_DIR}/DATA_SUMMARY.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n" + "="*60)
    print("DATA DOWNLOAD SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

    return summary_df

def main():
    print("="*60)
    print("COUNTRY-FOCUSED DATA DOWNLOADER")
    print("11 Countries: USA, Korea, Germany, France, UK,")
    print("              Japan, China, Russia, India, Taiwan, Canada")
    print("="*60)

    # Download all data types
    download_country_indices()
    download_country_stocks()
    download_exchange_rates()
    download_commodities()
    download_crypto()
    download_country_news()
    download_world_bank_data()
    download_fred_data()

    # Create summary
    create_data_summary()

    print("\n" + "="*60)
    print("ALL DOWNLOADS COMPLETE!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
