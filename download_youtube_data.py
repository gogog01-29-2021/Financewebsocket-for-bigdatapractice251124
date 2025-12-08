"""
YouTube Data Downloader for Country Financial Analytics
Downloads real YouTube videos and comments about finance/economy for each country

Uses YouTube Data API v3
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OUTPUT_DIR = 'organized_data/raw_data/country_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Country-specific search queries for financial content
COUNTRY_QUERIES = {
    'USA': [
        'US economy news',
        'Wall Street stock market',
        'Federal Reserve interest rate',
        'S&P 500 analysis',
        'US inflation news'
    ],
    'KOREA': [
        'Korea economy news',
        'KOSPI stock market',
        'Samsung stock analysis',
        'Korean won exchange rate',
        'Bank of Korea'
    ],
    'GERMANY': [
        'German economy news',
        'DAX stock market',
        'Bundesbank news',
        'Germany inflation',
        'German industry'
    ],
    'FRANCE': [
        'France economy news',
        'CAC 40 stock market',
        'French economy',
        'Paris stock exchange',
        'France inflation'
    ],
    'UK': [
        'UK economy news',
        'FTSE 100 stock market',
        'Bank of England',
        'British pound news',
        'London stock exchange'
    ],
    'JAPAN': [
        'Japan economy news',
        'Nikkei stock market',
        'Bank of Japan',
        'Yen exchange rate',
        'Toyota stock'
    ],
    'CHINA': [
        'China economy news',
        'Shanghai stock market',
        'Chinese yuan',
        'PBOC news',
        'Alibaba stock'
    ],
    'RUSSIA': [
        'Russia economy news',
        'Moscow stock exchange',
        'Russian ruble',
        'Russia sanctions economy',
        'Gazprom stock'
    ],
    'INDIA': [
        'India economy news',
        'Sensex stock market',
        'Reserve Bank of India',
        'Indian rupee',
        'Reliance stock'
    ],
    'TAIWAN': [
        'Taiwan economy news',
        'TSMC stock analysis',
        'Taiwan semiconductor',
        'Taiwan stock market',
        'Taiwan dollar'
    ],
    'CANADA': [
        'Canada economy news',
        'TSX stock market',
        'Bank of Canada',
        'Canadian dollar',
        'Canada interest rate'
    ]
}

def search_videos(query, max_results=10):
    """Search YouTube videos"""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'order': 'relevance',
        'maxResults': max_results,
        'key': YOUTUBE_API_KEY,
        'publishedAfter': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'relevanceLanguage': 'en'
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if 'error' in data:
            print(f"  API Error: {data['error']['message']}")
            return []

        return data.get('items', [])
    except Exception as e:
        print(f"  Error: {e}")
        return []

def get_video_details(video_ids):
    """Get video statistics and details"""
    if not video_ids:
        return {}

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'statistics,contentDetails',
        'id': ','.join(video_ids),
        'key': YOUTUBE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if 'error' in data:
            return {}

        result = {}
        for item in data.get('items', []):
            stats = item.get('statistics', {})
            result[item['id']] = {
                'viewCount': int(stats.get('viewCount', 0)),
                'likeCount': int(stats.get('likeCount', 0)),
                'commentCount': int(stats.get('commentCount', 0))
            }
        return result
    except Exception as e:
        print(f"  Stats error: {e}")
        return {}

def get_video_comments(video_id, max_comments=20):
    """Get comments for a video"""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': max_comments,
        'order': 'relevance',
        'key': YOUTUBE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if 'error' in data:
            return []

        comments = []
        for item in data.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'comment_id': item['id'],
                'video_id': video_id,
                'author': snippet.get('authorDisplayName', ''),
                'text': snippet.get('textDisplay', ''),
                'published_at': snippet.get('publishedAt', ''),
                'like_count': snippet.get('likeCount', 0)
            })
        return comments
    except Exception as e:
        return []

def download_country_youtube_data():
    """Download YouTube data for all countries"""
    print("="*60)
    print("YOUTUBE DATA DOWNLOADER")
    print("="*60)

    if not YOUTUBE_API_KEY:
        print("ERROR: No YOUTUBE_API_KEY found in .env!")
        return

    all_videos = []
    all_comments = []

    for country, queries in COUNTRY_QUERIES.items():
        print(f"\n{country}:")
        country_videos = []

        for query in queries:
            print(f"  Searching: '{query}'...")
            videos = search_videos(query, max_results=5)

            if videos:
                video_ids = [v['id']['videoId'] for v in videos]
                stats = get_video_details(video_ids)

                for video in videos:
                    video_id = video['id']['videoId']
                    snippet = video['snippet']
                    video_stats = stats.get(video_id, {})

                    video_data = {
                        'country': country,
                        'query': query,
                        'video_id': video_id,
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', '')[:500],
                        'channel': snippet.get('channelTitle', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'view_count': video_stats.get('viewCount', 0),
                        'like_count': video_stats.get('likeCount', 0),
                        'comment_count': video_stats.get('commentCount', 0)
                    }
                    country_videos.append(video_data)

                    # Get comments for top videos
                    if video_stats.get('commentCount', 0) > 0:
                        comments = get_video_comments(video_id, max_comments=10)
                        for comment in comments:
                            comment['country'] = country
                            comment['query'] = query
                            all_comments.append(comment)

                print(f"    Got {len(videos)} videos")

            time.sleep(0.2)  # Rate limiting

        all_videos.extend(country_videos)
        print(f"  Total for {country}: {len(country_videos)} videos")

    # Save videos
    if all_videos:
        videos_df = pd.DataFrame(all_videos)
        videos_file = f"{OUTPUT_DIR}/youtube_videos.csv"
        videos_df.to_csv(videos_file, index=False)
        print(f"\nSaved {len(videos_df)} videos to {videos_file}")

    # Save comments
    if all_comments:
        comments_df = pd.DataFrame(all_comments)
        comments_file = f"{OUTPUT_DIR}/youtube_comments.csv"
        comments_df.to_csv(comments_file, index=False)
        print(f"Saved {len(comments_df)} comments to {comments_file}")

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    if all_videos:
        summary = pd.DataFrame(all_videos).groupby('country').agg({
            'video_id': 'count',
            'view_count': 'sum',
            'like_count': 'sum',
            'comment_count': 'sum'
        }).rename(columns={'video_id': 'videos'})
        print(summary)
    print("="*60)

if __name__ == "__main__":
    download_country_youtube_data()
