import pandas as pd
import numpy as np
import requests
import json
import time
from typing import List, Dict, Any

KAGGLE_FILE = 'data/google_play_apps.csv'
REVIEWS_FILE = 'data/googleplaystore_user_reviews.csv'
COMBINED_FILE = 'data/processed/apps_combined.csv'

RAPIDAPI_KEY = '8065b3eb6fmsh9ff6a831c5406d4p1ff81ejsncec291d7a03d' 
RAPIDAPI_HOST = 'appstore-scrapper-api.p.rapidapi.com'
RAPIDAPI_URL = f'https://{RAPIDAPI_HOST}/v1/app-store-api/search'


def load_and_clean_kaggle_data() -> pd.DataFrame:
    print("-> Starting Kaggle data ingestion and cleaning...")
    
    try:
        df = pd.read_csv(KAGGLE_FILE)
    except FileNotFoundError:
        print(f"Error: Required file not found. Check if '{KAGGLE_FILE}' is in the 'data/' folder.")
        raise
    
    # CRITICAL FIX: Drop the known corrupted row
    df = df[df['Installs'] != 'Free'].copy()

    df.drop_duplicates(inplace=True)

    # 1. Clean 'Installs' (FIXED: added parentheses to Int64Dtype())
    df['Installs'] = df['Installs'].astype(str).apply(
        lambda x: x.replace('+', '').replace(',', '')
    ).astype(float).astype(pd.Int64Dtype())

    # 2. Clean 'Reviews' (Using Int64Dtype to allow for NaN)
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').astype(pd.Int64Dtype())

    # 3. Clean 'Size'
    def clean_size(size):
        if isinstance(size, str):
            if 'M' in size:
                return float(size.replace('M', '')) * 1024 * 1024
            elif 'k' in size:
                return float(size.replace('k', '')) * 1024
            elif size == 'Varies with device':
                return np.nan
        return size
    df['Size_Bytes'] = df['Size'].apply(clean_size)
    
    # 4. Clean 'Price'
    df['Price'] = df['Price'].astype(str).apply(
        lambda x: float(x.replace('$', '')) if '$' in x else float(x)
    )

    # 5. Drop remaining NA for critical columns
    df.dropna(subset=['App', 'Category', 'Rating', 'Installs'], inplace=True)
    
    # 6. Rename Columns
    df.rename(columns={
        'App': 'Name', 
        'Content Rating': 'Content_Rating',
        'Android Ver': 'Required_Android_Version',
        'Reviews': 'Review_Count',
        'Last Updated': 'Last_Updated_Date'
    }, inplace=True)
    
    df.drop(columns=['Size', 'Genres', 'Current Ver'], inplace=True, errors='ignore')
    df['Source'] = 'Google_Play'
    
    print(f"-> Kaggle metadata cleaned. Shape: {df.shape}")
    return df

def process_and_merge_reviews(df_main: pd.DataFrame) -> pd.DataFrame:
    print("-> Processing and merging user reviews data...")
    try:
        df_reviews = pd.read_csv(REVIEWS_FILE)
    except FileNotFoundError:
        print(f"    WARNING: Reviews file not found at {REVIEWS_FILE}. Skipping sentiment analysis.")
        df_main['Avg_Sentiment_Polarity'] = np.nan
        return df_main

    df_reviews.dropna(subset=['Sentiment_Polarity'], inplace=True)
    df_reviews['Sentiment_Polarity'] = pd.to_numeric(
        df_reviews['Sentiment_Polarity'], errors='coerce'
    )
    df_reviews.dropna(subset=['Sentiment_Polarity'], inplace=True)
    
    df_sentiment = df_reviews.groupby('App')['Sentiment_Polarity'].mean().reset_index()
    df_sentiment.rename(
        columns={'App': 'Name', 'Sentiment_Polarity': 'Avg_Sentiment_Polarity'}, 
        inplace=True
    )
    
    df_combined = pd.merge(
        df_main, 
        df_sentiment, 
        on='Name', 
        how='left'
    )
    
    print(f"-> User reviews processed and merged. Total apps with sentiment: {df_sentiment.shape[0]}")
    return df_combined

def get_mock_appstore_data(app_names: List[str]) -> pd.DataFrame:
    print("-> Generating mock App Store data...")
    
    categories = ['FINANCE', 'GAME', 'SOCIAL', 'PHOTOGRAPHY', 'BUSINESS']
    mock_list = []
    
    sample_names = app_names[:100] 
    
    for name in sample_names:
        mock_list.append({
            'Name': name,
            'Category': np.random.choice(categories),
            'Rating': np.random.uniform(3.8, 5.0),
            'Review_Count': np.random.randint(500, 50000),
            'Installs': np.random.choice([100000, 500000, 1000000, 5000000]),
            'Type': 'Paid' if np.random.rand() < 0.2 else 'Free',
            'Price': np.random.choice([0.0, 0.99, 1.99, 4.99]),
            'Content_Rating': np.random.choice(['Everyone', 'Teen', 'Mature 17+']),
            'Last_Updated_Date': '2024-08-01',
            'Size_Bytes': np.nan, 
            'Required_Android_Version': np.nan,
            'Avg_Sentiment_Polarity': np.random.uniform(0.1, 0.4),
            'Source': 'Apple_App_Store'
        })
        
    df_appstore = pd.DataFrame(mock_list)
    return df_appstore


def fetch_appstore_data(app_names: List[str], use_mock=True) -> pd.DataFrame:
    
    if use_mock:
        return get_mock_appstore_data(app_names)
        
    print("-> Attempting LIVE App Store API calls (Ensure RAPIDAPI_KEY is set)...")
    
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    api_data_list = []
    
    for app_name in app_names[:50]:
        querystring = {"query": app_name, "country": "us"}
        retries = 3
        
        for attempt in range(retries):
            try:
                response = requests.get(RAPIDAPI_URL, headers=headers, params=querystring, timeout=10)
                response.raise_for_status()
                
                app_details = response.json()
                if app_details and 'data' in app_details and app_details['data']:
                    result = app_details['data'][0] 
                    
                    api_data_list.append({
                        'Name': result.get('title', app_name),
                        'Category': result.get('category', 'UNKNOWN'),
                        'Rating': result.get('rating'),
                        'Review_Count': result.get('review_count'),
                        'Installs': result.get('installs_estimate', 0), 
                        'Type': 'Free' if result.get('price', 0) == 0 else 'Paid',
                        'Price': result.get('price', 0),
                        'Content_Rating': result.get('content_rating'),
                        'Last_Updated_Date': result.get('last_updated'),
                        'Size_Bytes': np.nan, 
                        'Required_Android_Version': np.nan,
                        'Avg_Sentiment_Polarity': np.nan,
                        'Source': 'Apple_App_Store'
                    })
                    break
                
                if response.status_code == 429:
                    print(f"    Rate limit hit. Retrying in {2**attempt} seconds...")
                    time.sleep(2**attempt)
                
            except requests.exceptions.RequestException as e:
                print(f"    API call failed for {app_name} on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                else:
                    print(f"    Failed to retrieve data for {app_name} after {retries} attempts.")
            except json.JSONDecodeError:
                print(f"    Failed to decode JSON response for {app_name}.")
                break

    df_appstore = pd.DataFrame(api_data_list)
    print(f"-> App Store data processed (Live API). Shape: {df_appstore.shape}")
    return df_appstore

def build_unified_dataset(use_mock_api=True) -> pd.DataFrame:
    
    df_google = load_and_clean_kaggle_data()
    df_google_full = process_and_merge_reviews(df_google)
    
    sample_apps = df_google['Name'].sample(200, replace=True).unique().tolist()
    
    df_appstore = fetch_appstore_data(sample_apps, use_mock=use_mock_api)
    
    columns_to_keep = [
        'Name', 'Category', 'Rating', 'Review_Count', 'Installs', 
        'Type', 'Price', 'Content_Rating', 'Size_Bytes', 'Required_Android_Version',
        'Last_Updated_Date', 'Avg_Sentiment_Polarity', 'Source'
    ]
    
    df_google_full.rename(columns={'Last Updated': 'Last_Updated_Date'}, inplace=True)
    
    df_google_final = df_google_full[columns_to_keep]
    df_appstore_final = df_appstore[columns_to_keep]
    
    df_combined = pd.concat([df_google_final, df_appstore_final], ignore_index=True)
    
    print(f"\n--- Unified Dataset Built ---")
    print(f"Total Records: {len(df_combined)}")
    
    df_combined.to_csv(COMBINED_FILE, index=False)
    print(f"DELIVERABLE 1: Saved clean combined dataset to {COMBINED_FILE}")
    
    return df_combined

if __name__ == '__main__':
    build_unified_dataset(use_mock_api=True)
