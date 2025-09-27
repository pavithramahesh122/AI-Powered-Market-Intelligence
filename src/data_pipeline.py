import pandas as pd
import numpy as np
import requests
import json
import time
import os
from typing import List, Dict, Any

KAGGLE_FILE = 'data/google_play_apps.csv'
REVIEWS_FILE = 'data/googleplaystore_user_reviews.csv'
COMBINED_FILE = 'data/processed/apps_combined.csv'

# MOCK API CONFIGURATION

RAPIDAPI_KEY = '8065b3eb6fmsh9ff6a831c5406d4p1ff81ejsncec291d7a03d' 
RAPIDAPI_HOST = 'appstore-scrapper-api.p.rapidapi.com'
RAPIDAPI_URL = f'https://{RAPIDAPI_HOST}/v1/app-store-api/search'

# Headers for the mock API call
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

def load_and_clean_kaggle_data() -> pd.DataFrame:
    """Loads, cleans, and standardizes the Kaggle Google Play dataset."""
    print("-> Starting Kaggle data ingestion and cleaning...")
    
    try:
        df = pd.read_csv(KAGGLE_FILE)
    except FileNotFoundError:
        print(f"Error: Required file not found. Check if '{KAGGLE_FILE}' is in the 'data/' folder.")
        raise
    

    df = df[df['Installs'] != 'Free'].copy()

    df.drop_duplicates(inplace=True)

    df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '', regex=False).astype('Int64')

    df['Price'] = df['Price'].str.replace('$', '', regex=False).astype(float)

    df['Reviews'] = df['Reviews'].astype(int)

    def size_to_bytes(size):
        if pd.isna(size):
            return np.nan
        size = str(size).upper()
        if 'M' in size:
            return float(size.replace('M', '')) * (1024 ** 2)
        elif 'K' in size:
            return float(size.replace('K', '')) * 1024
        return np.nan

    df['Size_Bytes'] = df['Size'].apply(size_to_bytes)
    df.drop('Size', axis=1, inplace=True)
    
    # Standardize metadata columns
    df.rename(columns={'App': 'Name'}, inplace=True)
    df['Source'] = 'Google Play Store'
    
    print("-> Kaggle data cleaning complete.")
    return df[['Name', 'Category', 'Rating', 'Reviews', 'Installs', 'Type', 'Price', 
               'Content Rating', 'Last Updated', 'Android Ver', 'Source', 'Size_Bytes']].copy()


def process_and_merge_reviews(df_google: pd.DataFrame) -> pd.DataFrame:
    """Loads and cleans reviews data, then merges calculated sentiment polarity with the main dataset."""
    print("-> Processing user reviews...")
    try:
        df_reviews = pd.read_csv(REVIEWS_FILE)
    except FileNotFoundError:
        print(f"Error: Required file not found. Check if '{REVIEWS_FILE}' is in the 'data/' folder. Skipping sentiment calculation.")
        df_google['Avg_Sentiment_Polarity'] = 0.0
        return df_google
    
    # Calculate average sentiment polarity per app
    df_reviews_avg = df_reviews.groupby('App')['Sentiment_Polarity'].mean().reset_index()
    df_reviews_avg.rename(columns={'App': 'Name', 'Sentiment_Polarity': 'Avg_Sentiment_Polarity'}, inplace=True)
    df_merged = pd.merge(df_google, df_reviews_avg, on='Name', how='left')
    df_merged['Avg_Sentiment_Polarity'] = df_merged['Avg_Sentiment_Polarity'].fillna(0) 
    
    print("-> User review processing and merge complete.")
    return df_merged


def fetch_appstore_data(app_names: List[str], use_mock: bool = True) -> pd.DataFrame:
    """
    Fetches data from the mock App Store API, retrieving details for the single 
    best search result to simulate an efficient, cost-conscious API call.
    """
    print(f"\n-> Fetching App Store data for {len(app_names)} unique apps...")
    api_data_list = []
    
    for app_name in app_names:
        

        if use_mock:
            app_details = {
                "results": [
                    {
                        "Name": app_name,
                        "Category": np.random.choice(['Games', 'Finance', 'Health']),
                        "Rating": round(np.random.uniform(4.0, 5.0), 1),
                        "Reviews": np.random.randint(5000, 500000),
                        "Installs": np.random.randint(100, 50000000),
                        "Type": np.random.choice(['Free', 'Paid'], p=[0.85, 0.15]),
                        "Price": np.random.choice([0.0, round(np.random.uniform(0.99, 9.99), 2)], p=[0.85, 0.15]),
                        "Content Rating": np.random.choice(['Everyone', '9+', '12+']),
                        "Size_Bytes": np.random.randint(15 * (1024**2), 150 * (1024**2)),
                        "Required Android Ver": "iOS 14.0+", # Mock data field name
                        "Last Updated": time.strftime("%B %d, %Y"),
                        "Avg_Sentiment_Polarity": round(np.random.uniform(-0.5, 0.9), 2),
                        "Source": "App Store (Mock)"
                    }
                ]
            }
        else:
            try:
                response = requests.get(RAPIDAPI_URL, headers=HEADERS, params={'query': app_name})
                response.raise_for_status() 
                app_details = response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"    Failed to fetch/decode API response for {app_name}. Error: {e}")
                continue 

        if 'results' in app_details and len(app_details['results']) > 0:
            result = app_details['results'][0]
            result['Review_Count'] = result.pop('Reviews')
            result['Required_Android_Version'] = result.pop('Required Android Ver')
            result['Last_Updated_Date'] = result.pop('Last Updated')
            
            api_data_list.append(result)
            
        else:
            print(f"    No results found for {app_name}.")

    df_appstore = pd.DataFrame(api_data_list)
    
    # Rename columns for final consistency
    df_appstore.rename(columns={'Content Rating': 'Content_Rating'}, inplace=True)

    print(f"-> App Store data processed (Mock API). Shape: {df_appstore.shape}")
    return df_appstore


def build_unified_dataset(use_mock_api=True) -> pd.DataFrame:
    """Loads, cleans, fetches, and merges all data sources into one unified DataFrame."""
    
    df_google = load_and_clean_kaggle_data()
    df_google_full = process_and_merge_reviews(df_google)
    
    # Sample a list of app names from Google Play for App Store lookup (e.g., 200 unique apps)
    sample_size = min(200, df_google['Name'].nunique())
    sample_apps = df_google['Name'].sample(sample_size, replace=False).unique().tolist()
    
    df_appstore = fetch_appstore_data(sample_apps, use_mock=use_mock_api)
    
    # Define the consistent column structure for the final DataFrame
    columns_to_keep = [
        'Name', 'Category', 'Rating', 'Review_Count', 'Installs', 
        'Type', 'Price', 'Content_Rating', 'Size_Bytes', 'Required_Android_Version',
        'Last_Updated_Date', 'Avg_Sentiment_Polarity', 'Source'
    ]
    
    # Rename columns in Google Play data to match the final structure (critical for merging)
    df_google_full.rename(columns={
        'Reviews': 'Review_Count', 
        'Content Rating': 'Content_Rating', 
        'Last Updated': 'Last_Updated_Date',
        'Android Ver': 'Required_Android_Version'
    }, inplace=True)
    
 
    df_google_final = df_google_full[columns_to_keep]
    df_appstore_final = df_appstore.reindex(columns=columns_to_keep, fill_value=np.nan)
    
    # Combine the datasets
    df_combined = pd.concat([df_google_final, df_appstore_final], ignore_index=True)
    
    # Final type consistency and save
    df_combined['Installs'] = df_combined['Installs'].astype('Int64')
    
    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
    df_combined.to_csv(COMBINED_FILE, index=False)
    
    print(f"\nDELIVERABLE 1: Saved unified combined dataset to {COMBINED_FILE}. Total rows: {len(df_combined)}")
    return df_combined

if __name__ == '__main__':
    # Run the full pipeline using the mock API
    build_unified_dataset(use_mock_api=True)
