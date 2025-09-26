import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
RAW_D2C_FILE = 'data/raw/d2c_campaigns_raw.csv'
# --- END CONFIGURATION ---

def generate_mock_d2c_data(num_records: int = 2000):
    """
    Generates mock D2C campaign performance data, including SEO keywords.
    """
    print(f"-> Generating {num_records} mock D2C campaign records...")
    
    # Mock data components
    platforms = ['Facebook', 'Google Search', 'Instagram', 'TikTok']
    campaign_types = ['Brand Awareness', 'Performance/ROAS', 'App Install', 'Re-engagement']
    
    # Base keywords (used for SEO/Search Campaign data)
    keywords = [
        "best budget planning app", "free habit tracker", "mobile finance solutions", 
        "easy goal setting", "top rated productivity app", "secure investment tracker"
    ]
    
    data = {
        'Date': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_records), unit='D'),
        'Campaign_Platform': np.random.choice(platforms, num_records, p=[0.35, 0.40, 0.15, 0.10]),
        'Campaign_Type': np.random.choice(campaign_types, num_records, p=[0.1, 0.5, 0.2, 0.2]),
        'Ad_Spend_USD': np.random.uniform(5.0, 500.0, num_records).round(2),
        'Clicks': np.random.randint(50, 5000, num_records),
        'Impressions': np.random.randint(500, 50000, num_records),
        'Conversions': np.random.poisson(10, num_records), # Conversions heavily weighted lower
        'Revenue_USD': np.random.uniform(0, 1000, num_records).round(2),
        'Target_Audience': np.random.choice(['Millennials', 'Gen Z', 'Working Professionals', 'Students'], num_records),
    }
    
    df = pd.DataFrame(data)

    # 1. Filter Conversions/Revenue: Reduce revenue if conversions are low
    df.loc[df['Conversions'] < 5, 'Revenue_USD'] = df['Revenue_USD'] * 0.1
    df.loc[df['Campaign_Type'] == 'App Install', 'Revenue_USD'] = 0 # App Installs are not direct revenue campaigns

    # 2. Inject Keyword Data (primarily for Google Search campaigns)
    df['Keyword'] = np.where(
        df['Campaign_Platform'] == 'Google Search',
        np.random.choice(keywords, len(df), p=[0.25, 0.15, 0.2, 0.15, 0.1, 0.15]),
        'N/A'
    )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(RAW_D2C_FILE), exist_ok=True)
    
    # Save the file
    df.to_csv(RAW_D2C_FILE, index=False)
    
    print(f"\nDELIVERABLE 4 (Part 1/3): Saved mock D2C raw data to {RAW_D2C_FILE}")
    
    return df

if __name__ == '__main__':
    generate_mock_d2c_data()
