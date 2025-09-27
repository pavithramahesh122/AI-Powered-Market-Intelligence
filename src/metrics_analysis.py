import pandas as pd
import os
import numpy as np

# CONFIGURATION 
# FIX 1: Corrected path to match output from d2c_data_generator.py
RAW_D2C_FILE = 'data/d2c_campaigns_raw.csv' 
PROCESSED_D2C_FILE = 'data/processed/d2c_campaigns_processed.csv'


def analyze_d2c_metrics():
    """
    Calculates key funnel metrics (CAC, ROAS) and prepares SEO data.
    """
    print("-> Starting D2C metrics analysis...")

    try:
        # Load the raw data
        df = pd.read_csv(RAW_D2C_FILE)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Raw D2C data not found at {RAW_D2C_FILE}. Ensure you have provided or generated this file.")
        return

    # Data Cleaning and Preparation 
    
    # FIX 2: Check for 'Campaign_Platform' (from d2c_data_generator.py) or 'channel' and standardize to 'Platform'
    if 'Campaign_Platform' in df.columns:
        df.rename(columns={'Campaign_Platform': 'Platform'}, inplace=True)
        print("  -> SUCCESS: Renamed 'Campaign_Platform' column to 'Platform'.")
    elif 'channel' in df.columns: 
        df.rename(columns={'channel': 'Platform'}, inplace=True)
        print("  -> SUCCESS: Renamed 'channel' column to 'Platform'.")
    
    # Defensive renames for common column naming conventions
    df.rename(columns={
        'spend_usd': 'Ad_Spend_USD',
        'revenue_usd': 'Revenue_USD',
        'installs': 'Conversions',
        'monthly_search_volume': 'SEO_Search_Volume',
        'search_difficulty': 'SEO_Difficulty',
        'target_keyword': 'SEO_Keyword'
    }, inplace=True)


    # Validation (Ensuring required columns for calculation exist after renaming)
    required_cols = ['Ad_Spend_USD', 'Conversions', 'Revenue_USD', 'Platform', 'SEO_Search_Volume', 'SEO_Difficulty', 'SEO_Keyword']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"CRITICAL ERROR: Input CSV is missing required columns: {missing_cols}")
        print("Available columns after attempted renaming are:", list(df.columns))
        return
        
    # Metric Calculation 

    # 1. Calculate Cost per Acquisition (CPA or CAC)
    df_valid = df[df['Conversions'] > 0].copy()
    # Handle NaNs and 0s in Ad_Spend_USD before calculation for robustness
    df_valid['Ad_Spend_USD'] = df_valid['Ad_Spend_USD'].fillna(0)
    df_valid['CAC_USD'] = df_valid['Ad_Spend_USD'] / df_valid['Conversions']

    # 2. Calculate Return on Ad Spend (ROAS)
    df_valid['ROAS'] = df_valid['Revenue_USD'] / df_valid['Ad_Spend_USD']
    # Handle division by zero (0 Ad Spend) explicitly
    df_valid.loc[df_valid['Ad_Spend_USD'] == 0, 'ROAS'] = 0.0

    # 3. Final cleanup and save
    df_merged = df.merge(df_valid[['CAC_USD', 'ROAS']], left_index=True, right_index=True, how='left')
    
    # Fill NaN for rows that had 0 conversions or 0 spend 
    df_merged.loc[df_merged['CAC_USD'].isna(), 'CAC_USD'] = 0 
    df_merged.loc[df_merged['ROAS'].isna(), 'ROAS'] = 0.0 
    
    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(PROCESSED_D2C_FILE), exist_ok=True)
    df_merged.to_csv(PROCESSED_D2C_FILE, index=False)
    print(f"-> D2C metrics analysis complete. Saved processed data to {PROCESSED_D2C_FILE}")
    return df_merged

if __name__ == '__main__':
    analyze_d2c_metrics()