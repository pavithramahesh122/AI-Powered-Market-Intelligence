import pandas as pd
import os

# CONFIGURATION 
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

    # Data Cleaning and Preparation (CRITICAL RENAMING FIX) 
    
    # Check if 'channel' exists before renaming
    if 'channel' in df.columns:
        df.rename(columns={'channel': 'Platform'}, inplace=True)
        print("  -> SUCCESS: Renamed 'channel' column to 'Platform'.")
    
    # Map other columns found in your file to the standardized names the script expects.
    df.rename(columns={
        'spend_usd': 'Ad_Spend_USD',
        'revenue_usd': 'Revenue_USD',
        'installs': 'Conversions',
        # 'channel' is now handled above
        'monthly_search_volume': 'SEO_Search_Volume',
        'seo_category': 'SEO_Keyword',
        'avg_position': 'SEO_Difficulty' # Assuming avg_position is a proxy for difficulty or should be treated as such
    }, inplace=True)

    # Check for required columns after renaming
    required_cols = ['Ad_Spend_USD', 'Conversions', 'Revenue_USD', 'Platform', 'SEO_Search_Volume', 'SEO_Difficulty', 'SEO_Keyword']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"CRITICAL ERROR: Input CSV is missing required columns: {missing_cols}")
        print("Available columns after attempted renaming are:", list(df.columns))
        return
        
    # Metric Calculation 

    # 1. Calculate Cost per Acquisition (CPA or CAC)
    df_valid = df[df['Conversions'] > 0].copy()
    df_valid['CAC_USD'] = df_valid['Ad_Spend_USD'] / df_valid['Conversions']

    # 2. Calculate Return on Ad Spend (ROAS)
    df_valid['ROAS'] = df_valid['Revenue_USD'] / df_valid['Ad_Spend_USD']
    df_valid.loc[df_valid['Ad_Spend_USD'] == 0, 'ROAS'] = 0.0

    # 3. Final cleanup and save
    df_merged = df.merge(df_valid[['CAC_USD', 'ROAS']], left_index=True, right_index=True, how='left')
    
    # Fill NaN (using loc to avoid FutureWarning)
    df_merged.loc[df_merged['CAC_USD'].isna(), 'CAC_USD'] = 0
    df_merged.loc[df_merged['ROAS'].isna(), 'ROAS'] = 0.0

    os.makedirs(os.path.dirname(PROCESSED_D2C_FILE), exist_ok=True)
    
    # Save the processed data
    df_merged.to_csv(PROCESSED_D2C_FILE, index=False)
    
    print(f"\nDELIVERABLE 4 (Part 2/3): Saved processed D2C metrics to {PROCESSED_D2C_FILE}")
    
    return df_merged

if __name__ == '__main__':
    analyze_d2c_metrics()
