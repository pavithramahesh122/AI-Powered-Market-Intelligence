import pandas as pd
import os

# --- CONFIGURATION ---
RAW_D2C_FILE = 'data/raw/d2c_campaigns_raw.csv'
PROCESSED_D2C_FILE = 'data/processed/d2c_campaigns_processed.csv'
# --- END CONFIGURATION ---

def analyze_d2c_metrics():
    """Calculates key funnel metrics (CAC, ROAS) and prepares SEO data."""
    print("-> Starting D2C metrics analysis...")

    try:
        df = pd.read_csv(RAW_D2C_FILE)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Raw D2C data not found at {RAW_D2C_FILE}. Run d2c_data_generator.py first.")
        return

    # 1. Calculate Cost per Acquisition (CPA or CAC)
    # Filter out records with no conversions to avoid division by zero
    df_valid = df[df['Conversions'] > 0].copy()
    
    # Calculation
    df_valid['CAC_USD'] = df_valid['Ad_Spend_USD'] / df_valid['Conversions']

    # 2. Calculate Return on Ad Spend (ROAS)
    df_valid['ROAS'] = df_valid['Revenue_USD'] / df_valid['Ad_Spend_USD']
    
    # Handle cases where spend is zero (assign max practical ROAS or Inf)
    df_valid.loc[df_valid['Ad_Spend_USD'] == 0, 'ROAS'] = 0.0

    # 3. Final cleanup and save
    
    # Merge the calculated metrics back to the original full dataframe (if needed), 
    # but for simplicity, we'll just save the cleaned/calculated one.
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_D2C_FILE), exist_ok=True)
    df_valid.to_csv(PROCESSED_D2C_FILE, index=False)
    
    print(f"DELIVERABLE 5 (Part 2/3): Saved processed D2C metrics to {PROCESSED_D2C_FILE}")
    return df_valid

if __name__ == '__main__':
    analyze_d2c_metrics()