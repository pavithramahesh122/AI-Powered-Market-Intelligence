import pandas as pd
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import os

# --- CONFIGURATION ---
D2C_PROCESSED_FILE = 'data/processed/d2c_campaigns_processed.csv'
CREATIVE_OUTPUT_FILE = 'reports/d2c_creative_output.json'
# NOTE: This client automatically looks for the GEMINI_API_KEY environment variable.
client = genai.Client()
# --- END CONFIGURATION ---

# --- PYDANTIC SCHEMA FOR STRUCTURED CREATIVE OUTPUT ---

class CreativeOutput(BaseModel):
    """A single AI-generated creative asset."""
    asset_type: str = Field(..., description="Headline, Meta Description, or PDP Text.")
    target_platform: str = Field(..., description="Google Search, Facebook, Product Page.")
    optimization_metric: str = Field(..., description="The KPI the creative is designed to improve (e.g., ROAS, Clicks, Conversion).")
    copy: str = Field(..., description="The final generated marketing text.")
    rationale: str = Field(..., description="Explanation of why this copy is effective based on the provided metrics (e.g., uses high-ROAS keywords).")

class D2CCreativeReport(BaseModel):
    """The final structured report for D2C creative assets."""
    focus_summary: str = Field(..., description="A summary of the core metrics guiding the creative generation (e.g., best performing platform, top SEO keyword).")
    creative_assets: List[CreativeOutput]


# --- LLM CREATIVE GENERATION ---

def get_d2c_analysis_summary(df_processed: pd.DataFrame) -> dict:
    """Calculates the necessary aggregated metrics for the LLM prompt."""
    
    # 1. Funnel Insights
    platform_summary = df_processed.groupby('Platform').agg(
        Avg_CAC=('CAC_USD', 'mean'),
        Avg_ROAS=('ROAS', 'mean'),
        Total_Spend=('Ad_Spend_USD', 'sum')
    ).sort_values(by='Avg_ROAS', ascending=False).nlargest(3).round(2).reset_index()
    
    # 2. SEO Insights (Find the highest potential keyword)
    df_processed['SEO_Potential_Score'] = df_processed['SEO_Search_Volume'] / df_processed['SEO_Difficulty']
    best_seo = df_processed.sort_values(by='SEO_Potential_Score', ascending=False).iloc[0]

    return {
        "Top_3_Platforms_by_ROAS": platform_summary.to_dict('records'),
        "Best_SEO_Keyword": {
            "keyword": best_seo['SEO_Keyword'],
            "search_volume": int(best_seo['SEO_Search_Volume']),
            "difficulty": best_seo['SEO_Difficulty'].round(1)
        },
        "Overall_Avg_ROAS": df_processed['ROAS'].mean().round(2)
    }

def generate_creative_outputs(df_processed: pd.DataFrame) -> dict:
    """
    Generates structured D2C creative assets using the Gemini API based on metrics.
    """
    print("-> Generating AI-powered D2C Creative Outputs...")

    analysis_summary = get_d2c_analysis_summary(df_processed)
    
    # Identify the best platform and keyword from the summary
    best_platform = analysis_summary['Top_3_Platforms_by_ROAS'][0]['Platform']
    best_roas = analysis_summary['Top_3_Platforms_by_ROAS'][0]['Avg_ROAS']
    best_keyword = analysis_summary['Best_SEO_Keyword']['keyword']
    
    data_prompt_summary = json.dumps(analysis_summary, indent=2)

    prompt = f"""
    You are an expert AI Marketing Specialist. Your goal is to generate 3 high-performing creative assets (Headline, Meta Description, PDP Text) for a sustainable D2C brand.
    
    Base your creative strategy ONLY on the following performance metrics and SEO analysis.
    
    METRICS SUMMARY:
    {data_prompt_summary}
    
    STRATEGY:
    1. **Ad Headline (for {best_platform})**: Focus on the platform with the highest ROAS ({best_roas}x) to maximize conversion.
    2. **SEO Meta Description**: Optimize for the best SEO Keyword: "{best_keyword}". This asset is designed to increase click-through rate (CTR) from search results.
    3. **PDP Text (Product Detail Page)**: Write a persuasive copy block that leverages the high **Overall_Avg_ROAS** to imply customer value and drive high-intent conversions.
    
    Your output MUST be a single JSON object that conforms strictly to the provided Pydantic schema for D2CCreativeReport.
    """
    
    # 3. Call the Gemini API with structured output
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=D2CCreativeReport,
            ),
        )
        
        creative_data = json.loads(response.text)
        
        # Ensure output directory exists and save the deliverable
        os.makedirs(os.path.dirname(CREATIVE_OUTPUT_FILE), exist_ok=True)
        with open(CREATIVE_OUTPUT_FILE, 'w') as f:
            json.dump(creative_data, f, indent=4)
            
        print(f"\nDELIVERABLE 5 (Part 3/3): Saved D2C Creative Outputs to {CREATIVE_OUTPUT_FILE}")
        return creative_data

    except Exception as e:
        print(f"An error occurred during LLM generation: {e}")
        return {"error": "LLM generation failed", "details": str(e)}


if __name__ == '__main__':
    try:
        df_processed = pd.read_csv(D2C_PROCESSED_FILE)
        generate_creative_outputs(df_processed)
        
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: D2C data not processed. Please run the following scripts first:\n1. d2c_data_generator.py\n2. metrics_analysis.py\nDetails: {e}")
    except genai.errors.APIError as e:
        print(f"\nCRITICAL ERROR: Gemini API call failed. Check your GEMINI_API_KEY environment variable. Details: {e}")