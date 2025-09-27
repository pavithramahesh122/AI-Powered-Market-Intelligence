import pandas as pd
import numpy as np
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

COMBINED_FILE = 'data/processed/apps_combined.csv'
INSIGHTS_FILE = 'reports/insights.json'


GEMINI_API_KEY = 'AIzaSyDNp9LU-lcyUg8ry_ZjldW07KVkmEngIKM' 


client = genai.Client(api_key=GEMINI_API_KEY) 


class Recommendation(BaseModel):
    """A concrete, actionable step based on the insight."""
    priority: str = Field(..., description="High, Medium, or Low priority.")
    action: str = Field(..., description="The specific, actionable recommendation.")

class MarketInsight(BaseModel):
    """A single, data-driven market intelligence insight."""
    insight_id: str = Field(..., description="Unique ID (e.g., MI-001).")
    title: str = Field(..., description="A concise title for the insight.")
    finding: str = Field(..., description="The core observation derived from the data.")
    confidence_score: float = Field(..., description="Statistical confidence (0.0 to 1.0) in the finding, derived from the data support.")
    data_support: str = Field(..., description="Specific data points or metrics supporting the finding.")
    recommendations: List[Recommendation]


class InsightsReport(BaseModel):
    """The final structured report containing multiple market intelligence insights."""
    report_summary: str = Field(..., description="An executive summary of the most critical findings and market trends.")
    key_metrics_snapshot: str = Field(..., description="A short summary of the key aggregated metrics from the dataset (e.g., 'Average rating is 4.3, 75% of top apps are free, etc.').")
    insights: List[MarketInsight]

def prepare_data_summary(df: pd.DataFrame) -> str:
    """Generates a text summary of the DataFrame to feed into the LLM."""
    
    # Analyze key categories and their performance
    category_summary = df.groupby('Category').agg(
        Total_Installs=('Installs', 'sum'),
        Avg_Rating=('Rating', 'mean'),
        Count=('Name', 'count')
    ).sort_values(by='Total_Installs', ascending=False).head(5).reset_index()
    
    # Analyze review polarity
    polarity_summary = df.groupby('Category')['Avg_Sentiment_Polarity'].mean().sort_values(ascending=False).head(5).reset_index()
    
    # App type distribution
    type_distribution = df['Type'].value_counts(normalize=True).mul(100).to_dict()
    
    # Price and rating correlation (simple check)
    paid_avg_rating = df[df['Type'] == 'Paid']['Rating'].mean()
    free_avg_rating = df[df['Type'] == 'Free']['Rating'].mean()
    
    # Create the detailed prompt data
    data_summary = f"""
    --- DATA SNAPSHOT ---
    Total Apps Analyzed: {len(df)}
    Data Sources: {', '.join(df['Source'].unique())}
    
    --- CATEGORY PERFORMANCE (Top 5 by Installs) ---
    {category_summary.to_string(index=False)}

    --- AVERAGE SENTIMENT POLARITY (Top 5 Categories) ---
    Note: Values closer to 1.0 are positive, -1.0 are negative.
    {polarity_summary.to_string(index=False)}
    
    --- APP TYPE DISTRIBUTION ---
    Free Apps: {type_distribution.get('Free', 0):.2f}%
    Paid Apps: {type_distribution.get('Paid', 0):.2f}%

    --- RATING COMPARISON ---
    Average Rating (Paid): {paid_avg_rating:.2f}
    Average Rating (Free): {free_avg_rating:.2f}
    """
    
    return data_summary

def generate_insights(df: pd.DataFrame) -> dict:
    """
    Generates structured market intelligence insights using the Gemini API.
    """
    data_summary = prepare_data_summary(df)
    
    system_prompt = """
    You are a world-class Market Intelligence Analyst. Your task is to analyze the provided raw data summary
    of a combined Google Play and Mock App Store dataset. Based strictly on the data provided,
    generate a detailed, structured Insights Report.

    The report MUST contain:
    1. An executive summary.
    2. A snapshot of key metrics.
    3. At least 3 unique, actionable Market Insights (MI).
    
    Each Market Insight must include:
    - A specific finding directly supported by the data.
    - A confidence score (0.0 to 1.0) reflecting the statistical clarity of the finding.
    - 2-3 concrete, prioritized recommendations for product strategy or marketing.
    
    Focus areas for insights should include:
    - The correlation between Install volume and Sentiment/Rating.
    - Strategic recommendations for 'Paid' vs 'Free' apps.
    - Opportunities in top/underperforming categories.
    """

    user_prompt = f"""
    Please generate the structured Market Insights Report based on the following data snapshot:
    
    {data_summary}
    """
    
    prompt = [
        {"role": "user", "parts": [{"text": user_prompt}]}
    ]
    
    print("-> Calling Gemini API to generate structured market insights...")
    print(f"System Prompt Size: {len(system_prompt)} characters.")
    
    # 3. Call the Gemini API with structured output
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InsightsReport,
                system_instruction=system_prompt,
            ),
        )
        
        insights_data = json.loads(response.text)
        
        # 4. Save the deliverable
        # Ensure output directory exists before saving (though data_pipeline already did this)
        os.makedirs(os.path.dirname(INSIGHTS_FILE), exist_ok=True)
        with open(INSIGHTS_FILE, 'w') as f:
            json.dump(insights_data, f, indent=4)
            
        print(f"\nDELIVERABLE 2: Saved structured insights JSON to {INSIGHTS_FILE}")
        return insights_data

    except Exception as e:
        print(f"\nCRITICAL ERROR: An error occurred during LLM generation. Please check your API key and network connection. Error: {e}")
        return {"error": "LLM generation failed", "details": str(e)}


if __name__ == '__main__':
    try:
        import os 
        combined_data = pd.read_csv(COMBINED_FILE, dtype={'Installs': 'Int64'})
        generate_insights(combined_data)
        
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: The combined data file was not found. Please ensure you ran 'data_pipeline.py' successfully first to create '{COMBINED_FILE}'.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: An unexpected error occurred in the main script flow: {e}")
