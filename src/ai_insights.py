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
    """The full report structure."""
    summary: str = Field(..., description="An executive summary of all key findings for the VP of Marketing.")
    insights: List[MarketInsight]

# --- LLM INSIGHTS GENERATION ---

def generate_insights(df: pd.DataFrame) -> dict:
    """
    Generates structured market intelligence insights using the Gemini API.
    """
    print("-> Generating AI-powered market intelligence insights...")

    # 1. Prepare data summary with statistical validation
    
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce').fillna(0)

    top_categories = df.groupby('Category')['Installs'].sum().nlargest(5)
    avg_rating_by_type = df.groupby('Type')['Rating'].mean()
    
    install_median = df['Installs'].median()
    df['Install_Segment'] = np.where(df['Installs'] > install_median, 'High_Install', 'Low_Install')
    
    # Handle NaN in Avg_Sentiment_Polarity before grouping
    df_temp = df.dropna(subset=['Avg_Sentiment_Polarity'])
    avg_sentiment_by_segment = df_temp.groupby('Install_Segment')['Avg_Sentiment_Polarity'].mean()

    # Calculate correlation, handling NaNs
    rating_review_corr = df['Rating'].corr(df['Review_Count'])

    data_summary = f"""
    Overall Market Data Snapshot (Google Play + Mock App Store):
    - Total App Records: {len(df)}
    - Top 5 Categories by Total Installs:\n{top_categories.to_string()}
    - Average Rating (Free vs. Paid):\n{avg_rating_by_type.to_string()}
    - **Key User Experience Metric**: Average Sentiment Polarity (0 to 1, higher is better):
        - High_Install Apps: {avg_sentiment_by_segment.get('High_Install', 'N/A'):.3f}
        - Low_Install Apps: {avg_sentiment_by_segment.get('Low_Install', 'N/A'):.3f}
    - Key Stat: Correlation between App Rating and Review_Count: {rating_review_corr:.3f}
    """
    
    # 2. Define the prompt
    prompt = f"""
    You are an expert Applied AI Engineer building a market intelligence report for executive leadership.
    Analyze the following market data summary to generate 3 to 5 highly actionable, structured market insights.
    
    Focus your analysis on:
    1. **Growth Opportunities**: Which categories or segments (e.g., Free vs. Paid) show the highest potential for growth.
    2. **Retention/Quality**: Use the Avg Sentiment Polarity to identify quality gaps or strong user loyalty.
    3. **Strategy**: Recommend specific actions (e.g., pricing shift, quality focus) based on the findings.
    
    DATA SUMMARY:
    {data_summary}
    
    Crucially, assign a **confidence_score** (0.0 to 1.0) for each insight based on the strength of the statistical support provided in the summary. Your analysis must be entirely driven by the provided summary statistics.
    
    Your output MUST be a single JSON object that conforms strictly to the provided Pydantic schema for InsightsReport.
    """
    
    # 3. Call the Gemini API with structured output
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InsightsReport,
            ),
        )
        
        insights_data = json.loads(response.text)
        
        # 4. Save the deliverable
        with open(INSIGHTS_FILE, 'w') as f:
            json.dump(insights_data, f, indent=4)
            
        print(f"\nDELIVERABLE 2: Saved structured insights JSON to {INSIGHTS_FILE}")
        return insights_data

    except Exception as e:
        print(f"\nCRITICAL ERROR: An error occurred during LLM generation: {e}")
        return {"error": "LLM generation failed", "details": str(e)}


if __name__ == '__main__':
    try:
        combined_data = pd.read_csv(COMBINED_FILE, dtype={'Installs': 'Int64'})
        generate_insights(combined_data)
        
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: The combined data file was not found. Please ensure you ran 'data_pipeline.py' successfully first to create '{COMBINED_FILE}'.\nDetails: {e}")
    except genai.errors.APIError as e:
        print(f"\nCRITICAL ERROR: Gemini API call failed. Check your GEMINI_API_KEY configuration. Details: {e}")
