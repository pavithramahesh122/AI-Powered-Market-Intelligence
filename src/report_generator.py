import json
import os
import pandas as pd
from datetime import datetime

INSIGHTS_FILE = 'reports/insights.json'
REPORT_FILE = 'reports/executive_report.md' # Renamed output file to match your input
COMBINED_DATA_FILE = 'data/processed/apps_combined.csv'


def generate_executive_report():
    """Loads the insights JSON and converts it into a structured Markdown report, including metadata."""
    print("-> Starting Executive Report generation...")
    
    try:
        with open(INSIGHTS_FILE, 'r') as f:
            insights_data = json.load(f)

        # Load data file to get basic metadata (Total Apps, Date, Source)
        df = pd.read_csv(COMBINED_DATA_FILE, dtype={'Installs': 'Int64'})
        total_records = len(df)
        data_source = "Google Play Store (Kaggle) & Mock App Store"
        report_date = datetime.now().strftime("%B %d, %Y")
            
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Insights file or combined data file not found. Run previous steps first.")
        return
    except json.JSONDecodeError:
        print(f"CRITICAL ERROR: Insights file is corrupted or empty. Check {INSIGHTS_FILE}.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    report_content = []
    
    # 1. Title and Metadata
    report_content.append("# 🎯 AI-Powered Market Intelligence Executive Report")
    report_content.append(f"\n**Date:** {report_date}")
    report_content.append(f"**Source Data:** {data_source}")
    report_content.append(f"**Total Apps Analyzed:** {total_records}")
    report_content.append("**AI Model:** Gemini 2.5 Flash (Structured Output)")
    report_content.append("\n---")
    
    # 2. Executive Summary
    report_content.append("## 📄 Executive Summary")
    report_content.append(f"**Objective:** This report summarizes key growth, quality, and strategic opportunities identified through a unified analysis of app market data, validated by LLM-driven confidence scores.")
    summary = insights_data.get('summary', 'No summary provided by AI model.')
    report_content.append(f"\n> **Key Takeaway:** {summary}\n")
    report_content.append("---")

    # 3. Detailed Market Insights
    report_content.append("## 📈 Detailed Market Insights & Strategic Recommendations")
    
    insights = insights_data.get('insights', [])
    for i, insight in enumerate(insights):
        # Format confidence score as percentage
        confidence_percent = f"{int(insight.get('confidence_score', 0) * 100)}%"

        report_content.append(f"\n### {insight.get('insight_id', f'MI-{i+1:03d}')}: {insight.get('title')}")
        report_content.append(f"**Finding:** {insight.get('finding')}")
        report_content.append(f"**Confidence Score:** `{insight.get('confidence_score', 0.0):.2f}` / 1.0 (Statistical Confidence: **{confidence_percent}**)\n")
        
        # Data Support Section
        report_content.append("*Data Support*:")
        report_content.append(f"```text\n{insight.get('data_support')}\n```")
        
        # Recommendations Section
        report_content.append("**Actionable Recommendations:**")
        
        recommendations = insight.get('recommendations', [])
        for rec in recommendations:
            priority = rec.get('priority', 'UNKNOWN').upper()
            report_content.append(f"- **[{priority} Priority]**: {rec.get('action')}")
        
        report_content.append("---")

    final_report = "\n".join(report_content)
    
    # Save the deliverable
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_report)
        
    print(f"\nDELIVERABLE 3: Saved Executive Report to {REPORT_FILE}")

if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    generate_executive_report()
