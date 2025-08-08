import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from tagging import *
from config import *
from clients import *


# Set up paths
project_dir = Path(os.getenv("PROJECT_DIR", "."))
exp_dir = project_dir / "Datasets" / "Aug25Experiments"
input_file = exp_dir / "Aug25Experiments_65_models_answers.xlsx"
output_file = exp_dir / "Aug25Experiments_65_models_answers_WITH_TAGS.xlsx"
backup_file = exp_dir / "tagging_backup.xlsx"


# Read the generated answers file
print(f"Reading file: {input_file}")
df = pd.read_excel(input_file)

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


# Get all model answer columns (columns ending with "Answer")
model_columns = [col for col in df.columns if col.endswith("Answer")]
print(f"Found {len(model_columns)} model columns:")
for col in model_columns:
    print(f"  - {col}")


# Load the criteria and scenarios file
criteria_scenarios_file = project_dir / "Datasets" / "Criterinos and Scenarios.xlsx"
print(f"Loading criteria and scenarios...")

try:
    df_criteria = pd.read_excel(criteria_scenarios_file)
    print(f"Criteria file shape: {df_criteria.shape}")
    print(f"Criteria file columns: {list(df_criteria.columns)}")
    
    # Check if the number of rows matches
    if len(df_criteria) != len(df):
        print(f"⚠️ WARNING: Row count mismatch!")
        print(f"   Main DataFrame: {len(df)} rows")
        print(f"   Criteria file: {len(df_criteria)} rows")
    
    # Add Criterion and Scenario columns to your main DataFrame
    # Assuming the criteria file has columns named 'Criterion' and 'Scenario'
    # Adjust column names if they're different in your file
    
    if 'Criterion' in df_criteria.columns:
        df['Criterion'] = df_criteria['Criterion']
        print("✅ Added Criterion column")
    else:
        print(f"❌ 'Criterion' column not found. Available columns: {list(df_criteria.columns)}")
        # If the column has a different name, use it:
        # df['Criterion'] = df_criteria['ACTUAL_CRITERION_COLUMN_NAME']
    
    if 'Scenario' in df_criteria.columns:
        df['Scenario'] = df_criteria['Scenario']
        print("✅ Added Scenario column")
    else:
        print(f"❌ 'Scenario' column not found. Available columns: {list(df_criteria.columns)}")
        # If the column has a different name, use it:
        # df['Scenario'] = df_criteria['ACTUAL_SCENARIO_COLUMN_NAME']
    
    # Verify the merge worked
    print(f"\n📊 Updated DataFrame shape: {df.shape}")
    print(f"First Criterion: '{df.iloc[0]['Criterion'] if 'Criterion' in df.columns else 'MISSING'}'")
    print(f"First Scenario: '{df.iloc[0]['Scenario'] if 'Scenario' in df.columns else 'MISSING'}'")
    
except FileNotFoundError:
    print(f"❌ Error: Could not find file {criteria_scenarios_file}")
    print("Please check the file path and name")
except Exception as e:
    print(f"❌ Error loading criteria file: {e}")



# Specify which 3 judges you want to use
selected_judges = ["Gemini-2.0-flash", "Claude-3-7-Sonnet", "Mistral-Small-3"]

print(f"Selected judges: {selected_judges}")
print(f"Models to judge: {len(model_columns)}")
print(f"Scenarios: {len(df)}")
print(f"Total evaluations: {len(df) * len(model_columns) * len(selected_judges)}")



# Start the tagging process
print(f"\nStarting tagging process...")

try:
    df_tagged = tag_all_models_with_selected_judges(
        df=df,
        openrouter_client=openrouter_client,
        model_columns=model_columns,
        explanation=False,  # Set to True if you want explanations
        selected_judges=selected_judges,
        temperature=config["temperature"],
        max_tokens=1000,
        delay=0.0,  # 0 second delay between requests
        backup_file=str(backup_file)
    )
    
    # Save final results
    df_tagged.to_excel(output_file, index=False)
    print(f"\n✅ Tagging completed successfully!")
    print(f"📁 Final results saved to: {output_file}")
    
    # Show summary
    tag_columns = [col for col in df_tagged.columns if "Tag for" in col]
    print(f"\nSummary:")
    print(f"   - Total scenarios: {len(df_tagged)}")
    print(f"   - Total models evaluated: {len(model_columns)}")
    print(f"   - Total judges used: {len(selected_judges)}")
    print(f"   - Total tag columns created: {len(tag_columns)}")
    print(f"   - Expected tag columns: {len(model_columns) * len(selected_judges)}")
    
    # Verify structure
    expected_columns = len(model_columns) * len(selected_judges)
    if len(tag_columns) == expected_columns:
        print(f"✅ Column structure is correct!")
    else:
        print(f"⚠️ Expected {expected_columns} tag columns, got {len(tag_columns)}")
    
    # Show tag distribution for first few columns
    print(f"\nTag distribution (first 3 columns):")
    for col in tag_columns[:3]:
        print(f"   {col}:")
        counts = df_tagged[col].value_counts().sort_index()
        print(f"     {counts.to_dict()}")
    
    # Show all judge columns created
    print(f"\nAll tag columns created:")
    for judge in selected_judges:
        judge_cols = [col for col in tag_columns if col.startswith(judge)]
        print(f"   {judge}: {len(judge_cols)} columns")
        if len(judge_cols) <= 3:  # Show all if 3 or fewer
            for col in judge_cols:
                print(f"      - {col}")
        else:  # Show first 3 and indicate more
            for col in judge_cols[:3]:
                print(f"      - {col}")
            print(f"      - ... and {len(judge_cols)-3} more")

except Exception as e:
    print(f"❌ Error during tagging: {e}")
    print(f"Check backup file: {backup_file}")
    
    # Try to load backup if it exists
    if backup_file.exists():
        print("Loading backup to see progress...")
        df_backup = pd.read_excel(backup_file)
        tag_columns = [col for col in df_backup.columns if "Tag for" in col]
        print(f"✅ Backup contains {len(tag_columns)} tag columns")

print("\nScript completed!")