# %% [markdown]
# ## Scoring 3 pilot LLMs answers WITH explanation prompt

# %%
from pathlib import Path
from clients import openrouter_client
import tagging
import pandas as pd
import os
from config import config


# %%
# Directories
project_dir  = Path(os.getenv("PROJECT_DIR", "."))
exp_dir      = project_dir / "Datasets" / "Aug25Experiments"
input_file       = project_dir / "Datasets" / "3_pilot_LLMS_answers.xlsx"
output_file      = exp_dir / "Aug25Experiments_3_judged_pilot_answers_tagged_WITH_explanation_prompt.xlsx"
backup_file = exp_dir / "Aug25Experiments_judging_backup_WITH_explanation.xlsx"
exp_dir.mkdir(parents=True, exist_ok=True)



# %%
# Read the data
df = pd.read_excel(input_file)


# %%

# Models to evaluate (the column names in your Excel file)
contester_models = [
    "GPT-4-o-mini Answer",
    "Gemini-1.5-flash Answer", 
    "llama-3.1-70b-versatile Answer",    
]

# Get judge functions with explanation enabled
judge_functions_WITH_explanation = tagging.get_judge_functions( 
    openrouter_client,
    explanation=True,
    temperature=config["temperature"],
    max_tokens=3000
)

# Run the judging process
df_judged_WITH_explanation = tagging.tag_answers_with_judges(
    df=df,
    judge_functions=judge_functions_WITH_explanation,
    contester_models=contester_models,
    backup_file=str(backup_file)
)

# Save final results
df_judged_WITH_explanation.to_excel(output_file, index=False)
print(f"✅ Judging complete! Results saved to: {backup_file}")

# %% [markdown]
# ## Scoring 3 pilot LLMs answers without explanation prompt

# %%
from pathlib import Path
from clients import openai_client, openrouter_client
import tagging
import pandas as pd
import os
from config import config


# # %%
# # Directories
# project_dir  = Path(os.getenv("PROJECT_DIR", "."))
# exp_dir      = project_dir / "Datasets" / "Aug25Experiments"
# input_file       = project_dir / "Datasets" / "3_pilot_LLMS_answers.xlsx"
# output_file      = exp_dir / "Aug25Experiments_3_judged_pilot_answers_tagged_WITHOUT_explanation_prompt.xlsx"
# backup_file = exp_dir / "Aug25Experiments_judging_backup-WITHOUT_explanation.xlsx"
# exp_dir.mkdir(parents=True, exist_ok=True)



# # %%
# # Read the data
# df = pd.read_excel(input_file)



# # %%
# df.columns

# # %%
# # Models to evaluate
# contester_models = [
#     "GPT-4-o-mini Answer",
#     "Gemini-1.5-flash Answer", 
#     "llama-3.1-70b-versatile Answer",    
# ]

# # Get judge functions with explanation enabled
# judge_functions_WTIHOUT_explanation = tagging.get_judge_functions( 
#     openrouter_client,
#     explanation=False,
#     temperature=config["temperature"],
#     max_tokens=5
# )

# # Run the judging process
# df_judged_WITHOUT_explanation = tagging.tag_answers_with_judges(
#     df=df,
#     judge_functions=judge_functions_WTIHOUT_explanation,
#     contester_models=contester_models,
#     backup_file=str(backup_file)
# )

# # Save final results
# df_judged_WITHOUT_explanation.to_excel(output_file, index=False)
# print(f"✅ Judging complete! Results saved to: {output_file}")