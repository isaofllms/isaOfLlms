# Dataset Folders and Files

We have three primary folders and three standalone files that store the datasets produced in this study:

##  Folders

1. **Main Experiments**
   * All datasets and results for the core paper.
   * Includes:
     * The 10 models’ answers to the 30 scenarios.
     * The 10 models' tags for the 3 pilot LLMs’ 90 answers.
     * Krippendorff’s alpha results.
     * Judge-tags for all 10 models’ answers.

2. **System-Prompt Experiment**
   * The 10 models’ answers under two new system prompts (“generic” and “awareness”).
   * 3 Judge-tags for both system prompt sets.

3. **Temperature Experiment**
   * **Answer datasets** – ten files, each with 30 rows (scenarios) and 40 columns (10 models × 4 temperatures).
   * **Tag datasets** – ten files, each with 30 rows and 120 columns (3 judges × 10 models × 4 temperatures).
   * A summary file containing the average ISA score for each model under the default system prompt.

##  Standalone Files

1. **`3_pilot_LLMS_answers.xlsx`**
   * Contains the answers of the 3 pilot models (ChatGPT, Gemini, and LLaMA) to the 30 scenarios.
   * Total of 90 answers (30 per model).

2. **`Criterions_and_Scenarios.xlsx`**
   * The full list of 30 ISA-related criteria and their corresponding scenarios used for evaluation.

3. **`Human_Majority_Vote.xlsx`**
   * A 90×1 vector representing the majority vote from human judges for each of the 90 pilot model answers.
   * Ordered by model: first 30 rows for ChatGPT, next 30 for Gemini, last 30 for LLaMA.
