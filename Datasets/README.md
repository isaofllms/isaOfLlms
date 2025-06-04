### Dataset Folders

We have three primary folders that store the datasets produced in this study:

1. **Main Experiments**

   * All datasets and results for the core paper.
   * Includes:

     * The 10 models’ answers to the 30 scenarios.
     * Judge-tags for the pilot LLMs’ 90 answers.
     * Krippendorff’s alpha and Spearman correlation results.
     * Judge-tags for all 10 models’ answers.

2. **System-Prompt Experiment**

   * The 10 models’ answers under two new system prompts (“generic” and “awareness”).
   * Judge-tags (from three judges) for both answer sets.

3. **Temperature Experiment**

   * **Answer datasets** – ten files, each with 30 rows (scenarios) and 40 columns (10 models × 4 temperatures).
   * **Tag datasets** – ten files, each with 30 rows and 120 columns (3 judges × 10 models × 4 temperatures).
   * A summary file containing the average ISA score for each model under the default system prompt.
