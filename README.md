# The Information Security Awareness of Large Language Models

## Abstract
The popularity of large language models (LLMs) continues to grow, and LLM-based assistants have become ubiquitous.
Information security awareness (ISA) is an important yet underexplored safety aspect of LLMs. 
ISA encompasses LLMs' security knowledge, which has been explored in the past, as well as attitudes and behaviors, which are crucial to LLMs' ability to understand the implicit security context and reject the unsafe requests that are potentially causing the LLM to fail the user. 
We present an automated method for measuring the ISA of LLMs, which covers all 30 security topics in a mobile ISA taxonomy, using realistic scenarios that create tension between implicit security implications and user satisfaction.
Applying this method to leading LLMs, we find that most of the popular models exhibit only medium to low levels of ISA, exposing their users to cybersecurity threats.
Smaller variants of the same model family are significantly riskier, while newer versions show no consistent ISA improvement, suggesting that providers are not actively working toward mitigating this issue. 
These results reveal a widespread vulnerability affecting current LLM deployments: the majority of popular models, and particularly their smaller variants, may systematically endanger users. 
We propose a practical mitigation: incorporating our security awareness instruction into model system prompts to help LLMs better detect and reject unsafe requests.


## Repository Structure

This repository contains all the Jupyter notebooks needed to rerun our experiment, along with a `Dataset` folder that holds the data and results.

### Helper modules

We use four Python modules to improve the modularity and robustness of these experiments:

*config.py* — Centralizes global variables (paths, seeds, flags, experiment settings).

*clients.py* — Adapters for all client/providers (unified call interface, retries/rate-limits).

*generation.py* —

- Maintains a model registry that maps each model_id → provider/client.

- Helper utilities for generating model answers with or without a modified system prompt.

*tagging.py* —

- Helper utilities for tagging model answers (with/without modified system prompt).

- Defines two global lists: (i) an extended judge-selection phase, and (ii) the final tagging phase after judges have been determined.


Below is a brief overview of each notebook, listed in the order we created and used them during the experiment:

**1. Create Dataset With 10 Models**
In this notebook we presented the 30 scenarios to our 10 LLM contenster models and saved their answers (temperature = 0).

**2. Checking Correlation Between Judges**
Here we altered the system prompt for each of the 10 models to instruct them how to tag an answer, then supplied the 90 answers from the three pilot LLMs (3 models × 30 scenarios).
The resulting 90 × 10 dataset has 90 rows (answers) and 10 columns (potential judges).
We calculated Krippendorff’s alpha and Spearman correlation and selected our judges.

**3. 3 New Judges’ Tags on 10 Models’ Answers**
In this notebook we altered the system prompt for each LLM acting as a judge (three judges in total) and gave them the 30 scenarios × 10 models answers to tag.
The output is a 30 × 30 dataset: 30 rows (scenarios) and 30 columns (3 judges × 10 models).

**4. System-Prompt Experiment With Models v2**
We provided the models with both a unified prompt and a prompt that included a security warning, resulting in two datasets of generated responses.
Then, these responses were labeled by the selected judges.
Then, we computed the ISA score of each model for every system prompt.

**5. Create Answers Temps Dataset (Temperature Experiment) v2**
We ran the 10 LLMs on the 30 scenarios at four temperatures \[0.25, 0.5, 0.75, 1], generating 10 samples each—
a total of 10 × 30 × 4 × 10 = 12,000 answers.

**6. Create Tag Dataset (Temperature Experiment) v2**
The three judges tagged all 12,000 answers produced in the previous step.

**7. Temperature Experiment v2**
Using the judges’ tags, we calculated the average ISA score for each model at every temperature and graphed the results, comparing each temperature’s average score to that at temperature 0.

**8. Aug 25 Experiments**

After completing our baseline experiments, we extended the study as follows (default system prompt unless noted):

A. Scale-up: +55 models
We added 55 additional models to the benchmark—demonstrating the robustness and dynamic nature of our framework. For each model we generated responses and had judges compute ISA scores.

B. Visualization & statistics
We aggregated results, produced radar charts, and performed statistical analyses on pairs and trios of models, stratified by version and size.

C. Prompt-format ablation
We expanded the prompt experiment by allowing models to answer with and without an initial explanation, then measured performance differences.

D. System-prompt ablation
We expanded the system-prompt study by selecting 30 diverse models and evaluating their performance under multiple system prompts.



