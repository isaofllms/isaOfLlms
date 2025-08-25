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

This repository contains all artifacts required to reproduce our USENIX Security paper on measuring and analyzing the Informatino Security Awareness (ISA) of Large Language Models.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Helper Modules](#helper-modules)
- [Pre-existing Datasets](#pre-existing-datasets)
- [Experimental Pipeline](#experimental-pipeline)
- [Requirements](#requirements)

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create a `.env` file with your API keys and settings:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and project directory path
   ```

3. **Run experiments:**
   Execute notebooks sequentially (1-8) - **order is crucial** as each depends on the previous ones.

## 📁 Repository Structure

```
├── Dataset/                 # All experimental data and results
├── notebooks/              # Jupyter notebooks (1-8)
├── config.py               # Global experiment settings
├── clients.py              # Provider adapters and unified API interface
├── generation.py           # Model registry and response generation utilities
├── tagging.py              # Judge selection and tagging utilities
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠 Helper Modules

Our codebase is organized into four main modules for improved modularity and robustness:

### `config.py`
Centralizes global variables including:
- Model configurations (temperature, max_tokens)

### `clients.py`
Unified interface for multiple LLM providers:
- Adapters for OpenAI, Anthropic, Google, etc.

### `generation.py`
Model response generation utilities:
- Model registry mapping model_id → provider/client
- Support for system prompt modifications
- Batch generation capabilities

### `tagging.py`
Judge evaluation and scoring utilities:
- Judge selection algorithms
- Answer tagging with configurable judge sets
- Support for both individual and batch evaluation

## 📊 Pre-existing Datasets

The repository includes three foundational datasets:

1. **Criteria and Scenarios** (`Criterinos and Scenarios`)
   - 30 carefully crafted security scenarios
   - One scenario per sub-focus area from Mobile Security Taxonomy
   - Created through extensive manual review process

2. **Pilot LLM Answers** (`3_pilot_LLMS_answers`)
   - Responses from 3 pilot models (GPT, Gemini, Llama) 
   - 90 total answers (3 models × 30 scenarios)
   - Baseline for judge evaluation

3. **Human Majority Vote** (`Human_Majority_Vote`)
   - Gold standard human evaluations
   - 90 tagged answers with majority vote from 3 human judges
   - Used for judge validation and correlation analysis

## 🧪 Experimental Pipeline

Run notebooks **sequentially** - each builds on previous results:

### Core Experiments (Notebooks 1-4)

**1. Create Dataset With 10 Models**
- Generate responses from 10 contester LLMs
- 30 scenarios × 10 models = 300 answers
- Temperature = 0 (deterministic)

**2. Checking Correlation Between Judges** 
- Convert 10 models into potential judges
- Tag pilot LLM answers (90 answers × 10 judges = 900 tags)
- Calculate Krippendorff's α and Spearman correlation
- Select optimal judge combination

**3. 3 New Judges' Tags on 10 Models' Answers**
- Selected judges evaluate all 10 model responses
- Output: 30 scenarios × 30 tags (3 judges × 10 models)
- Foundation for ISA score calculation

**4. System-Prompt Experiment With Models v2**
- Test models with different system prompts:
  - Unified (generic) prompt (baseline)
  - Security-aware prompt (with warnings)
- Compare ISA scores across prompt conditions

### Scale Experiments (Notebooks 5-7)

**5. Create Answers Temps Dataset (Temperature Experiment) v2**
- Generate responses at 4 temperature levels: [0.25, 0.5, 0.75, 1.0]
- 10 samples per condition
- Total: 10 models × 30 scenarios × 4 temps × 10 samples = **12,000 answers**

**6. Create Tag Dataset (Temperature Experiment) v2**
- 3 judges evaluate all 12,000 temperature-varied responses
- Output: **36,000 total tags** (3 judges × 12,000 answers)

**7. Temperature Experiment v2**
- Calculate average ISA scores per temperature
- Statistical analysis and visualization
- Compare temperature effects across models

### Extended Analysis (Notebook 8)

**8. Aug 25 Experiments - Scale-up and Analysis**

**A. Scale-up: +55 Models**
- Extended benchmark to 65 total models
- Demonstrates framework scalability and robustness

**B. Visualization & Statistics**
- Comprehensive radar charts and statistical analyses
- Pairwise and trio model comparisons
- Stratified analysis by model family and size

**C. Prompt-format Ablation**
- Test explanation vs. direct answer formats
- Measure impact on security assistance behavior

**D. System-prompt Ablation**
- Evaluate 30 diverse models across multiple system prompts
- Comprehensive prompt sensitivity analysis

## 📋 Requirements

- Python 3.8+
- ~2GB free disk space for datasets
- API keys for model providers (OpenAI, Anthropic, Google, etc.)
- Estimated runtime: 36-120 hours for full reproduction (depnading if running temperature experiment as well)

### System Requirements
- Modern CPU (reproducible on single CPU)
- 8GB RAM recommended
- Internet connection for API calls

## 🔄 Reproduction Notes

- Notebooks include progress tracking and backup mechanisms
- Each notebook saves intermediate results for robustness
- Full end-to-end reproduction generates all paper figures and tables


---

**⚠️ Important:** This is an anonymous submission repository. Upon paper acceptance, we will migrate to a permanent repository with full attribution and long-term Zenodo archival.



