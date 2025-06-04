# The Information Security Awareness of Large Language Models

## Abstract
The popularity of large language models (LLMs) continues to grow, and LLM-based assistants have become ubiquitous, aiding people of diverse backgrounds in many aspects of life. Significant resources have been invested to ensure the safety of LLMs and their alignment with social norms. Information security awareness (ISA) is an important safety aspect. ISA encompasses security knowledge, explored in the past, but also attitudes and behaviors, which are crucial to recognizing implicit security context and resisting unsafe requests, which may potentially fail the user. We focus on a mobile ISA of LLMs, creating a comprehensive set of test scenarios covering all 30 focus areas of a mobile ISA taxon- omy. Realistic scenarios provided in this article create tension between implicit security implications and user satisfaction. Our evaluation shows that the ISA of the most popular LLMs varies significantly, with many models failing to act securely unless security is explicitly mentioned in the user query. We also observe that the system prompt can greatly affect ISA. Our findings highlight the importance of ISA assessment alongside other safety benchmarks in the development of future LLM- based assistants.


## Repository Structure

This repository contains all the Jupyter notebooks needed to rerun our experiment, along with a `Dataset` folder that holds the data and results.
Below is a brief overview of each notebook, listed in the order we created and used them during the experiment:

**1. Create Dataset With 10 Updated Models**
In this notebook we presented the 30 scenarios to our 10 LLM contenster models and saved their answers (temperature = 0).

**2. Checking Correlation Between Judges (Updated)**
Here we altered the system prompt for each of the 10 models to instruct them how to tag an answer, then supplied the 90 answers from the three pilot LLMs (3 models × 30 scenarios).
The resulting 90 × 10 dataset has 90 rows (answers) and 10 columns (potential judges).
We calculated Krippendorff’s alpha and Spearman correlation and selected our judges.

**3.3 New Judges’ Tags on 10 Models’ Answers**
In this notebook we altered the system prompt for each LLM acting as a judge (three judges in total) and gave them the 30 scenarios × 10 models answers to tag.
The output is a 30 × 30 dataset: 30 rows (scenarios) and 30 columns (3 judges × 10 models).

**4. System-Prompt Experiment With Updated Models**
We computed the ISA score of each model for every system prompt.

**5. Create Answers Temps Dataset (Temperature Experiment)**
We ran the 10 LLMs on the 30 scenarios at four temperatures \[0.25, 0.5, 0.75, 1], generating 10 samples each—
a total of 10 × 30 × 4 × 10 = 12,000 answers.

**6. Create Tag Dataset (Temperature Experiment)**
The three judges tagged all 12,000 answers produced in the previous step.

**7. Temperature Experiment**
Using the judges’ tags, we calculated the average ISA score for each model at every temperature and graphed the results, comparing each temperature’s average score to that at temperature 0.




