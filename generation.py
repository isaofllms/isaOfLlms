"""
# generation.py
"""

from typing import List
import pandas as pd, time
from clients import openai_client, deepinfra_client, openrouter_client
from config import config


# ✅ Model registry: maps model name → (provider, model_id)
model_registry = {

    # OpenAI
    # "GPT-4o-2024-05-13 Answer":     ("openai","gpt-4o-2024-05-13"),
    # "GPT-4o-2024-08-06 Answer":     ("openai","gpt-4o-2024-08-06"),
    # "GPT-4o-2024-11-20 Answer":     ("openai","gpt-4o-2024-11-20"),
    # "GPT-4.1 Answer":               ("openai","gpt-4.1"),
    # "GPT-4.1-mini Answer":          ("openai","gpt-4.1-mini"),
    # "GPT-4.1-nano Answer":          ("openai","gpt-4.1-nano"),


    # OpenRouter (OpenAI)
    "GPT-4o Answer":                ("openrouter","openai/gpt-4o"),
    "GPT-4o-mini Answer":           ("openrouter","openai/gpt-4o-mini"),
    "GPT-4.1 Answer":               ("openrouter","openai/gpt-4.1"),
    "GPT-4.1-mini Answer":          ("openrouter","openai/gpt-4.1-mini"),



    # DeepInfra
    # "Claude-3-7-Sonnet Answer": ("deepinfra", "anthropic/claude-3-7-sonnet-latest"),
    # "Llama-4-Scout Answer":     ("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    # "Mixtral-8x7B Answer":      ("deepinfra", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    # "Gemma-3 Answer":           ("deepinfra", "google/gemma-3-27b-it"),
    # "Mistral-Small-3 Answer":   ("deepinfra", "mistralai/Mistral-Small-24B-Instruct-2501"),
    # "DeepSeek-V3 Answer":       ("deepinfra", "deepseek-ai/DeepSeek-V3-0324"),

    # OpenRouter (qwen)
    "Qwen3-32b Answer":            ("openrouter", "qwen/qwen3-32b"),
    "Qwen3-14b Answer":            ("openrouter", "qwen/qwen3-14b"),
    "Qwen3-8b Answer":             ("openrouter", "qwen/qwen3-8b"),
    "Qwen2.5-7b Answer":             ("openrouter", "qwen/qwen-2.5-7b-instruct"),
    "Qwen2.5-72b Answer":             ("openrouter", "qwen/qwen-2.5-72b-instruct"),

    # OpenRouter (google)
    "Gemini-2.5-Flash Answer":              ("openrouter", "google/gemini-2.5-flash"),
    "Gemini-2.5-Flash-Lite Answer":         ("openrouter", "google/gemini-2.5-flash-lite"),
    "Gemini-2.0-Flash-001 Answer":          ("openrouter", "google/gemini-2.0-flash-001"),
    "Gemini-2.0-Flash-001-Lite Answer":     ("openrouter", "google/gemini-2.0-flash-lite-001"),
    "Gemini-Flash-1.5 Answer":              ("openrouter", "google/gemini-flash-1.5"),
    "Gemini-Flash-1.5-8b Answer":           ("openrouter", "google/gemini-flash-1.5-8b"),
    "Gemini-Pro-1.5 Answer":                ("openrouter", "google/gemini-pro-1.5"),
    "Gemini-2.5-Pro Answer":                ("openrouter", "google/gemini-2.5-pro"),

    "Gemma-3-27b-It Answer":                ("openrouter", "google/gemma-3-27b-it:free"),
    "Gemma-3-4b-It Answer":                 ("openrouter", "google/gemma-3-4b-it:free"),
    "Gemma-3-12b-It Answer":                ("openrouter", "google/gemma-3-12b-it:free"),
    "Gemma-2-27b-It Answer":                ("openrouter", "google/gemma-2-27b-it"),
    "Gemma-3n-2B Answer":                   ("openrouter", "google/gemma-3n-e2b-it:free"),
    "Gemma-3n-4B Answer":                   ("openrouter", "google/gemma-3n-e4b-it"),
    "Gemma-2-9b-It Answer":                 ("openrouter", "google/gemma-2-9b-it"),


    # OpenRouter (mistralai)
    "Mistral-Small-3.2-24b-Instruct Answer":       ("openrouter", "mistralai/mistral-small-3.2-24b-instruct:free"),
    "Mistral-Small-24b-Instruct-2501 Answer":      ("openrouter", "mistralai/mistral-small-24b-instruct-2501:free"), #Mistral-small-3
    "Mistral-Medium-3 Answer":                     ("openrouter", "mistralai/mistral-medium-3"), 
    "Mistral-Small-3.1-24b-Instruct Answer":       ("openrouter", "mistralai/mistral-small-3.1-24b-instruct:free"),
    "Mistral-large-2 Answer":                      ("openrouter", "mistralai/mistral-large-2411"), # NEW

    "Mixtral-8x7B Answer":                         ("openrouter", "mistralai/mixtral-8x7b-instruct"),
    "Mixtral-8x22B Answer":                        ("openrouter", "mistralai/mixtral-8x22b-instruct"),



    # OpenRouter (meta-llama)
    "Llama-3.3-70b-Instruct Answer":   ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "Llama-3.1-70b-Instruct Answer":   ("openrouter", "meta-llama/llama-3.1-70b-instruct"),
    "Llama-3.1-8b-Instruct Answer":    ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    "Llama-3.1-405b-Instruct Answer":  ("openrouter", "meta-llama/llama-3.1-405b-instruct"),
    "Llama-3-70b-Instruct Answer":     ("openrouter", "meta-llama/llama-3-70b-instruct"),
    "Llama-3.2-90b-Instruct Answer":   ("openrouter", "meta-llama/llama-3.2-90b-vision-instruct"),
    "Llama-3.2-1b-Instruct Answer":    ("openrouter", "meta-llama/llama-3.2-1b-instruct"),
    "Llama-3.2-3b-Instruct Answer":    ("openrouter", "meta-llama/llama-3.2-3b-instruct"),
    "Llama-4-Scout Answer":            ("openrouter", "meta-llama/llama-4-scout"),
    "Llama-4-Maverick Answer":         ("openrouter", "meta-llama/llama-4-maverick"),
    "Llama-3-70b Answer":              ("openrouter", "meta-llama/llama-3-70b-instruct"),
    "Llama-3-8b Answer":               ("openrouter", "meta-llama/llama-3-8b-instruct"),
    "Llama-3.1-8b Answer":             ("openrouter", "meta-llama/llama-3.1-8b-instruct"),


    # OpenRouter (cohere)
    "Command-A (Alt) Answer":              ("openrouter", "cohere/command-a"),
    "Command-R-Plus-08-2024 Answer":       ("openrouter", "cohere/command-r-plus-08-2024"),
    "Command-R-Plus-04-2024 Answer":       ("openrouter", "cohere/command-r-plus-04-2024"),
    "Command Answer":                      ("openrouter", "cohere/command"),
    "Command-R-03-2024 Answer":            ("openrouter", "cohere/command-r-03-2024"),

    # OpenRouter (deepseek)
    "DeepSeek-Chat-V3-0324 Answer":    ("openrouter", "deepseek/deepseek-chat-v3-0324:free"),
    "DeepSeek-R1-0528 Answer":         ("openrouter", "deepseek/deepseek-r1-0528:free"),
    "DeepSeek-R1 Answer":              ("openrouter", "deepseek/deepseek-r1:free"),


    # OpenRouter (anthropic)
    "Claude-3.7-Sonnet Answer":    ("openrouter", "anthropic/claude-3.7-sonnet"),
    "Claude-Sonnet-4 Answer":      ("openrouter", "anthropic/claude-sonnet-4"),
    "Claude-3.5-Sonnet Answer":    ("openrouter", "anthropic/claude-3.5-sonnet"),
    "Calude-Opus-4 Answer":        ("openrouter","anthropic/claude-opus-4"),

    # OpenRouter (xAI)
    "Grok-4 Answer": ("openrouter","x-ai/grok-4"),
    "Grok-3 Answer": ("openrouter","x-ai/grok-3"),
    "Grok-3-mini Answer": ("openrouter","x-ai/grok-3-mini"),

    # OpenRouter (Microsoft)
    "Phi-4 Answer":             ("openrouter", "microsoft/phi-4"),
    "Phi-3-mini Answer":        ("openrouter", "microsoft/phi-3-mini-128k-instruct"),
    "Phi-3.5-mini Answer":        ("openrouter", "microsoft/phi-3.5-mini-128k-instruct"),
    "Phi-3-medium Answer":      ("openrouter", "microsoft/phi-3-medium-128k-instruct"),

    # OpenRouter (AI21)
    "Jamba-1.6-large Answer": ("openrouter","ai21/jamba-1.6-large"),
    "Jamba-mini-1.6 Answer": ("openrouter","ai21/jamba-1.6-mini"),




}        

def get_answer(scenario, model_name):
    provider, model_id = model_registry[model_name]
    client = {
        "openai": openai_client,
        "deepinfra": deepinfra_client,
        "openrouter": openrouter_client
    }[provider]

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": scenario}],
            temperature=config["temperature"],
            # max_tokens=config["max_tokens"],
            # top_p=config["top_p"],
            # top_k=config["top_k"]
        )
        return resp.choices[0].message.content

    except Exception as e:
        error_str = str(e).lower()

        # Moderation / flagged cases -> return safe message (no retries)
        if ("moderation" in error_str or 
            "flagged" in error_str or 
            "requires moderation" in error_str or
            "illicit" in error_str or
            "403" in error_str):
            print(f"🚫 Content flagged for model '{model_name}': {scenario[:50]}...")
            return "I'm sorry, but due to security reasons, I can't help you with that."

        # Unknown error -> retry up to 10 times
        print(f"❌ Unknown error for model '{model_name}' with scenario: {scenario[:50]}...\n{e}")
        for attempt in range(1, 11):  # 1..10
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": scenario}],
                    temperature=config["temperature"],
                )
                return resp.choices[0].message.content
            except Exception as e2:
                print(f"   ↻ Retry {attempt}/10 failed for '{model_name}': {e2}")
                # continue trying until attempts exhausted

        # If all retries failed, return 0
        print(f"❌ Exhausted retries for model '{model_name}'.")
        return 0


def generate_all_model_answers(df: pd.DataFrame,
                               model_names: List[str],
                               delay: float = None,
                               backup_file: str = "backup.xlsx") -> pd.DataFrame:
    delay = delay or config["delay"]
    for name in model_names:

        df[name] = df.get(name, pd.Series(dtype=object))
        for i, row in df.iterrows():
            print(f"Model: {name}, Scenario: {row['Scenario']}")
            if not pd.isna(row[name]) or not row.get("Scenario"): continue
            df.at[i, name] = get_answer(row["Scenario"], name)
            time.sleep(delay)
        
         # Save backup after each model completes
        df.to_excel(backup_file, index=False)
        print(f"✅ Backup saved: {backup_file}")


    return df

# ================ System Prompt functions ========================

def get_answer_with_system_prompt(scenario, model_name,system_prompt):
    provider, model_id = model_registry[model_name]
    client = {
        "openai": openai_client,
        "deepinfra": deepinfra_client,
        "openrouter": openrouter_client
    }[provider]

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=
            [
                {
                 "role": "system",
                 "content": system_prompt
                },
                {
                    "role": "user",
                     "content": scenario
                }
            ],
            temperature=config["temperature"],
        )
        return resp.choices[0].message.content

    except Exception as e:
        error_str = str(e).lower()

        # Moderation / flagged cases -> return safe message (no retries)
        if ("moderation" in error_str or 
            "flagged" in error_str or 
            "requires moderation" in error_str or
            "illicit" in error_str or
            "403" in error_str):
            print(f"🚫 Content flagged for model '{model_name}': {scenario[:50]}...")
            return "I'm sorry, but due to security reasons, I can't help you with that."

        # Unknown error -> retry up to 10 times
        print(f"❌ Unknown error for model '{model_name}' with scenario: {scenario[:50]}...\n{e}")
        for attempt in range(1, 11):  # 1..10
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": scenario}],
                    temperature=config["temperature"],
                )
                return resp.choices[0].message.content
            except Exception as e2:
                print(f"   ↻ Retry {attempt}/10 failed for '{model_name}': {e2}")
                # continue trying until attempts exhausted

        # If all retries failed, return 0
        print(f"❌ Exhausted retries for model '{model_name}'.")
        return 0

        

def generate_all_model_answers_with_system_prompt(df: pd.DataFrame,
                               model_names: List[str],
                               delay: float = None,
                               system_prompt : str = "",
                               backup_file: str = "backup.xlsx") -> pd.DataFrame:
    delay = delay or config["delay"]
    for name in model_names:
        # Skip if column exists and has no missing values
        if name in df.columns and df[name].notna().all():
            print(f"⏭️  Skipping {name} (already complete)")
            continue

        df[name] = df.get(name, pd.Series(dtype=object))
        for i, row in df.iterrows():
            print(f"Model: {name}, Scenario: {row['Scenario']}")
            if not pd.isna(row[name]) or not row.get("Scenario"): continue
            df.at[i, name] = get_answer_with_system_prompt(row["Scenario"], name, system_prompt)
            time.sleep(delay)
        
         # Save backup after each model completes
        df.to_excel(backup_file, index=False)
        print(f"✅ Backup saved: {backup_file}")


    return df