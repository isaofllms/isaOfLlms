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
    "GPT-4.1-mini Answer":      ("openai", "gpt-4.1-mini"),

    # DeepInfra
    "Claude-3-7-Sonnet Answer": ("deepinfra", "anthropic/claude-3-7-sonnet-latest"),
    "Llama-4-Scout Answer":     ("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    "Mixtral-8x7B Answer":      ("deepinfra", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "Phi-4 Answer":             ("deepinfra", "microsoft/phi-4"),
    "Gemma-3 Answer":           ("deepinfra", "google/gemma-3-27b-it"),
    "Mistral-Small-3 Answer":   ("deepinfra", "mistralai/Mistral-Small-24B-Instruct-2501"),
    "DeepSeek-V3 Answer":       ("deepinfra", "deepseek-ai/DeepSeek-V3-0324"),

    # OpenRouter (qwen)
    "Qwen3-32b Answer":            ("openrouter", "qwen/qwen3-32b"),
    "Qwen3-14b Answer":            ("openrouter", "qwen/qwen3-14b"),
    "Qwen3-8b Answer":             ("openrouter", "qwen/qwen3-8b"),

    # OpenRouter (google)
    "Gemini-2.5-Flash Answer":         ("openrouter", "google/gemini-2.5-flash"),
    "Gemini-2.0-Flash-001 Answer":     ("openrouter", "google/gemini-2.0-flash-001"),
    "Gemini-Flash-1.5 Answer":         ("openrouter", "google/gemini-flash-1.5"),
    "Gemini-Flash-1.5-8b Answer":      ("openrouter", "google/gemini-flash-1.5-8b"),
    "Gemini-Pro-1.5 Answer":           ("openrouter", "google/gemini-pro-1.5"),
    "Gemma-3-27b-It Answer":           ("openrouter", "google/gemma-3-27b-it:free"),
    "Gemma-3-4b-It Answer":            ("openrouter", "google/gemma-3-4b-it:free"),
    "Gemma-3-12b-It Answer":           ("openrouter", "google/gemma-3-12b-it:free"),

    # OpenRouter (mistralai)
    "Mistral-Small-3.2-24b-Instruct Answer":       ("openrouter", "mistralai/mistral-small-3.2-24b-instruct:free"),
    "Mistral-Small-24b-Instruct-2501 Answer":      ("openrouter", "mistralai/mistral-small-24b-instruct-2501:free"),
    "Mistral-Small-3.1-24b-Instruct Answer":       ("openrouter", "mistralai/mistral-small-3.1-24b-instruct:free"),

    # OpenRouter (meta-llama)
    "Llama-3.3-70b-Instruct Answer":   ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "Llama-3.1-70b-Instruct Answer":   ("openrouter", "meta-llama/llama-3.1-70b-instruct"),
    "Llama-3.1-8b-Instruct Answer":    ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    "Llama-3.1-405b-Instruct Answer":  ("openrouter", "meta-llama/llama-3.1-405b-instruct"),
    "Llama-3-70b-Instruct Answer":     ("openrouter", "meta-llama/llama-3-70b-instruct"),

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
            max_tokens=config["max_tokens"]
        )
        return resp.choices[0].message.content

    except Exception as e:
        error_str = str(e)
        
        # Check specifically for content moderation/flagging errors
        if ("moderation" in error_str.lower() or 
            "flagged" in error_str.lower() or 
            "requires moderation" in error_str.lower() or
            "illicit" in error_str.lower()):
            print(f"🚫 Content flagged for model '{model_name}': {scenario[:50]}...")
            return 2
        
        # Keep existing 403 handling for other permission issues
        elif "403" in error_str:
            print(f"⚠️  403 Error for model '{model_name}' with scenario: {scenario}")
            return 2
        else:
            print(f"❌ Unknown error for model '{model_name}' with scenario: {scenario[:50]}...\n{e}")
            return 0

def generate_all_model_answers(df: pd.DataFrame,
                               model_names: List[str],
                               delay: float = None,
                               backup_file: str = "backup.xlsx") -> pd.DataFrame:
    delay = delay or config["delay"]
    for name in model_names:
        # Skip if column exists and has no missing values
        if name in df.columns and df[name].notna().all():
            print(f"⏭️  Skipping {name} (already complete)")
            continue

        df[name] = df.get(name, pd.Series(dtype=object))
        for i, row in df.iterrows():
            if i != 4:
                continue
            print(f"Model: {name}, Scenario: {row['Scenario']}")
            if not pd.isna(row[name]) or not row.get("Scenario"): continue
            df.at[i, name] = get_answer(row["Scenario"], name)
            time.sleep(delay)
        
         # Save backup after each model completes
        df.to_excel(backup_file, index=False)
        print(f"✅ Backup saved: {backup_file}")


    return df