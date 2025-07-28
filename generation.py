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

    # OpenRouter
    "Command-A Answer":         ("openrouter", "cohere/command-r-plus"),
    "Gemini-2.0-flash Answer":  ("openrouter", "google/gemini-pro")
}

def get_answer(scenario: str, model_name: str) -> str:
    provider, model_id = model_registry[model_name]
    client = {
      "openai": openai_client,
      "deepinfra": deepinfra_client,
      "openrouter": openrouter_client
    }[provider]

    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role":"user","content":scenario}],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )
    return resp.choices[0].message.content

def generate_all_model_answers(df: pd.DataFrame,
                               model_names: List[str],
                               delay: float = None) -> pd.DataFrame:
    delay = delay or config["delay"]
    for name in model_names:
        df[name] = df.get(name, pd.Series(dtype=object))
        for i, row in df.iterrows():
            if not pd.isna(row[name]) or not row.get("Scenario"): continue
            df.at[i, name] = get_answer(row["Scenario"], name)
            time.sleep(delay)
    return df