"""
clients.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ✅ Set up clients
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

deepinfra_client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)

openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

