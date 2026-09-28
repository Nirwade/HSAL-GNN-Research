"""
Shared Ollama client: the low-level call, retry, and JSON-parsing logic that
should be identical across every relation type (prequel, sequel, spin-off, ...).
Relation-specific scripts import this and supply their own prompt text.
"""

import json
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 2


def chat_json(model, prompt_text, timeout=REQUEST_TIMEOUT):
    """Send one prompt, get back (parsed_json_dict, None) on success or
    (None, error_type_name) on failure after retries."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return json.loads(content), None
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return None, type(e).__name__