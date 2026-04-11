import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"


def generate_recommendation(disease: str, confidence: float):

    prompt = f"""
You are a medical assistant AI.

A skin disease has been detected.

Disease: {disease}
Confidence: {confidence:.2f}

Return ONLY valid JSON (no extra text):

{{
  "recommendations": "simple explanation of the disease",
  "next_steps": "what the patient should do next",
  "tips": "home care tips in simple English"
}}

Rules:
- No markdown
- No explanation
- Only JSON output
- Use simple English
- No medical prescriptions
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        # Ollama response text
        content = result["message"]["content"]

        # Convert string → JSON
        parsed = json.loads(content)

        return parsed

    except Exception as e:
        return {
            "recommendations": "Error generating response",
            "next_steps": "Try again later",
            "tips": f"Ollama error: {str(e)}"
        }