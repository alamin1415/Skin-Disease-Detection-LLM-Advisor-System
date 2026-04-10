import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"   # your installed model


def generate_recommendation(disease: str, confidence: float):

    prompt = f"""
You are a medical assistant AI.

A skin disease has been detected from an image analysis system.

Disease: {disease}
Confidence: {confidence:.2f}

Give:
1. Simple explanation of the disease
2. Possible causes
3. Care instructions at home
4. When to see a doctor

Rules:
- Use simple English
- Make it easy for patients to understand
- Do NOT give dangerous medical prescriptions
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

        # safety check
        if "message" in result and "content" in result["message"]:
            return {
                "recommendation": result["message"]["content"]
            }

        return {
            "recommendation": "Invalid response from LLM"
        }

    except Exception as e:
        return {
            "recommendation": f"Ollama error: {str(e)}"
        }