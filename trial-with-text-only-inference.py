import requests
import json

url = "http://localhost:30002/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "What is 2+2? Give a short answer."
        }
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

response = requests.post(url, json=payload)
result = response.json()
print("Text-only output:")
print(result['choices'][0]['message']['content'])