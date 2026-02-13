import base64
import requests
import json
from pathlib import Path

url = "http://localhost:30002/v1/chat/completions"

# Use a simple test image (can be any PNG/JPG)
with open("test_image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

payload = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": "Describe this image in one sentence."
                }
            ]
        }
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

response = requests.post(url, json=payload)
result = response.json()
print("Image-only output:")
print(result['choices'][0]['message']['content'])