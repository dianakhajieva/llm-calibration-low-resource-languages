import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly:\n\nANSWER: A\nCONFIDENCE: 90"
        }
    ],
    temperature=0,
    max_tokens=50,
)

print("=" * 60)
print("LLAMA RESPONSE")
print("=" * 60)

print(response.choices[0].message.content)