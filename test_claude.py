from src.models import get_provider
from src.prompts import build_prompt
from src.parse import parse_response

import pandas as pd


# Load one sample question
df = pd.read_csv("data/sample/sample_500.csv")

row = df.iloc[0]


# Build prompt
prompt = build_prompt(
    passage=row["en_passage"],
    question=row["en_question"],
    a1=row["en_a1"],
    a2=row["en_a2"],
    a3=row["en_a3"],
    a4=row["en_a4"],
)


# Create Claude provider
provider = get_provider(
    "anthropic",
    "claude-sonnet-4-6"
)


# Generate response
response = provider.generate(prompt)


print("=" * 60)
print("RAW RESPONSE")
print("=" * 60)

print(response)


parsed = parse_response(response)


print()
print("=" * 60)
print("PARSED RESPONSE")
print("=" * 60)

print(parsed)
