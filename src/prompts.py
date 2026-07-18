def build_prompt(passage, question, a1, a2, a3, a4):
    return f"""
You are answering a multiple-choice reading comprehension question.

PASSAGE:
{passage}

QUESTION:
{question}

OPTIONS:
A. {a1}
B. {a2}
C. {a3}
D. {a4}

Instructions:
1. Read the passage carefully.
2. Choose exactly one answer (A, B, C, or D).
3. Estimate the probability that your chosen answer is correct.
4. Confidence must be an INTEGER from 0 to 100.
5. Do not use decimals.
6. Do not include the % symbol.
7. Do not provide explanations.
8. Return ONLY valid JSON.

The JSON must follow exactly this schema:

{{
  "answer": "A",
  "confidence": 87
}}

Rules:
- "answer" must be one of "A", "B", "C", or "D".
- "confidence" must be an integer between 0 and 100.
- Do not include any text before or after the JSON.
"""