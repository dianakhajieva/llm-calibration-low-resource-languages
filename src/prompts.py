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
- Select exactly one answer.
- Report confidence as an integer between 0 and 100.
- Do not provide explanations.

Return exactly:

ANSWER: <A/B/C/D>
CONFIDENCE: <0-100>
"""