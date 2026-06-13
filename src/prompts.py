"""
The text we wrap around each Belebele question before sending it to a model.

We keep the INSTRUCTIONS in English on purpose, so that the only thing that
changes between languages is the question content itself. This is a deliberate
design choice and you should mention it in your paper. If you later want fully
localized instructions, translate the two strings below and note the change.
"""

OPTION_LETTERS = ["A", "B", "C", "D"]


def format_options(options):
    """Turn ['cat', 'dog', ...] into 'A. cat\nB. dog\n...'."""
    return "\n".join(f"{letter}. {text}" for letter, text in zip(OPTION_LETTERS, options))


def build_answer_prompt(passage, question, options):
    """
    Prompt for the LOGPROB method: we want the model to output ONLY the letter,
    so the very first token it produces is its answer and we can read the
    probability it assigned to that token.
    """
    return (
        "Read the passage and answer the multiple-choice question.\n\n"
        f"Passage:\n{passage}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "Reply with ONLY the single letter (A, B, C or D) of the correct answer. "
        "Do not explain."
    )


def build_confidence_prompt(passage, question, options):
    """
    Prompt for the VERBALIZED method: the model gives its answer AND states how
    confident it is, in a fixed format that is easy to parse.
    """
    return (
        "Read the passage and answer the multiple-choice question.\n\n"
        f"Passage:\n{passage}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "First give your answer, then state how confident you are that it is "
        "correct, as a number from 0 to 100.\n"
        "Respond in exactly this format and nothing else:\n"
        "Answer: <letter>\n"
        "Confidence: <number>"
    )
