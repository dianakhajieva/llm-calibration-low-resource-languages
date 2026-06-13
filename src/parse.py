"""Pull the answer letter and the confidence number out of a model's raw reply.

Models do not always answer in a tidy format, so these functions try a few
patterns and fall back gracefully. Anything they cannot parse returns None, and
the main loop records that as a parse failure (so you can see how often it happens).
"""
import re

VALID_LETTERS = {"A", "B", "C", "D"}


def parse_answer(text):
    """
    Return the answer letter (A-D) found in the reply, or None if none is found.
    Handles 'Answer: B', a lone 'B', or 'B.' at the start.
    """
    if not text:
        return None

    # Look for an explicit 'Answer: X' first.
    match = re.search(r"answer\s*[:\-]?\s*([ABCD])", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Otherwise take the first standalone A/B/C/D in the text.
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1).upper()

    return None


def parse_confidence(text):
    """
    Return confidence as a fraction between 0 and 1, or None if not found.
    Handles 'Confidence: 80', '80%', or a stray number.
    """
    if not text:
        return None

    match = re.search(r"confidence\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%", text)
    if not match:
        # last resort: the first number anywhere in the text
        match = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)", text)
    if not match:
        return None

    value = float(match.group(1))
    value = max(0.0, min(100.0, value))  # clamp into the 0-100 range
    return value / 100.0


def letter_to_index(letter):
    """A->0, B->1, C->2, D->3. Returns None for anything else."""
    if letter in VALID_LETTERS:
        return ord(letter) - ord("A")
    return None
