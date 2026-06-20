import re


def parse_response(text):
    answer_match = re.search(
        r"ANSWER:\s*([ABCD])",
        text,
        re.IGNORECASE,
    )

    confidence_match = re.search(
        r"CONFIDENCE:\s*(\d+)",
        text,
        re.IGNORECASE,
    )

    if not answer_match:
        return None

    return {
        "answer": answer_match.group(1).upper(),
        "confidence": int(confidence_match.group(1))
        if confidence_match
        else None,
    }