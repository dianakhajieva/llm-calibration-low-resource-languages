import json
import re


def parse_response(text):
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try JSON
    try:
        data = json.loads(text)

        answer = data["answer"].strip().upper()
        confidence = int(data["confidence"])

        if answer not in ["A", "B", "C", "D"]:
            return None

        if not (0 <= confidence <= 100):
            return None

        return {
            "answer": answer,
            "confidence": confidence,
        }

    except Exception:
        pass

    # Fallback legacy format
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