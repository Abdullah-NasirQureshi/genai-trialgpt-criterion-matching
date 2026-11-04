"""Prompt construction, output schema and robust output parsing.

Input to the model : numbered patient sentences + criterion type + criterion text.
Output schema      : a single-line JSON object
                     {"label": "<one label>", "evidence": [<sentence ids>]}
Only dataset fields allowed as input are used (no gpt4_* columns).
"""
import json
import re

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]

SYSTEM = ("You are a clinical trial screening assistant (educational prototype). "
          "Judge ONE eligibility criterion against a patient note using only the note.")


def labels_for(criterion_type: str):
    return INCLUSION_LABELS if criterion_type == "inclusion" else EXCLUSION_LABELS


def split_sentences(note: str) -> list[str]:
    """Sentences of a note that is formatted '0. text\\n1. text ...' (IDs are 0-based)."""
    parts = re.split(r"(?m)^(\d+)\. ", note)
    sents = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts), 2)}
    if not sents:  # free text without numbering -> split on sentence ends
        raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", note.strip()) if s.strip()]
        return raw
    return [sents[i] for i in sorted(sents)]


def format_note(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def user_prompt(sentences: list[str], criterion_type: str, criterion_text: str) -> str:
    labs = ", ".join(f'"{l}"' for l in labels_for(criterion_type))
    return (
        f"Patient note (sentence IDs in front):\n{format_note(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Choose one label from [{labs}] and list the IDs of the patient sentences "
        f"that support it (empty list if none). "
        f'Answer only as JSON: {{"label": ..., "evidence": [...]}}'
    )


def build_messages(sentences, criterion_type, criterion_text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(sentences, criterion_type, criterion_text)}]


def target_text(label: str, evidence: list[int]) -> str:
    """Gold completion used as the supervised target."""
    return json.dumps({"label": label, "evidence": sorted(evidence)})


def encode_prompt(tokenizer, sentences, criterion_type, criterion_text):
    """Chat-template prompt string ending right before the assistant answer."""
    return tokenizer.apply_chat_template(
        build_messages(sentences, criterion_type, criterion_text),
        tokenize=False, add_generation_prompt=True)


def parse_output(text: str, criterion_type: str, n_sentences: int):
    """Parse model output -> (label|None, evidence list, valid flag).

    Invalid / unparseable output is NOT silently mapped to a class: label is None
    and counted as an error by the evaluator.
    """
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None, [], False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        lab = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        ev = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', text)
        obj = {"label": lab.group(1) if lab else None,
               "evidence": [int(x) for x in re.findall(r"\d+", ev.group(1))] if ev else []}
    label = obj.get("label")
    label = label.strip().lower() if isinstance(label, str) else None
    if label not in labels_for(criterion_type):
        return None, [], False
    ev = obj.get("evidence", [])
    ev = sorted({int(x) for x in ev if isinstance(x, (int, float)) and 0 <= int(x) < n_sentences}) \
        if isinstance(ev, list) else []
    return label, ev, True
