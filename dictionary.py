"""dictionary.py: personal dictionary of terms and corrections for meeting transcripts.

The dictionary is a local JSON file (dictionary.json, git-ignored) with two parts:

    {
      "terms": ["neuro-symbolique", "TeraSPIN", "LABIX"],
      "corrections": {"neuro symbolique": "neuro-symbolique", "Lefebre": "Lefebvre"}
    }

terms        names, acronyms and jargon, most important first: they guide Whisper and the report.
corrections  wrong -> right replacements applied to the transcript, whole words only, case-sensitive.
"""

import json
import re
from pathlib import Path


def load(path: Path) -> dict:
    """The dictionary at path, or an empty one if the file does not exist."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"terms": list(data.get("terms", [])), "corrections": dict(data.get("corrections", {}))}


def save(dictionary: dict, path: Path) -> None:
    path.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pattern(corrections: dict) -> re.Pattern | None:
    if not corrections:
        return None
    # Longest first, so that "neuro symbolique" wins over a rule for "neuro" alone.
    words = sorted(corrections, key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(map(re.escape, words)) + r")(?!\w)")


def apply_corrections(segments: list[dict], corrections: dict) -> int:
    """Replace known errors in each segment's text, in place. Returns the number of replacements.

    A corrected segment keeps Whisper's text in "whisper_text", so the effect of the corrections
    can be measured later (see word_error_rate)."""
    pattern = _pattern(corrections)
    if pattern is None:
        return 0
    total = 0
    for s in segments:
        text, n = pattern.subn(lambda m: corrections[m.group(1)], s["text"])
        if n:
            s["whisper_text"], s["text"] = s["text"], text
            total += n
    return total


# Whisper keeps at most 223 prompt tokens and drops the beginning beyond that. French names and
# jargon measured at about 2.6 characters per token, so this budget stays under the limit even
# for denser terms (2 characters per token).
PROMPT_MAX_CHARS = 400


def prompt_terms(terms: list[str], vocabulary: str | None = None) -> list[str]:
    """The terms sent to Whisper: --vocabulary first, then the dictionary, within PROMPT_MAX_CHARS."""
    extra = [t.strip() for t in (vocabulary or "").split(",") if t.strip()]
    kept, size = [], 0
    for term in dict.fromkeys(extra + terms):  # removes duplicates, keeps order
        if size + len(term) + 2 > PROMPT_MAX_CHARS:
            break
        kept.append(term)
        size += len(term) + 2
    return kept


def whisper_prompt(terms: list[str]) -> str | None:
    return ", ".join(terms) + "." if terms else None


def glossary(dictionary: dict, vocabulary: str | None = None) -> str:
    """Glossary section of the report prompt: every term, with no length limit."""
    extra = [t.strip() for t in (vocabulary or "").split(",") if t.strip()]
    terms = dict.fromkeys(extra + dictionary["terms"] + list(dictionary["corrections"].values()))
    if not terms:
        return "No additional glossary provided."
    return ("Names and terms that may appear in the meeting. Spell them exactly as written here:\n"
            + "\n".join(f"- {t}" for t in terms))
