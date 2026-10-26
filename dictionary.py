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
