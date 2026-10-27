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
from collections import Counter
from difflib import SequenceMatcher
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


# ---------------------------------------------------------------- learning from hand corrections

WORD = re.compile(r"\w+(?:[-'’]\w+)*")
HEADER = re.compile(r"^(# .*|\*\*.*\*\* \[\d\d:\d\d:\d\d\])$")  # title and speaker lines
TIMESTAMP = re.compile(r"^\[\d\d:\d\d:\d\d\] ")
MAX_RULE_WORDS = 3  # longer rewrites are edits of meaning, not recognition errors
MIN_SIMILARITY = 0.5  # misheard words are spelled alike: "Lefebre" / "Lefebvre" is 0.93


def words(text: str) -> list[str]:
    """Words without punctuation: punctuation edits are not recognition errors."""
    return WORD.findall(text)


def transcript_words(markdown: str) -> list[str]:
    """Words of a transcript.md, without its title, speaker names and timestamps."""
    lines = (TIMESTAMP.sub("", line) for line in markdown.splitlines() if not HEADER.match(line.strip()))
    return words("\n".join(lines))


def _opcodes(a: list[str], b: list[str]):
    return SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def _similarity(a: str, b: str) -> float:
    """Spelling similarity, ignoring case, spaces and hyphens."""
    def norm(t):
        return re.sub(r"[\s-]", "", t.lower())
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Substituted, inserted and deleted words over the reference length, ignoring case."""
    ref, hyp = [w.lower() for w in reference], [w.lower() for w in hypothesis]
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in _opcodes(hyp, ref) if op != "equal")
    return errors / max(len(ref), 1)


def looks_like_a_term(text: str) -> bool:
    """Names, acronyms and jargon: an uppercase letter, a digit or a hyphen."""
    return any(c.isupper() or c.isdigit() or c == "-" for c in text)


def propose(original: list[str], edited: list[str], dictionary: dict) -> tuple[list, list]:
    """New corrections and terms suggested by the differences between two versions of a transcript.

    Returns ([(wrong, right, count)], [(term, count)]), most frequent first."""
    corrections, terms = Counter(), Counter()
    for op, i1, i2, j1, j2 in _opcodes(original, edited):
        wrong, right = " ".join(original[i1:i2]), " ".join(edited[j1:j2])
        if op == "equal" or max(i2 - i1, j2 - j1) > MAX_RULE_WORDS:
            continue
        if op == "replace":
            if _similarity(wrong, right) < MIN_SIMILARITY:
                continue  # a rewrite, not a misheard word
            if dictionary["corrections"].get(wrong) != right:
                corrections[wrong, right] += 1
        if op != "delete" and looks_like_a_term(right) and right not in dictionary["terms"]:
            terms[right] += 1
    return ([(w, r, n) for (w, r), n in corrections.most_common()], terms.most_common())
