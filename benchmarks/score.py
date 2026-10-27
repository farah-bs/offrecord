"""score.py: how the benchmarks turn model outputs into numbers, with the standard tools.

Word error rate   jiwer (fewest substitutions, deletions and insertions), after Whisper's own text
                  normalization for English, and its basic one for other languages, accents kept
                  so that accent errors count. Computed over the whole corpus: all errors over all
                  reference words, so that a long meeting weighs more than a short one.
Confidence        95 % intervals from resampling meetings, not words: errors within one meeting
                  are correlated (same speakers, same microphone, same topic).
Comparison        two systems are scored on the same meetings, and "A is better than B" holds only
                  when the interval of their difference excludes zero.

self_check() runs before every benchmark: a scorer that fails it would make every number wrong.
"""

from functools import lru_cache

import jiwer
import numpy as np
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

RESAMPLES = 10_000
SEED = 0  # the same intervals on every run


@lru_cache
def normalizer(language: str):
    """Whisper's normalization: case, punctuation, hesitations, numbers and spellings for English."""
    return EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()


def count_errors(reference: str, hypothesis: str, language: str, meeting: str = "") -> dict:
    """Word errors of one meeting, after normalization."""
    norm = normalizer(language)
    ref, hyp = norm(reference).split(), norm(hypothesis).split()
    if not ref:
        raise ValueError(f"{meeting or 'reference'}: nothing to score after normalization")
    if not hyp:
        sub, dels, ins = 0, len(ref), 0
    else:
        out = jiwer.process_words(" ".join(ref), " ".join(hyp))
        sub, dels, ins = out.substitutions, out.deletions, out.insertions
    return {"meeting": meeting, "words": len(ref), "substitutions": sub, "deletions": dels,
            "insertions": ins}


def _arrays(meetings: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    errors = np.array([m["substitutions"] + m["deletions"] + m["insertions"] for m in meetings], dtype=float)
    return errors, np.array([m["words"] for m in meetings], dtype=float)


def wer(meetings: list[dict]) -> float:
    """Corpus word error rate."""
    errors, words = _arrays(meetings)
    return float(errors.sum() / words.sum())


def _resamples(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))


def interval(meetings: list[dict]) -> tuple[float, float]:
    """95 % confidence interval of the corpus word error rate, from resampling meetings."""
    errors, words = _arrays(meetings)
    idx = _resamples(len(meetings))
    samples = errors[idx].sum(axis=1) / words[idx].sum(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def compare(a: list[dict], b: list[dict]) -> dict:
    """Word error rate of a minus that of b, on the same meetings, with its 95 % interval."""
    if [m["meeting"] for m in a] != [m["meeting"] for m in b]:
        raise ValueError("both systems must be scored on the same meetings, in the same order")
    (errors_a, words), (errors_b, _) = _arrays(a), _arrays(b)
    idx = _resamples(len(a))  # the same resampled meetings for both: a paired comparison
    samples = (errors_a[idx].sum(axis=1) - errors_b[idx].sum(axis=1)) / words[idx].sum(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"difference": wer(a) - wer(b), "low": float(low), "high": float(high),
            "significant": bool(low > 0 or high < 0)}


def self_check() -> None:
    """Cases whose answer is known. Raises RuntimeError if the scorer gets any of them wrong."""
    cases = [
        ("a transcript scored against itself", "The cat sat on the mat.", "The cat sat on the mat.", "en", 0.0),
        ("an empty output", "The cat sat on the mat.", "", "en", 1.0),
        ("one substitution in six words", "the cat sat on the mat", "the cat sat on the hat", "en", 1 / 6),
        ("one insertion in six words", "the cat sat on the mat", "the cat sat on the red mat", "en", 1 / 6),
        ("case and punctuation", "Hello, World!", "hello world", "en", 0.0),
        ("English numbers and hesitations", "Uh, we need ten units.", "we need 10 units", "en", 0.0),
        ("French accents count", "le modèle est prêt", "le modele est pret", "fr", 0.5),
        ("French case and punctuation", "Bonjour, à tous !", "bonjour à tous", "fr", 0.0),
    ]
    for label, reference, hypothesis, language, expected in cases:
        got = wer([count_errors(reference, hypothesis, language)])
        if abs(got - expected) > 1e-9:
            raise RuntimeError(f"scorer self-check failed on {label}: {got:.3f} instead of {expected:.3f}")
