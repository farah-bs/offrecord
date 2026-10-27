import json

import pytest

import dictionary
import meeting
from benchmarks import score


def _meeting(name, words, errors):
    return {"meeting": name, "words": words, "substitutions": errors, "deletions": 0, "insertions": 0}


def test_scorer_passes_its_self_check():
    score.self_check()


def test_errors_are_counted_by_kind_after_normalization():
    counts = score.count_errors("The cat sat on the mat.", "the bat sat the mat today", "en", "ES2004a")
    assert counts == {"meeting": "ES2004a", "words": 6, "substitutions": 1, "deletions": 1, "insertions": 1}
    with pytest.raises(ValueError, match="nothing to score"):
        score.count_errors("Uh, hmm.", "anything", "en")


def test_corpus_wer_weighs_meetings_by_their_length():
    meetings = [_meeting("short", 10, 1), _meeting("long", 100, 50)]
    assert score.wer(meetings) == 51 / 110  # not the mean of 10 % and 50 %


def test_interval_is_reproducible_and_contains_the_wer():
    meetings = [_meeting(str(i), 100 + i, e) for i, e in enumerate([5, 12, 8, 30, 9, 11, 7, 15])]
    low, high = score.interval(meetings)
    assert low < score.wer(meetings) < high
    assert score.interval(meetings) == (low, high)
    assert score.interval([_meeting(str(i), 100, 10) for i in range(8)]) == pytest.approx((0.1, 0.1))


def test_comparison_is_paired_and_significant_only_when_clear():
    base = [_meeting(str(i), 100 + 10 * i, e) for i, e in enumerate([5, 30, 8, 20, 9, 25, 7, 15])]
    same = score.compare(base, base)
    assert same["difference"] == 0 and not same["significant"]
    # One error fewer on every meeting: small, but consistent across meetings, so significant.
    better = [_meeting(m["meeting"], m["words"], m["substitutions"] - 1) for m in base]
    result = score.compare(better, base)
    assert result["difference"] < 0 and result["significant"]
    with pytest.raises(ValueError, match="same meetings"):
        score.compare(base, base[::-1])


def test_benchmarks_transcribe_exactly_like_meeting_py(fake_whisper, wav_file, tmp_path, no_personal_dictionary):
    dictionary.save({"terms": ["TeraSPIN"], "corrections": {"Bonjour": "Salut"}}, no_personal_dictionary)
    meeting.main(["--file", str(wav_file), "--no-telemetry", "--directory", str(tmp_path / "Meetings")])
    (output,) = (tmp_path / "Meetings").iterdir()
    produced = json.loads((output / "segments.json").read_text())

    measured = meeting.transcript_segments(meeting.load_audio(wav_file), meeting.DEFAULT_WHISPER_MODEL, "fr",
                                           dictionary.load(no_personal_dictionary))
    assert measured == produced
    assert produced[0]["text"] == "Salut à tous." and fake_whisper["initial_prompt"] == "TeraSPIN."
