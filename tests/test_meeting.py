import json
import sys
import types

import meeting


def test_french_and_english_hallucinations_are_filtered(fake_whisper):
    segments = meeting.transcribe(None, "model", "fr", None)
    assert [s["text"] for s in segments] == ["Bonjour à tous."]


def test_language_defaults_to_french_and_accepts_auto(fake_whisper, wav_file, tmp_path):
    common = ["--file", str(wav_file), "--no-telemetry", "--directory", str(tmp_path)]
    meeting.main(common)
    assert fake_whisper["language"] == "fr"
    meeting.main(common + ["--language", "auto"])
    assert fake_whisper["language"] is None
    meeting.main(common + ["--language", "en"])
    assert fake_whisper["language"] == "en"


def test_assign_speakers_uses_largest_overlap():
    segments = [{"start": 0, "end": 2, "text": "a"}, {"start": 2, "end": 4, "text": "b"},
                {"start": 10, "end": 11, "text": "c"}]
    meeting.assign_speakers(segments, [(0, 2.5, "SPK_7"), (2.5, 4, "SPK_3")])
    assert [s["speaker"] for s in segments] == ["Speaker 1", "Speaker 2", "?"]


def test_format_transcript_groups_consecutive_turns():
    segments = [{"start": 0, "end": 1, "text": "a", "speaker": "Speaker 1"},
                {"start": 1, "end": 2, "text": "b", "speaker": "Speaker 1"},
                {"start": 65, "end": 66, "text": "c", "speaker": "Speaker 2"}]
    assert meeting.format_transcript(segments) == (
        "**Speaker 1** [00:00:00]\na b\n\n**Speaker 2** [00:01:05]\nc\n")


def test_cli_end_to_end_with_speakers_and_telemetry(fake_whisper, wav_file, tmp_path, monkeypatch):
    received = {}

    def fake_diarize(audio, model, num_speakers):
        received["num_speakers"] = num_speakers
        return [(0.0, 3.0, "SPK_0")]

    monkeypatch.setattr(meeting, "diarize", fake_diarize)
    meeting.main(["--file", str(wav_file), "--speakers", "--nb-speakers", "2",
                  "--directory", str(tmp_path / "Meetings"),
                  "--telemetry-directory", str(tmp_path / "telemetry")])

    assert received["num_speakers"] == 2
    (output,) = (tmp_path / "Meetings").iterdir()
    segments = json.loads((output / "segments.json").read_text())
    assert segments == [{"start": 0.0, "end": 1.0, "text": "Bonjour à tous.", "speaker": "Speaker 1"}]

    (run,) = [p for p in (tmp_path / "telemetry").iterdir() if not p.name.startswith("_")]
    assert json.loads((run / "run.json").read_text())["status"] == "ok"
    assert (run / "report" / "step_summary.csv").exists()


def test_cli_without_telemetry(fake_whisper, wav_file, tmp_path):
    meeting.main(["--file", str(wav_file), "--no-telemetry", "--directory", str(tmp_path / "Meetings"),
                  "--telemetry-directory", str(tmp_path / "telemetry")])
    assert not (tmp_path / "telemetry").exists()


def _run_with_failing_diarization(error, fake_whisper, wav_file, tmp_path, monkeypatch, capsys):
    def failing_diarize(*args):
        raise error

    monkeypatch.setattr(meeting, "diarize", failing_diarize)
    meeting.main(["--file", str(wav_file), "--speakers", "--no-telemetry",
                  "--directory", str(tmp_path / "Meetings")])
    (output,) = (tmp_path / "Meetings").iterdir()
    assert (output / "transcript.md").exists(), "a failing optional step must not lose the transcript"
    return capsys.readouterr()


def test_expected_failure_of_optional_step_is_a_short_warning(fake_whisper, wav_file, tmp_path,
                                                               monkeypatch, capsys):
    out = _run_with_failing_diarization(ImportError("No module named 'pyannote'"),
                                        fake_whisper, wav_file, tmp_path, monkeypatch, capsys)
    assert "Speaker identification unavailable" in out.out
    assert "Traceback" not in out.err


def test_bug_in_optional_step_shows_its_traceback(fake_whisper, wav_file, tmp_path, monkeypatch, capsys):
    out = _run_with_failing_diarization(AttributeError("'Namespace' object has no attribute 'x'"),
                                        fake_whisper, wav_file, tmp_path, monkeypatch, capsys)
    assert "Speaker identification unavailable" in out.out
    assert "Traceback" in out.err and "AttributeError" in out.err


def test_split_transcript_respects_budget_and_line_boundaries():
    lines = [f"[00:00:{i:02d}] word word word" for i in range(10)]  # 5 "tokens" per line with the +1
    parts = meeting.split_transcript("\n".join(lines), lambda line: len(line.split()), max_tokens=12)
    assert ["\n".join(lines[0:2]), "\n".join(lines[2:4])] == parts[:2]
    assert "\n".join(parts).splitlines() == lines


class _FakeLLM:
    """Stands in for mlx_lm: records every prompt and answers with a numbered reply."""

    def __init__(self, monkeypatch):
        self.prompts = []
        monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(
            load=lambda model: ("llm", self), stream_generate=self.stream_generate))

    def encode(self, text):
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)

    def stream_generate(self, llm, tokenizer, prompt, max_tokens):
        self.prompts.append(prompt)
        yield types.SimpleNamespace(text=f"reply {len(self.prompts)}", prompt_tokens=1, generation_tokens=1,
                                    prompt_tps=1.0, generation_tps=1.0, peak_memory=1.0, finish_reason="stop")


def test_short_transcript_is_reported_in_one_pass(monkeypatch):
    fake = _FakeLLM(monkeypatch)
    assert meeting.summarize("[00:00:00] short meeting", "model", "", "fr") == "reply 1"
    assert len(fake.prompts) == 1
    assert "Write the report in French" in fake.prompts[0]


def test_long_transcript_is_reported_with_map_reduce(monkeypatch):
    fake = _FakeLLM(monkeypatch)
    monkeypatch.setattr(meeting, "CHUNK_TOKENS", 10)
    text = "\n".join(f"[00:00:{i:02d}] one two three" for i in range(6))
    assert meeting.summarize(text, "model", "", "en") == "reply 4"
    notes, final = fake.prompts[:3], fake.prompts[3]
    assert all("part" in p and "of 3" in p for p in notes)
    assert "reply 1" in final and "reply 3" in final and "Write the report in English" in final


def test_private_inputs_stay_out_of_telemetry(fake_whisper, wav_file, tmp_path, monkeypatch):
    monkeypatch.setattr(meeting, "summarize", lambda *args: "report")
    secret_audio = wav_file.rename(tmp_path / "Camille-private-review.wav")
    argv = ["--file", str(secret_audio), "--report", "--report-context=Supervisor: Hélène",
            "--vocabulary", "TeraSPIN", "--directory", str(tmp_path / "Meetings"),
            "--telemetry-directory", str(tmp_path / "telemetry")]
    monkeypatch.setattr(sys, "argv", ["meeting.py", *argv])
    meeting.main(argv)

    (run,) = [p for p in (tmp_path / "telemetry").iterdir() if not p.name.startswith("_")]
    for name in ("run.json", "events.jsonl"):
        content = (run / name).read_text()
        for private in ("Camille", "Hélène", "H\\u00e9l\\u00e8ne", "TeraSPIN"):
            assert private not in content, f"{private!r} leaked into {name}"


def test_redacted_command():
    assert meeting.redacted_command(["meeting.py", "--file", "a.m4a", "--report-context=x", "--report",
                                     "--vocabulary", "y", "--dictionary", "d.json", "--language", "fr"]) == [
        "meeting.py", "--file", "<redacted>", "--report-context=<redacted>", "--report",
        "--vocabulary", "<redacted>", "--dictionary", "<redacted>", "--language", "fr"]
