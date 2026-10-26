import json

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
