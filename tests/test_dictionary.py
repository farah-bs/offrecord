import json
import sys
import types

import numpy as np

import dictionary
import meeting


def test_missing_file_is_an_empty_dictionary(tmp_path):
    assert dictionary.load(tmp_path / "none.json") == {"terms": [], "corrections": {}}


def test_save_and_load_round_trip(tmp_path):
    d = {"terms": ["TeraSPIN", "neuro-symbolique"], "corrections": {"Lefebre": "Lefebvre"}}
    dictionary.save(d, tmp_path / "d.json")
    assert dictionary.load(tmp_path / "d.json") == d


def test_corrections_match_whole_words_only_and_keep_whisper_text():
    segments = [{"text": "Lefebre parle de neuro symbolique."},
                {"text": "Les Lefebrettes et la neuro symboliquement."},
                {"text": "IA et AI."}]
    n = dictionary.apply_corrections(segments, {"Lefebre": "Lefebvre",
                                                "neuro symbolique": "neuro-symbolique",
                                                "neuro": "NEURO"})
    assert n == 3
    assert segments[0] == {"text": "Lefebvre parle de neuro-symbolique.",
                           "whisper_text": "Lefebre parle de neuro symbolique."}
    assert segments[1]["text"] == "Les Lefebrettes et la NEURO symboliquement."
    assert "whisper_text" not in segments[2]


def test_cli_applies_dictionary_corrections(fake_whisper, wav_file, tmp_path, no_personal_dictionary):
    dictionary.save({"terms": [], "corrections": {"tous": "toutes"}}, no_personal_dictionary)
    meeting.main(["--file", str(wav_file), "--no-telemetry", "--directory", str(tmp_path / "Meetings")])
    (output,) = (tmp_path / "Meetings").iterdir()
    assert json.loads((output / "segments.json").read_text())[0]["text"] == "Bonjour à toutes."
    assert "Bonjour à toutes." in (output / "transcript.md").read_text()


def test_prompt_terms_put_vocabulary_first_and_fit_the_budget(monkeypatch):
    assert dictionary.prompt_terms(["LABIX", "TeraSPIN"], "MLX, LABIX") == ["MLX", "LABIX", "TeraSPIN"]
    monkeypatch.setattr(dictionary, "PROMPT_MAX_CHARS", 20)
    assert dictionary.prompt_terms(["abcdefgh", "ijklmnop", "qrstuvwx"]) == ["abcdefgh", "ijklmnop"]
    assert dictionary.whisper_prompt(["MLX", "LABIX"]) == "MLX, LABIX."
    assert dictionary.whisper_prompt([]) is None


def test_audio_blocks_cut_at_silence_and_cover_everything():
    sr = meeting.SR
    audio = np.ones(sr * 100, dtype=np.float32)
    audio[int(sr * 41.5):int(sr * 41.6)] = 0  # silence 1.5 s after the 40 s target
    blocks = meeting.audio_blocks(audio, block_s=40)
    assert blocks[0][0] == 0 and blocks[-1][1] == len(audio)
    assert all(a == b for (_, a), (b, _) in zip(blocks, blocks[1:]))
    assert abs(blocks[0][1] / sr - 41.55) < 0.06
    assert len(blocks) == 2  # the remaining 18 s after the second target join the last block


def test_prompt_is_sent_to_every_block_with_realigned_timestamps(monkeypatch, wav_file, tmp_path,
                                                                 no_personal_dictionary):
    calls = []

    def transcribe(audio, **kwargs):
        calls.append(kwargs["initial_prompt"])
        return {"segments": [{"start": 1.0, "end": 2.0, "text": f"bloc {len(calls)}"}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(meeting, "audio_blocks", lambda audio: [(0, 16000), (16000, 32000), (32000, 48000)])
    dictionary.save({"terms": ["TeraSPIN"], "corrections": {}}, no_personal_dictionary)
    meeting.main(["--file", str(wav_file), "--no-telemetry", "--vocabulary", "LABIX",
                  "--directory", str(tmp_path / "Meetings")])

    assert calls == ["LABIX, TeraSPIN."] * 3
    (output,) = (tmp_path / "Meetings").iterdir()
    segments = json.loads((output / "segments.json").read_text())
    assert [(s["start"], s["text"]) for s in segments] == [(1.0, "bloc 1"), (2.0, "bloc 2"), (3.0, "bloc 3")]


def test_without_prompt_the_audio_is_transcribed_in_one_call(fake_whisper):
    meeting.transcribe(None, "model", "fr", None)
    assert fake_whisper["initial_prompt"] is None


def test_glossary_lists_every_term_once():
    d = {"terms": ["TeraSPIN", "LABIX"], "corrections": {"Lefebre": "Lefebvre", "labix": "LABIX"}}
    assert dictionary.glossary(d, "MLX") == (
        "Names and terms that may appear in the meeting. Spell them exactly as written here:\n"
        "- MLX\n- TeraSPIN\n- LABIX\n- Lefebvre")
    assert dictionary.glossary({"terms": [], "corrections": {}}) == "No additional glossary provided."


def test_glossary_reaches_both_report_prompts(monkeypatch):
    from test_meeting import _FakeLLM
    fake = _FakeLLM(monkeypatch)
    monkeypatch.setattr(meeting, "CHUNK_TOKENS", 10)
    text = "\n".join(f"[00:00:{i:02d}] one two three" for i in range(6))
    meeting.summarize(text, "model", "", "fr", "- TeraSPIN")
    assert len(fake.prompts) == 4 and all("## Glossary\n- TeraSPIN" in p for p in fake.prompts)
