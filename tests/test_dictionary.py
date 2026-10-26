import json

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
