import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from telemetry import Telemetry, event, metric, span  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def telemetry_run(tmp_path):
    """A small finished run: a parent span with a child, a metric curve, an event and a failing span."""
    with Telemetry("test", directory=tmp_path / "telemetry", interval=0.1) as t:
        with span("parent", agent="planner"):
            with span("child", iteration=0) as s:
                time.sleep(0.3)
                s["items"] = 3
            for step in range(3):
                metric("loss", 1 / (step + 1), step=step)
            event("feedback", accepted=True)
        with pytest.raises(ValueError):
            with span("broken", agent="tool"):
                raise ValueError("boom")
    return t


@pytest.fixture
def fake_whisper(monkeypatch):
    """Replace mlx_whisper with a stub: one real segment, then a French and an English hallucination."""
    calls = {}

    def transcribe(audio, **kwargs):
        calls.update(kwargs)
        return {"segments": [
            {"start": 0.0, "end": 1.0, "text": " Bonjour à tous. "},
            {"start": 1.0, "end": 2.0, "text": "Sous-titres réalisés par la communauté d'Amara.org"},
            {"start": 2.0, "end": 3.0, "text": "Thanks for watching!"},
        ]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    return calls


@pytest.fixture
def wav_file(tmp_path):
    import soundfile as sf
    path = tmp_path / "meeting.wav"
    sf.write(path, (0.1 * np.sin(np.linspace(0, 2000, 16000 * 3))).astype(np.float32), 16000)
    return path
