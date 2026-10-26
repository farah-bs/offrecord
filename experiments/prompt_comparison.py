#!/usr/bin/env python3
"""prompt_comparison.py: does a Whisper prompt fix the spelling of names and jargon?

Transcribes short clips of a recording with every combination of Whisper model and prompt style,
and scores each output on the terms the clip is known to contain.

Usage:
    python experiments/prompt_comparison.py --audio meeting.m4a --clips experiments/meeting.clips.json

The clips file lists, for each clip, its start and end in seconds and the terms it contains,
spelled correctly (each clip at most 30 s, one Whisper window):

    {"terms": ["TeraSPIN", "LABIX"],
     "clips": [{"start": 105, "end": 125, "expected": ["TeraSPIN"]}]}

Each run is recorded with the project's telemetry (telemetry/<date>_prompt_comparison_<id>/):
one span per transcription, the scores as metrics, the transcribed texts in outputs.jsonl and the
chart in prompt_comparison.png. Successive runs are kept, so they form a history.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyze_telemetry import SERIES  # noqa: E402
from meeting import SR, load_audio  # noqa: E402
from telemetry import Telemetry, metric, span  # noqa: E402

MODELS = ("mlx-community/whisper-large-v3-turbo", "mlx-community/whisper-large-v3-mlx")


def prompts(terms: list[str]) -> dict[str, str | None]:
    """Prompt styles compared: none, the list sent by meeting.py, and a sentence in French."""
    return {
        "none": None,
        "term list": ", ".join(terms) + ".",
        "sentence": "Dans cet échange, on parle de " + ", ".join(terms) + ".",
    }


def found(term: str, text: str) -> bool:
    """The term, spelled exactly (case included), as whole words; a version number may follow
    ("AlphaGeometry2")."""
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?![^\W\d])", text) is not None


def score(text: str, expected: list[str], terms: list[str], baseline: str) -> tuple[float, int]:
    """(share of expected terms spelled right, prompt terms that appear although not said).

    A term counts as invented when it is neither expected in the clip nor in the unprompted output."""
    hits = sum(found(t, text) for t in expected) / len(expected)
    invented = sum(found(t, text) and t not in expected and not found(t, baseline) for t in terms)
    return hits, invented


def chart(results: list[dict], path: Path) -> None:
    models = list(dict.fromkeys(r["model"] for r in results))
    styles = list(dict.fromkeys(r["prompt"] for r in results))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    width = 0.8 / len(models)
    for ax, key, title in ((axes[0], "accuracy", "Expected terms spelled right (%)"),
                           (axes[1], "invented", "Prompt terms invented (count)")):
        for i, model in enumerate(models):
            values = []
            for style in styles:
                rows = [r for r in results if r["model"] == model and r["prompt"] == style]
                values.append(100 * sum(r["accuracy"] for r in rows) / len(rows) if key == "accuracy"
                              else sum(r["invented"] for r in rows))
            x = [j + (i - (len(models) - 1) / 2) * width for j in range(len(styles))]
            bars = ax.bar(x, values, width * 0.9, color=SERIES[i], label=model.split("/")[-1],
                          edgecolor="white", linewidth=2)
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2, color="#333333")
        ax.set_xticks(range(len(styles)), styles, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100 if key == "accuracy" else None)
        ax.margins(y=0.15)
        if key == "invented":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(fontsize=8, title="Whisper model", title_fontsize=8, loc="upper left")
    fig.suptitle("Whisper prompt comparison", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Whisper prompt styles on clips with known terms")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--language", default="fr")
    p.add_argument("--telemetry-directory", type=Path, default=ROOT / "telemetry")
    args = p.parse_args()

    import mlx_whisper

    spec = json.loads(args.clips.read_text(encoding="utf-8"))
    audio = load_audio(args.audio.expanduser())
    styles = prompts(spec["terms"])
    results = []
    # The audio and clip file names may be private: only counts and model names are tagged.
    with Telemetry("prompt_comparison", directory=args.telemetry_directory,
                   tags={"clips": len(spec["clips"]), "terms": len(spec["terms"])},
                   command=["prompt_comparison.py", "<redacted>"]) as t:
        for model in args.models:
            for c, clip in enumerate(spec["clips"]):
                piece = audio[int(clip["start"] * SR):int(clip["end"] * SR)]
                baseline = None
                for style, prompt in styles.items():
                    with span("transcribe_clip", agent="whisper", model=model, prompt=style, clip=c):
                        text = mlx_whisper.transcribe(
                            piece, path_or_hf_repo=model, language=args.language, initial_prompt=prompt,
                            condition_on_previous_text=False, verbose=None)["text"].strip()
                    baseline = text if baseline is None else baseline
                    accuracy, invented = score(text, clip["expected"], spec["terms"], baseline)
                    metric("term_accuracy", accuracy, model=model, prompt=style, clip=c)
                    metric("invented_terms", invented, model=model, prompt=style, clip=c)
                    results.append({"model": model, "prompt": style, "clip": c, "accuracy": accuracy,
                                    "invented": invented, "expected": clip["expected"], "text": text})
                    print(f"{model.split('/')[-1]:24} clip {c:2} {style:9} {accuracy:4.0%}  {text[:90]}")

        with open(t.directory / "outputs.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        chart(results, t.directory / "prompt_comparison.png")
    print(f"\n📁 {t.directory}")


if __name__ == "__main__":
    main()
