#!/usr/bin/env python3
"""meeting.py: fully local meeting recording and transcript for Apple Silicon Macs.

Usage:
    python meeting.py                         # record from the microphone, press Enter to stop, then transcribe
    python meeting.py --file meeting.m4a      # transcribe an existing file
    python meeting.py --speakers --report     # identify speakers and generate a meeting report

Everything runs locally: Whisper through MLX, pyannote for speakers, and an MLX LLM for the report. No audio leaves the machine."""

import argparse
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from telemetry import Telemetry, event, metric, span

SR = 16000
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Implementation note.
HALLUCINATIONS = (
    "amara.org",
    "subtitles created",
    "subtitling",
    "thanks for watching",
    "subscribe",
)


# Implementation note.

def record(wav_path: Path) -> None:
    import sounddevice as sd
    import soundfile as sf

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, t, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()

    print("\n🎙  Recording in progress. Press Enter to stop.\n")
    start = time.time()
    # Implementation note.
    with sf.SoundFile(wav_path, mode="w", samplerate=SR, channels=1, subtype="PCM_16") as f, \
            sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
        try:
            while not stop.is_set():
                try:
                    block = q.get(timeout=0.2)
                    f.write(block)
                    level = float(np.sqrt(np.mean(block ** 2)))
                except queue.Empty:
                    level = 0.0
                s = int(time.time() - start)
                bar = "█" * min(30, int(level * 300))
                print(f"\r  {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}  {bar:<30}",
                      end="", flush=True)
        except KeyboardInterrupt:
            pass
        while not q.empty():
            f.write(q.get())
    print(f"\n\n✅ Audio recorded : {wav_path}")


def load_audio(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr == SR:
            return audio
    # Implementation note.
    from mlx_whisper.audio import load_audio
    return np.asarray(load_audio(str(path)), dtype=np.float32)


# Implementation note.

def transcribe(audio: np.ndarray, model: str, language: str, vocabulary: str | None) -> list[dict]:
    import mlx_whisper

    print(f"\n📝 Transcribing with {model} (first run: model download)...")
    res = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        language=language,
        initial_prompt=vocabulary,
        condition_on_previous_text=False,  # Limit repetition loops.
        verbose=False,
    )
    segments = []
    for s in res["segments"]:
        text = s["text"].strip()
        if not text or any(h in text.lower() for h in HALLUCINATIONS):
            continue
        segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": text})
    metric("raw_segments", len(res["segments"]))
    metric("filtered_segments", len(res["segments"]) - len(segments))
    return segments


# Implementation note.

def diarize(audio: np.ndarray, model: str, num_speakers: int | None) -> list[tuple]:
    import torch
    from pyannote.audio import Pipeline

    print("\n👥 Identifying speakers with pyannote (this may take several minutes)...")
    token = os.environ.get("HF_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)            # Implementation note.
    except TypeError:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)   # Implementation note.
    if pipeline is None:
        raise RuntimeError(
            "Access to the pyannote model was denied. Accept the Hugging Face terms "
            "and set HF_TOKEN (see instructions)."
        )

    input_data = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipeline(input_data, **kwargs)
    annotation = getattr(output, "speaker_diarization", output)
    return [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]


def assign_speakers(segments: list[dict], speaker_turns: list[tuple]) -> None:
    names: dict[str, str] = {}
    for s in segments:
        best_speaker, recouvrement = None, 0.0
        for a, b, spk in speaker_turns:
            r = min(s["end"], b) - max(s["start"], a)
            if r > recouvrement:
                best_speaker, recouvrement = spk, r
        if best_speaker is None:
            s["speaker"] = "?"
            continue
        if best_speaker not in names:
            names[best_speaker] = f"Speaker {len(names) + 1}"
        s["speaker"] = names[best_speaker]


# Implementation note.

def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def format_transcript(segments: list[dict]) -> str:
    lines, current, block, start = [], None, [], 0.0
    for s in segments:
        spk = s.get("speaker")
        if spk is None:
            lines.append(f"[{hms(s['start'])}] {s['text']}")
            continue
        if spk != current and block:
            lines.append(f"**{current}** [{hms(start)}]\n{' '.join(block)}\n")
            block = []
        if not block:
            start = s["start"]
        current = spk
        block.append(s["text"])
    if block:
        lines.append(f"**{current}** [{hms(start)}]\n{' '.join(block)}\n")
    return "\n".join(lines)


# Implementation note.

PROMPT = """You are a private meeting assistant. Extract all scientific, technical, and organizational content from the automatic transcript; do not summarize away details. Preserve every idea, number, reference, recommendation, and task.

The user may provide optional context about the meeting below. Use it only to
interpret the transcript and tailor the report. Do not invent information that
is not present in the transcript or context.

## User-provided context
{context}

## Rules
1. Invent nothing; do not fill in missing information.
2. Correct speech-recognition errors only when context makes the correction obvious. Otherwise keep the original term and add [uncertain].
3. Preserve model, architecture, method, dataset, metric, library, author, conference, and number names exactly.
4. Attribute statements when possible; otherwise write "unattributed".
5. Treat implicit suggestions as actionable tasks. Do not merge distinct ideas or omit minor points.

## Glossary
No additional glossary provided.

## Output format
Write in English Markdown with exactly these sections, in this order. If a section is empty, write "Nothing mentioned".

## Summary
Meeting subject, main results, and conclusions in one paragraph.

## Scientific points discussed
One ### subsection per topic in chronological order. Include the question, approaches, methods, numerical results, arguments, objections, and conclusion.

## Advice and recommendations
Each recommendation, attributed to its author, with its justification when provided.

## Research directions
New ideas, hypotheses, experiments, and alternatives, including speculative ideas. Mark each as "to explore", "discarded" with its reason, or "to revisit".

## References and resources mentioned
Papers, authors, tools, repositories, datasets, benchmarks, conferences, and people to contact.

## Decisions
What was explicitly decided, or "No explicit decision".

## Milestones and deadlines
Thesis milestones, submission deadlines, delivery dates, and next meeting, with exact dates when given.

## TODO list
- [ ] **Specific actionable task** | Owner: X | Deadline: date or "not specified" | Priority: high/medium/low or "not specified"

Group tasks by owner and provide enough detail to understand each task without rereading the transcript.

## Open questions
Unresolved questions, disagreements, and postponed points.

## Uncertain passages
Passages too degraded to interpret confidently, with your best reading hypothesis.

"""


def summarize(text: str, model: str, report_context: str) -> str:
    from mlx_lm import load, stream_generate

    max_tokens = 2000
    print(f"\n🧠 Writing the meeting report with {model} (first run: ~4 GB to download)...")
    with span("llm_loading", model=model):
        model, tokenizer = load(model)
    messages = [
        {"role": "system", "content": PROMPT.format(
            context=report_context.strip() or "No additional context provided.",
        )},
        {"role": "user", "content": "Meeting transcript :\n\n" + text},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    with span("generation_llm", model=model, max_tokens=max_tokens) as s:
        morceaux, r = [], None
        for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            morceaux.append(r.text)
        s["input_tokens"], s["output_tokens"] = r.prompt_tokens, r.generation_tokens
        metric("prefill_tokens_per_s", r.prompt_tps)        # Implementation note.
        metric("generation_tokens_per_s", r.generation_tps)  # Implementation note.
        metric("mlx_peak_gb", r.peak_memory)
        if r.finish_reason == "length":
            event("truncated_output", output_tokens=r.generation_tokens, max_tokens=max_tokens)
    return "".join(morceaux).strip()


# Implementation note.

def main() -> None:
    p = argparse.ArgumentParser(description="Local meeting transcript")
    p.add_argument("--file", type=Path, help="existing audio file (otherwise: microphone recording)")
    p.add_argument("--title", default="", help="short title, added to the directory name")
    p.add_argument("--speakers", action="store_true", help="identify speakers (pyannote)")
    p.add_argument("--nb-speakers", type=int, help="number of people, if known (improves accuracy)")
    p.add_argument("--report", action="store_true", help="generate a meeting report with a local LLM")
    p.add_argument("--report-context", default="",
                   help="optional context to customize the report (project, audience, terminology, etc.)")
    p.add_argument("--vocabulary", help="technical words to guide Whisper (names, acronyms...)")
    p.add_argument("--language", default="en")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    p.add_argument("--directory", type=Path, default=Path.cwd() / "Meetings",
                   help="where to save (default: ./Meetings in the current directory)")
    p.add_argument("--no-telemetry", action="store_true",
                   help="do not measure resources (default: ./telemetry next to the script)")
    args = p.parse_args()

    if args.no_telemetry:
        context = nullcontext()
    else:
        context = Telemetry("meeting", directory=Path(__file__).resolve().parent / "telemetry",
                              tags={k: v for k, v in vars(args).items()
                                    if k != "report_context" and v not in (None, False, "")})
    with context:
        run_pipeline(args)

    if isinstance(context, Telemetry):
        try:
            from analyze_telemetry import load_runs, run_report
            report = run_report(load_runs(context.directory.parent, run_id=context.id), context.id)
            print(f"📊 Resources : {report}")
        except Exception as e:
            print(f"⚠️  Telemetry report unavailable ({e}). Raw measurements are in {context.directory}")


def run_pipeline(args: argparse.Namespace) -> None:
    name = dt.datetime.now().strftime("%Y-%m-%d_%Hh%M")
    if args.title:
        name += "_" + "".join(c if c.isalnum() else "-" for c in args.title)
    output = args.directory / name
    output.mkdir(parents=True, exist_ok=True)

    if args.file:
        source = args.file.expanduser()
    else:
        source = output / "audio.wav"
        with span("recording", agent="audio"):
            record(source)

    with span("audio_loading", agent="audio", file=source.name):
        audio = load_audio(source)
    audio_duration = len(audio) / SR
    metric("audio_duration_s", audio_duration)
    print(f"Duration: {hms(audio_duration)}")
    if len(audio) < SR:
        sys.exit("Audio too short or empty: check microphone access.")

    t0 = time.time()
    with span("transcript", agent="whisper", model=args.whisper_model, language=args.language) as s:
        segments = transcribe(audio, args.whisper_model, args.language, args.vocabulary)
        s["segments"] = len(segments)
        # Implementation note.
        metric("rtf", (time.time() - t0) / audio_duration)
    print(f"   {len(segments)} segments in {time.time() - t0:.0f} s")

    if args.speakers:
        try:
            with span("diarization", agent="pyannote", model=args.diarization_model) as s:
                assign_speakers(segments, diarize(audio, args.diarization_model, args.num_speakers))
                s["speakers"] = len({x.get("speaker") for x in segments} - {None, "?"})
        except Exception as e:
            print(f"⚠️  Speaker identification unavailable ({e}). Continuing without it.")

    text = format_transcript(segments)
    (output / "transcript.md").write_text(f"# Transcript: {name}\n\n{text}\n", encoding="utf-8")
    (output / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ Transcript: {output / 'transcript.md'}")

    if args.report:
        try:
            with span("meeting_report", agent="llm", model=args.llm_model):
                cr = summarize(text, args.llm_model, args.report_context)
            (output / "meeting_report.md").write_text(f"# Meeting report: {name}\n\n{cr}\n", encoding="utf-8")
            print(f"✅ Meeting report: {output / 'meeting_report.md'}")
        except Exception as e:
            print(f"⚠️  Meeting report unavailable ({e}). The transcript was saved successfully.")

    print(f"\n📁 Directory: {output}")


if __name__ == "__main__":
    main()
