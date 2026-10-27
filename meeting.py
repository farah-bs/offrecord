#!/usr/bin/env python3
"""meeting.py: fully local meeting recording and transcript for Apple Silicon Macs.

Usage:
    python meeting.py                         # record from the microphone, press Enter to stop, then transcribe
    python meeting.py --file meeting.m4a      # transcribe an existing file
    python meeting.py --speakers --report     # identify speakers and generate a meeting report
    python meeting.py review Meetings/<dir>   # let the LLM propose corrections, confirm each one
    python meeting.py learn Meetings/<dir>    # learn dictionary rules from a transcript corrected by hand

Everything runs locally: Whisper through MLX, pyannote for speakers, and an MLX LLM for the report. No audio leaves the machine."""

import argparse
import datetime as dt
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import dictionary as dictionary_file
from telemetry import Telemetry, event, metric, span

SR = 16000
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_DICTIONARY = Path(__file__).resolve().parent / "dictionary.json"

# Phrases Whisper tends to invent during silences (it was trained on subtitled videos).
HALLUCINATIONS = (
    "amara.org",
    # French
    "sous-titres réalisés",
    "sous-titrage st'",
    "merci d'avoir regardé",
    "abonnez-vous",
    # English
    "subtitles by",
    "subtitles created",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
)

# Language the report is written in, from the --language code.
REPORT_LANGUAGES = {"fr": "French", "en": "English"}

# Options whose values may identify people or projects: never written to telemetry.
PRIVATE_OPTIONS = ("--file", "--report-context", "--vocabulary", "--dictionary")

# Failures of optional steps that come from the environment, not from a bug: missing package,
# refused model access, download or disk problem. Anything else also prints its traceback.
EXPECTED_ERRORS = (ImportError, RuntimeError, OSError)


# ---------------------------------------------------------------- recording

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
    # Written to disk continuously: nothing is lost if the script crashes.
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
    print(f"\n\n✅ Audio recorded: {wav_path}")


def load_audio(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr == SR:
            return audio
    # Other formats or sample rates: decoded and resampled by ffmpeg (brew install ffmpeg).
    from mlx_whisper.audio import load_audio
    return np.asarray(load_audio(str(path)), dtype=np.float32)


# ---------------------------------------------------------------- transcription

# With condition_on_previous_text=False, Whisper uses the prompt for the first 30-second window
# of each call only. When there is a prompt, the audio is transcribed in blocks of at most one
# window, each with the prompt, so that terms are recognized throughout the meeting.
BLOCK_S = 30
CUT_SEARCH_S = 6  # each block ends at the quietest moment of its last CUT_SEARCH_S seconds


def audio_blocks(audio: np.ndarray, block_s: float | None = None) -> list[tuple[int, int]]:
    """(start, end) sample ranges of at most block_s seconds, each ending at the quietest 0.1 s
    of its last CUT_SEARCH_S seconds, so that words are not cut in half."""
    block = int((block_s or BLOCK_S) * SR)
    frame, search = SR // 10, CUT_SEARCH_S * SR
    cuts = [0]
    while len(audio) - cuts[-1] > block:
        lo = cuts[-1] + block - search
        n = search // frame
        energy = (audio[lo:lo + n * frame].reshape(n, frame) ** 2).mean(axis=1)
        cuts.append(lo + int(energy.argmin()) * frame + frame // 2)
    return list(zip(cuts, cuts[1:] + [len(audio)]))


def transcribe(audio: np.ndarray, model: str, language: str | None, prompt: str | None) -> list[dict]:
    import mlx_whisper

    print(f"\n📝 Transcribing with {model} (first run: model download)...")
    blocks = audio_blocks(audio) if prompt else [(0, None)]
    metric("whisper_blocks", len(blocks))
    raw, segments = 0, []
    for i, (a, b) in enumerate(blocks, 1):
        if len(blocks) > 1:
            print(f"\r   block {i}/{len(blocks)}...", end="", flush=True)
        res = mlx_whisper.transcribe(
            audio[a:b] if b else audio,
            path_or_hf_repo=model,
            language=language,
            initial_prompt=prompt,
            condition_on_previous_text=False,  # Limit repetition loops.
            verbose=False if len(blocks) == 1 else None,  # None: no progress bar for each block
        )
        offset = a / SR
        raw += len(res["segments"])
        for s in res["segments"]:
            text = s["text"].strip()
            if not text or any(h in text.lower() for h in HALLUCINATIONS):
                continue
            segments.append({"start": float(s["start"]) + offset, "end": float(s["end"]) + offset,
                             "text": text})
    if len(blocks) > 1:
        print()
    metric("raw_segments", raw)
    metric("filtered_segments", raw - len(segments))
    return segments


# ---------------------------------------------------------------- speakers

def diarize(audio: np.ndarray, model: str, num_speakers: int | None) -> list[tuple]:
    import torch
    from pyannote.audio import Pipeline

    print("\n👥 Identifying speakers with pyannote (this may take several minutes)...")
    token = os.environ.get("HF_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(model, token=token)            # pyannote >= 4
    except TypeError:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=token)   # pyannote 3.x
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
        best_speaker, overlap = None, 0.0
        for a, b, spk in speaker_turns:
            r = min(s["end"], b) - max(s["start"], a)
            if r > overlap:
                best_speaker, overlap = spk, r
        if best_speaker is None:
            s["speaker"] = "?"
            continue
        if best_speaker not in names:
            names[best_speaker] = f"Speaker {len(names) + 1}"
        s["speaker"] = names[best_speaker]


# ---------------------------------------------------------------- formatting

def redacted_command(argv: list[str]) -> list[str]:
    """argv with the values of PRIVATE_OPTIONS replaced, for both `--opt value` and `--opt=value`."""
    out, hide_next = [], False
    for arg in argv:
        option = arg.split("=", 1)[0]
        if hide_next:
            out.append("<redacted>")
        elif "=" in arg and option in PRIVATE_OPTIONS:
            out.append(f"{option}=<redacted>")
        else:
            out.append(arg)
        hide_next = arg in PRIVATE_OPTIONS
    return out


def optional_step_failed(step: str, error: Exception, consequence: str) -> None:
    print(f"⚠️  {step} unavailable ({error}). {consequence}")
    if not isinstance(error, EXPECTED_ERRORS):
        traceback.print_exc()


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


# ---------------------------------------------------------------- meeting report

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
{glossary}

## Output format
Write the report in {language}, in Markdown, with exactly these sections, in this order. If a section is empty, write "Nothing mentioned".

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


NOTES_PROMPT = """You are a private meeting assistant. You receive part {index} of {total} of an automatic meeting transcript. A final report will later be written from the notes of every part, so extract everything now; do not write the report yet.

Write exhaustive bullet points in {language}, in chronological order, covering every idea, number, reference, recommendation, decision, deadline, task (with its owner when stated) and open question in this part. Keep timestamps and speaker names when present. Invent nothing, keep technical names and numbers exactly, and mark unclear speech recognition with [uncertain].

## User-provided context
{context}

## Glossary
{glossary}
"""

# Transcripts longer than this (in tokens) are summarized in parts, then merged (map-reduce):
# a 7B model keeps details better over several short prompts than over one very long one.
CHUNK_TOKENS = 12000  # about 20 minutes of speech
NOTES_MAX_TOKENS = 3000
REPORT_MAX_TOKENS = 4000


def split_transcript(text: str, count_tokens, max_tokens: int | None = None) -> list[str]:
    """Split on line boundaries (timestamps or speaker turns) into parts of at most max_tokens."""
    max_tokens = max_tokens or CHUNK_TOKENS
    parts, current, size = [], [], 0
    for line in text.splitlines():
        n = count_tokens(line) + 1
        if current and size + n > max_tokens:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        parts.append("\n".join(current))
    return parts


def chat_prompt(tokenizer, system: str, user: str) -> str:
    """The model's chat prompt. Some templates (Gemma, older Mistral) refuse a system message: it is
    then put at the start of the user message. enable_thinking=False keeps Qwen3 from writing its
    reasoning first; other templates ignore it."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": system + "\n\n" + user}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)


def _generate(llm, tokenizer, system: str, user: str, max_tokens: int, step: str, **attrs) -> str:
    from mlx_lm import stream_generate

    prompt = chat_prompt(tokenizer, system, user)
    with span(step, max_tokens=max_tokens, **attrs) as s:
        chunks, r = [], None
        for r in stream_generate(llm, tokenizer, prompt, max_tokens=max_tokens):
            chunks.append(r.text)
        s["input_tokens"], s["output_tokens"] = r.prompt_tokens, r.generation_tokens
        metric("prefill_tokens_per_s", r.prompt_tps)        # reading the input
        metric("generation_tokens_per_s", r.generation_tps)  # writing the output
        metric("mlx_peak_gb", r.peak_memory)
        if r.finish_reason == "length":
            event("truncated_output", step=step, output_tokens=r.generation_tokens, max_tokens=max_tokens)
            print(f"⚠️  {step}: output cut at {max_tokens} tokens, the end may be missing.")
    # Reasoning models may still write <think>...</think> before the answer.
    return re.sub(r"<think>.*?</think>", "", "".join(chunks), flags=re.DOTALL).strip()


def summarize(text: str, model: str, report_context: str, language: str | None,
              glossary: str = "No additional glossary provided.") -> str:
    from mlx_lm import load

    print(f"\n🧠 Writing the meeting report with {model} (first run: ~4 GB to download)...")
    with span("llm_loading", model=model):
        llm, tokenizer = load(model)
    context = report_context.strip() or "No additional context provided."
    report_language = REPORT_LANGUAGES.get(language, "the language of the transcript")
    system = PROMPT.format(context=context, language=report_language, glossary=glossary)

    parts = split_transcript(text, lambda line: len(tokenizer.encode(line)))
    metric("report_parts", len(parts))
    if len(parts) == 1:
        return _generate(llm, tokenizer, system, "Meeting transcript:\n\n" + text,
                         REPORT_MAX_TOKENS, "generation_llm", model=model)

    notes = []
    for i, part in enumerate(parts, 1):
        print(f"   part {i}/{len(parts)}...")
        notes_system = NOTES_PROMPT.format(index=i, total=len(parts), language=report_language,
                                           context=context, glossary=glossary)
        notes.append(_generate(llm, tokenizer, notes_system, "Transcript part:\n\n" + part,
                               NOTES_MAX_TOKENS, "notes_llm", model=model, part=i))
    merged = "\n\n".join(f"### Part {i}\n{n}" for i, n in enumerate(notes, 1))
    return _generate(llm, tokenizer, system,
                     "Detailed notes taken from the meeting transcript, part by part, in chronological "
                     "order:\n\n" + merged, REPORT_MAX_TOKENS, "generation_llm", model=model)


# ---------------------------------------------------------------- proofreading with the LLM

REVIEW_PROMPT = """You proofread an automatic speech-recognition transcript. Find the words the speech recognition misheard: misspelled names, acronyms and technical terms, and words that make no sense in context but sound like the right ones.

## Rules
1. Fix only misheard words. Do not change grammar, style, punctuation, word order, or spoken hesitations.
2. Do not guess: when unsure, leave the text as it is.
3. Spell names and terms from the glossary exactly as written there.

## User-provided context
{context}

## Glossary
{glossary}

## Answer format
The transcript lines are numbered. Answer with a JSON array only, one object per error, "wrong" copied exactly from the line:
[{{"line": 12, "wrong": "misheard words", "right": "corrected words"}}]
Answer [] if there is no error.
"""

# Short parts: a 7B model overlooks fewer errors, and each answer stays well within its limit.
REVIEW_CHUNK_TOKENS = 1500
REVIEW_MAX_TOKENS = 2000


def _whole_words(text: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)")


def parse_suggestions(answer: str, segments: list[dict]) -> list[dict]:
    """The corrections in the model's JSON answer, without those whose words are not in their line:
    the model sometimes invents or paraphrases the text it corrects."""
    match = re.search(r"\[.*\]", answer, re.DOTALL)
    try:
        items = json.loads(match.group()) if match else []
    except json.JSONDecodeError:
        event("unreadable_review", answer_chars=len(answer))
        return []
    suggestions = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        wrong, right = str(item.get("wrong", "")).strip(), str(item.get("right", "")).strip()
        if (0 <= i < len(segments) and wrong and right and wrong != right
                and _whole_words(wrong).search(segments[i]["text"])):
            suggestions.append({"segment": i, "wrong": wrong, "right": right})
    return suggestions


def suggest_corrections(segments: list[dict], model: str, context: str, glossary: str) -> list[dict]:
    """Corrections proposed by the LLM, as [{"segment": index, "wrong": ..., "right": ...}]."""
    from mlx_lm import load

    print(f"\n🧠 Proofreading the transcript with {model}...")
    with span("llm_loading", model=model):
        llm, tokenizer = load(model)
    system = REVIEW_PROMPT.format(context=context.strip() or "No additional context provided.",
                                  glossary=glossary)
    lines = "\n".join(f"{i}: {' '.join(s['text'].split())}" for i, s in enumerate(segments))
    parts = split_transcript(lines, lambda line: len(tokenizer.encode(line)), REVIEW_CHUNK_TOKENS)
    metric("review_parts", len(parts))
    suggestions = []
    for i, part in enumerate(parts, 1):
        if len(parts) > 1:
            print(f"   part {i}/{len(parts)}...")
        answer = _generate(llm, tokenizer, system, "Transcript lines:\n\n" + part,
                           REVIEW_MAX_TOKENS, "review_llm", model=model, part=i)
        suggestions += parse_suggestions(answer, segments)
    return suggestions


# ---------------------------------------------------------------- learning from corrections

def ask(question: str, choices: str = "ynq") -> str:
    """One of choices ('y', 'n', 'q' to quit...), asked until the answer is one of them."""
    while True:
        answer = input(f"{question} [{'/'.join(choices)}] ").strip().lower()[:1]
        if answer and answer in choices:
            return answer


def add_rules(corrections: list, terms: list, dictionary: dict, path: Path) -> int:
    """Asks about each proposed correction, then each term, and saves the accepted ones.
    Returns how many were accepted."""
    print("\nAccept a rule only if it is always right: it will apply to every future transcript.")
    accepted, new_terms = 0, []
    for wrong, right, n in corrections:
        answer = ask(f'  correction  "{wrong}" -> "{right}"  ({n}x)')
        if answer == "q":
            break
        if answer == "y":
            dictionary["corrections"][wrong] = right
            accepted += 1
    else:
        for term, n in terms:
            answer = ask(f'  term        "{term}"  ({n}x)')
            if answer == "q":
                break
            if answer == "y":
                new_terms.append(term)
    # New terms first: the Whisper prompt keeps the first terms when the list is too long.
    dictionary["terms"] = new_terms + dictionary["terms"]
    accepted += len(new_terms)
    if accepted:
        dictionary_file.save(dictionary, path)
        print(f"✅ {accepted} rule(s) added to {path}")
    return accepted


def learn(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="meeting.py learn",
                                description="Learn dictionary rules from a transcript corrected by hand")
    p.add_argument("meeting", type=Path, help="meeting directory, with segments.json and the edited transcript.md")
    p.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--telemetry-directory", type=Path,
                   default=Path(__file__).resolve().parent / "telemetry")
    args = p.parse_args(argv)

    segments = json.loads((args.meeting / "segments.json").read_text(encoding="utf-8"))
    edited = dictionary_file.transcript_words((args.meeting / "transcript.md").read_text(encoding="utf-8"))
    pipeline = dictionary_file.words(" ".join(s["text"] for s in segments))
    whisper = dictionary_file.words(" ".join(s.get("whisper_text", s["text"]) for s in segments))
    path = args.dictionary.expanduser()
    dictionary = dictionary_file.load(path)

    if args.no_telemetry:
        context = nullcontext()
    else:
        # Only the meeting date: the directory name may contain a private title.
        context = Telemetry("learn", directory=args.telemetry_directory,
                            tags={"meeting_date": args.meeting.resolve().name[:10]},
                            command=["meeting.py", "learn", "<redacted>"])
    with context:
        wer, wer_whisper = (dictionary_file.word_error_rate(edited, pipeline),
                            dictionary_file.word_error_rate(edited, whisper))
        metric("words", len(edited))
        metric("wer", wer)                  # transcript as produced, dictionary included
        metric("wer_whisper", wer_whisper)  # Whisper output, before the correction pass
        print(f"Word error rate: {wer:.1%} ({wer_whisper:.1%} before the dictionary corrections) "
              f"over {len(edited)} words")

        corrections, terms = dictionary_file.propose(pipeline, edited, dictionary)
        metric("proposed", len(corrections) + len(terms))
        if not corrections and not terms:
            print("No new correction or term found.")
            return
        metric("accepted", add_rules(corrections, terms, dictionary, path))


def review(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="meeting.py review",
                                description="Let the local LLM propose transcript corrections, confirm each one, "
                                            "then choose which ones to add to the dictionary")
    p.add_argument("meeting", type=Path, help="meeting directory, with segments.json and transcript.md")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--context", default="",
                   help="optional context for the model (project, people, terminology, etc.)")
    p.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--telemetry-directory", type=Path,
                   default=Path(__file__).resolve().parent / "telemetry")
    args = p.parse_args(argv)

    segments_path, transcript_path = args.meeting / "segments.json", args.meeting / "transcript.md"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    markdown = transcript_path.read_text(encoding="utf-8")
    # transcript.md is rewritten from segments.json: refuse rather than lose corrections made by hand.
    if dictionary_file.transcript_words(markdown) != dictionary_file.words(" ".join(s["text"] for s in segments)):
        sys.exit(f"{transcript_path} was edited by hand: review would overwrite your edits. "
                 f"Run `meeting.py learn {args.meeting}` to learn from them instead.")
    path = args.dictionary.expanduser()
    dictionary = dictionary_file.load(path)

    if args.no_telemetry:
        context = nullcontext()
    else:
        # Only the meeting date: the directory name may contain a private title.
        context = Telemetry("review", directory=args.telemetry_directory,
                            tags={"meeting_date": args.meeting.resolve().name[:10], "llm_model": args.llm_model},
                            command=["meeting.py", "review", "<redacted>"])
    with context:
        suggestions = suggest_corrections(segments, args.llm_model, args.context,
                                          dictionary_file.glossary(dictionary))
        groups: dict[tuple, list] = {}  # (wrong, right) -> segment indices, in transcript order
        for s in suggestions:
            groups.setdefault((s["wrong"], s["right"]), []).append(s["segment"])
        metric("proposed", len(groups))
        if not groups:
            print("The model found nothing to correct.")
            return

        print(f"\n{len(groups)} correction(s) proposed. y: apply, n: skip, e: type your own, q: stop.")
        applied = []
        for (wrong, right), indices in groups.items():
            indices = list(dict.fromkeys(indices))
            print()
            for i in indices[:2]:
                print(f"  [{hms(segments[i]['start'])}] {segments[i]['text'].replace(wrong, f'«{wrong}»')}")
            answer = ask(f'  "{wrong}" -> "{right}"  ({len(indices)}x)', "ynqe")
            if answer == "q":
                break
            if answer == "e":
                right = input(f'  "{wrong}" -> ').strip()
            if answer == "n" or not right or right == wrong:
                continue
            pattern, changed = _whole_words(wrong), 0
            for i in indices:
                s = segments[i]
                text = pattern.sub(lambda m: right, s["text"])
                if text != s["text"]:
                    # Whisper's own text, kept the first time a segment changes, for the word error rate.
                    s.setdefault("whisper_text", s["text"])
                    s["text"], changed = text, changed + 1
            if changed:
                applied.append((wrong, right, changed))
        metric("applied", len(applied))
        if not applied:
            return
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
        title = markdown.split("\n", 1)[0]
        transcript_path.write_text(f"{title}\n\n{format_transcript(segments)}\n", encoding="utf-8")
        print(f"\n✅ {len(applied)} correction(s) applied to {transcript_path}")

        corrections = [(w, r, n) for w, r, n in applied if dictionary["corrections"].get(w) != r]
        terms = Counter()
        for _, r, n in applied:
            if dictionary_file.looks_like_a_term(r) and r not in dictionary["terms"]:
                terms[r] += n
        if corrections or terms:
            metric("accepted", add_rules(corrections, terms.most_common(), dictionary, path))


# ---------------------------------------------------------------- main program

def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["learn"]:
        return learn(argv[1:])
    if argv[:1] == ["review"]:
        return review(argv[1:])
    p = argparse.ArgumentParser(description="Local meeting transcript")
    p.add_argument("--file", type=Path, help="existing audio file (otherwise: microphone recording)")
    p.add_argument("--title", default="", help="short title, added to the directory name")
    p.add_argument("--speakers", action="store_true", help="identify speakers (pyannote)")
    p.add_argument("--nb-speakers", type=int, help="number of people, if known (improves accuracy)")
    p.add_argument("--report", action="store_true", help="generate a meeting report with a local LLM")
    p.add_argument("--report-context", default="",
                   help="optional context to customize the report (project, audience, terminology, etc.)")
    p.add_argument("--vocabulary", help="technical words to guide Whisper (names, acronyms...)")
    p.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY,
                   help="personal dictionary of terms and corrections (default: dictionary.json next to "
                        "the script; ignored if missing)")
    p.add_argument("--language", default="fr",
                   help="language code spoken in the meeting (fr, en, ...), or 'auto' to detect it "
                        "(default: fr)")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    p.add_argument("--directory", type=Path, default=Path.cwd() / "Meetings",
                   help="where to save (default: ./Meetings in the current directory)")
    p.add_argument("--no-telemetry", action="store_true", help="do not measure resources")
    p.add_argument("--telemetry-directory", type=Path,
                   default=Path(__file__).resolve().parent / "telemetry",
                   help="where to store telemetry runs (default: ./telemetry next to the script)")
    args = p.parse_args(argv)
    if args.language == "auto":
        args.language = None  # Whisper detects the language from the first 30 seconds.

    if args.no_telemetry:
        context = nullcontext()
    else:
        private = {opt.lstrip("-").replace("-", "_") for opt in PRIVATE_OPTIONS}
        tags = {k: v for k, v in vars(args).items() if k not in private and v not in (None, False, "")}
        if args.file:
            tags["audio_format"] = args.file.suffix.lower()
        context = Telemetry("meeting", directory=args.telemetry_directory, tags=tags,
                            command=redacted_command(sys.argv))
    with context:
        run_pipeline(args)

    if isinstance(context, Telemetry):
        try:
            from analyze_telemetry import load_runs, run_report
            report = run_report(load_runs(context.directory.parent, run_id=context.id), context.id)
            print(f"📊 Resources: {report}")
        except Exception as e:
            optional_step_failed("Telemetry report", e, f"Raw measurements are in {context.directory}")


def run_pipeline(args: argparse.Namespace) -> None:
    name = dt.datetime.now().strftime("%Y-%m-%d_%Hh%M")
    if args.title:
        name += "_" + "".join(c if c.isalnum() else "-" for c in args.title)
    # Loaded before recording, so that a malformed file does not cost a whole meeting.
    dictionary = dictionary_file.load(args.dictionary.expanduser())
    metric("dictionary_terms", len(dictionary["terms"]))
    metric("dictionary_corrections", len(dictionary["corrections"]))
    output = args.directory / name
    output.mkdir(parents=True, exist_ok=True)

    if args.file:
        source = args.file.expanduser()
    else:
        source = output / "audio.wav"
        with span("recording", agent="audio"):
            record(source)

    with span("audio_loading", agent="audio", format=source.suffix.lower()):
        audio = load_audio(source)
    audio_duration = len(audio) / SR
    metric("audio_duration_s", audio_duration)
    print(f"Duration: {hms(audio_duration)}")
    if len(audio) < SR:
        sys.exit("Audio too short or empty: check microphone access.")

    terms = dictionary_file.prompt_terms(dictionary["terms"], args.vocabulary)
    metric("prompt_terms", len(terms))
    t0 = time.time()
    with span("transcript", agent="whisper", model=args.whisper_model, language=args.language) as s:
        segments = transcribe(audio, args.whisper_model, args.language,
                              dictionary_file.whisper_prompt(terms))
        s["segments"] = len(segments)
        # compute time / audio duration: below 1 means faster than real time
        metric("rtf", (time.time() - t0) / audio_duration)
    print(f"   {len(segments)} segments in {time.time() - t0:.0f} s")

    corrected = dictionary_file.apply_corrections(segments, dictionary["corrections"])
    metric("corrections_applied", corrected)
    if corrected:
        print(f"   {corrected} corrections from the dictionary")

    if args.speakers:
        try:
            with span("diarization", agent="pyannote", model=args.diarization_model) as s:
                assign_speakers(segments, diarize(audio, args.diarization_model, args.nb_speakers))
                s["speakers"] = len({x.get("speaker") for x in segments} - {None, "?"})
        except Exception as e:
            optional_step_failed("Speaker identification", e, "Continuing without it.")

    text = format_transcript(segments)
    (output / "transcript.md").write_text(f"# Transcript: {name}\n\n{text}\n", encoding="utf-8")
    (output / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ Transcript: {output / 'transcript.md'}")

    if args.report:
        try:
            with span("meeting_report", agent="llm", model=args.llm_model):
                report_text = summarize(text, args.llm_model, args.report_context, args.language,
                                        dictionary_file.glossary(dictionary, args.vocabulary))
            (output / "meeting_report.md").write_text(f"# Meeting report: {name}\n\n{report_text}\n", encoding="utf-8")
            print(f"✅ Meeting report: {output / 'meeting_report.md'}")
        except Exception as e:
            optional_step_failed("Meeting report", e, "The transcript was saved successfully.")

    print(f"\n📁 Directory: {output}")


if __name__ == "__main__":
    main()
