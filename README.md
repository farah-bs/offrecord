# Meeting Assistant

Local meeting assistant for macOS Apple Silicon. The project records or loads audio, transcribes it with Whisper through MLX, can identify speakers with pyannote, and can produce a detailed meeting report with a local LLM. Audio and text remain on the machine: no remote API call is required.

> **Project status**: the pipeline is designed for an Apple Silicon Mac and Python 3.13, or a version compatible with the MLX packages in use.

## Features

- recording from the microphone or transcription of an existing file;
- local English transcription with `mlx-community/whisper-large-v3-turbo`;
- filtering of common Whisper hallucinations during silence;
- optional diarization with `pyannote/speaker-diarization-3.1`;
- structured English meeting reports with `mlx-community/Qwen2.5-7B-Instruct-4bit`;
- local telemetry: duration, CPU, memory, Apple GPU, MLX/MPS memory, metrics, and pipeline steps;
- CSV reports and charts for analyzing runs.

## Requirements

1. An Apple Silicon Mac.
2. [Homebrew](https://brew.sh/) and `ffmpeg`:

   ```bash
   brew install ffmpeg
   ```

3. Python 3.13 recommended.
4. For microphone recording, allow microphone access for Terminal (or the application running Python) in **System Settings > Privacy & Security > Microphone**.
5. For diarization, a Hugging Face account, an access token, and acceptance of the terms for the relevant pyannote models.

## Step-by-step installation

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy psutil pandas matplotlib sounddevice soundfile
python -m pip install mlx mlx-lm mlx-whisper
python -m pip install torch pyannote.audio
```

MLX packages are specific to Apple Silicon. If installing `pyannote.audio` or PyTorch fails, first install Python and package versions compatible with the MLX environment, then rerun the last command.

Verify that the active interpreter is the virtual environment:

```bash
which python
python --version
python -c "import mlx, mlx_whisper, psutil, pandas, matplotlib"
```

Hugging Face models are downloaded automatically on first use. Plan for several gigabytes of free space, especially for the LLM.

### Enable diarization

Transcription works without a Hugging Face token. For `--speakers`:

1. create a Hugging Face token with read permissions;
2. accept the terms for `pyannote/speaker-diarization-3.1` on Hugging Face;
3. export it only in the current shell:

   ```bash
   export HF_TOKEN="hf_..."
   ```

Never add this token to Git, a tracked `.env` file, or a shared report.

## Usage

### 1. Transcribe an audio file

```bash
source .venv/bin/activate
python meeting.py --file "/path/to/meeting.m4a" --title "project-update"
```

Formats readable by `ffmpeg` are accepted. For a mono 16 kHz WAV file, playback is direct; other formats or sample rates go through the MLX audio loader.

### 2. Record from the microphone

```bash
python meeting.py --title "project-update"
```

Speak normally and press **Enter** to stop recording. The WAV file is written progressively to the meeting folder.

### 3. Add speakers

```bash
HF_TOKEN="hf_..." python meeting.py \
  --file "/path/to/meeting.m4a" \
  --speakers \
  --nb-speakers 3 \
  --title "team-meeting"
```

`--nb-speakers` is optional, but can improve accuracy when known. If diarization fails, the transcript is kept and the program continues without speaker attribution.

### 4. Generate a meeting report

```bash
python meeting.py \
  --file "/path/to/meeting.m4a" \
  --report \
  --title "team-meeting"
```

The report is generated locally with the default Qwen model. Change the model with `--llm-model` if needed. `--vocabulary` accepts technical terms to guide Whisper:

```bash
python meeting.py \
  --file meeting.m4a \
  --report \
  --vocabulary "neuro-symbolic,MLX,pyannote,agentic"
```

Customize the report for a specific project or audience with `--report-context`.
This text is sent only to the local LLM together with the transcript and is
excluded from telemetry metadata:

```bash
python meeting.py \
  --file meeting.m4a \
  --report \
  --report-context "Project: robotics planning. Audience: engineering team. Focus on decisions, risks, and owners."
```

For longer context, put the text in a local file and pass it through the shell:

```bash
python meeting.py \
  --file meeting.m4a \
  --report \
  --report-context "$(cat ./private-report-context.txt)"
```

Keep personal names, affiliations, supervisor details, and other sensitive
information in local context files rather than in the source code. Do not
commit those files.

### 5. Choose the output folder and language

```bash
python meeting.py \
  --file meeting.wav \
  --language en \
  --directory ./my-meetings \
  --whisper-model mlx-community/whisper-large-v3-turbo
```

By default, results are written to `Meetings/YYYY-MM-DD_HHhMM[_title]/`. Each run produces at least:

- `transcript.md`: readable transcript;
- `segments.json`: segments with start, end, text, and optional speaker;
- `audio.wav`: only when recording is performed;
- `meeting_report.md`: only with `--report`.

## Telemetry and reports

Telemetry is enabled by default and recorded in `telemetry/<date>_meeting_<id>/`. It remains local and includes `run.json`, `events.jsonl`, and `samples.jsonl`.

Disable measurement for a run:

```bash
python meeting.py --file meeting.wav --no-telemetry
```

Analyze all runs and produce CSV files and charts:

```bash
python analyze_telemetry.py
```

Analyze only `meeting` runs:

```bash
python analyze_telemetry.py --name meeting
```

Analyze one run:

```bash
python analyze_telemetry.py --run ed6c3e64
```

Reports are created in each run folder, with the global comparison in `telemetry/_comparison/`. The `spans.csv`, `step_summary.csv`, `timeline.png`, and other outputs allow comparison of duration, CPU, and memory by step.

## Layout

```text
.
├── meeting.py              # recording, transcription, and report CLI
├── telemetry.py           # resource instrumentation and collection
├── analyze_telemetry.py   # CSV aggregation and charts
├── Meetings/               # audio and text outputs
└── telemetry/             # telemetry runs and reports
```

## Troubleshooting

- **`No module named ...`**: check `which python` and activate `.venv`, then install the missing package in that environment.
- **Microphone inaccessible**: grant permission to Terminal, then restart Python.
- **`ffmpeg` not found**: run `brew install ffmpeg` and check `which ffmpeg`.
- **pyannote model refused**: accept the Hugging Face terms and verify `echo "$HF_TOKEN"` without publishing its value.
- **Insufficient memory**: start without `--report` or `--speakers`, use a shorter file, or select smaller models.
- **Report generation failed**: the transcript already written remains available; check disk space and the LLM model download.

## Privacy and generated files

Recordings, transcripts, reports, and telemetry traces may contain sensitive information. Check files before publishing them. Python caches, system files, and private runtime outputs are ignored by Git.

## License

No license is currently defined.
