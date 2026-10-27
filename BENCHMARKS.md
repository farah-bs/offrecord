# Benchmarks: protocol

How offrecord's model choices are measured, and how anyone can check the numbers. These rules are
written, and committed, before the benchmarks are run: the commit history shows they were not
adjusted to fit the results.

> **Status**: the scorer and its checks are implemented (`benchmarks/score.py`, tested in
> `tests/test_benchmark_score.py`). The runners and the first results are not published yet; this
> page will link them when they are.

## Questions answered

| Question | Measure |
|---|---|
| Why Whisper large-v3-turbo? | Word error rate, speed and memory of each Whisper size, full precision and 4-bit |
| Does the personal dictionary help? | Spelling of names and jargon, and word error rate, over successive meetings |
| Does the pipeline invent text during silences? | Phrases produced on silence and noise, with and without the filter |
| Who speaks when, how well? | Diarization error rate of pyannote 3.1 |
| Why this report model? | Facts recalled, content invented, language, speed and memory of each local LLM |
| Is the LLM proofreading (`review`) useful? | Share of its proposals that are right, and word error rate before and after |
| Does anything leave the machine? | Network connections during a full run |

## Data

Public meeting recordings with human transcripts, so that the numbers can be checked and compared.

**AMI Meeting Corpus** (English, CC BY 4.0): the test set of the official *full-corpus-ASR*
partition, used by Kaldi, ESPnet and Hugging Face. 16 meetings, about 9 hours:

| Series | Meetings |
|---|---|
| ES2004 | ES2004a, ES2004b, ES2004c, ES2004d |
| IS1009 | IS1009a, IS1009b, IS1009c, IS1009d |
| TS3003 | TS3003a, TS3003b, TS3003c, TS3003d |
| EN2002 | EN2002a, EN2002b, EN2002c, EN2002d |

Each meeting is measured with two microphones:

- **headset mix** (`Mix-Headset`): close microphones, the condition most published results use;
- **one distant microphone** (`Array1-01`): a microphone on the table, closest to a laptop in a
  meeting room, so closest to what offrecord users record.

Settings are tuned only on the *dev* set of the same partition; the test set is run once they are
frozen.

**SUMM-RE** (French, CC BY-SA 4.0): spontaneous meeting-style conversations. The subset, and whether
it holds reference summaries for the report, will be fixed here before any French result is
published. Outputs derived from it keep its share-alike license.

The full corpora (about 100 hours each) are not used by default: published results are measured on
the test sets, which makes ours comparable, and 9 hours already narrow the confidence intervals below
the differences between models.

## Rules

1. **The real pipeline.** Benchmarks call the functions `meeting.py` uses, with its settings; nothing
   is reimplemented. A test checks that a benchmark transcript is identical to the `segments.json`
   that `meeting.py --file` writes for the same audio.
2. **Everything pinned.** Each result records the dataset revision and a checksum of each audio file,
   the Hugging Face revision of each model, the offrecord commit, package versions, and the machine
   (chip, memory, macOS version, power source).
3. **Deterministic outputs.** Whisper keeps its default decoding, which falls back to sampling when
   unsure, with a fixed random seed; LLMs use greedy decoding. One meeting per configuration is run
   twice, and both outputs must be identical.
4. **The same conditions for every model.** Same audio, language forced (`en` for AMI, `fr` for
   SUMM-RE), same prompt and output limit for every LLM.
5. **No test data in the dictionary.** Building it from the test transcripts would inflate the
   results. AMI meetings come in series of four with the same people (a, b, c, d), so the dictionary
   is learned the way `learn` does it: from meeting a, tested on b; from a and b, tested on c; and so
   on.
6. **Fact checklists written first.** For the report, the facts each meeting should yield are listed
   from its human transcript and summary, and committed before any model output is produced.
7. **Every result reported**, including models that fail, with the reason. No best-of-n runs.

## Metrics

| Metric | Definition | Tool |
|---|---|---|
| Word error rate (WER) | (substitutions + deletions + insertions) / reference words, the fewest edits, over the whole corpus | [jiwer](https://github.com/jitsi/jiwer) |
| Normalization | English: Whisper's English normalizer (case, punctuation, hesitations, numbers, spellings). Other languages: Whisper's basic normalizer, accents kept | [whisper-normalizer](https://github.com/kurianbenoy/whisper_normalizer) |
| Diarization error rate (DER) | Missed speech + false alarm + speaker confusion, over speech time; no forgiveness collar, overlapping speech scored | pyannote.metrics |
| Term accuracy | Share of dictionary terms said in the audio that the transcript spells exactly | `experiments/prompt_comparison.py` |
| Real-time factor | Compute time / audio duration, without download and model loading; median of 3 runs after one warm-up | offrecord telemetry |
| Peak memory | Highest MLX memory during the step | offrecord telemetry |
| Report facts | Share of the checklist found in the report; numbers and names absent from the transcript count as invented | `experiments/report_comparison.py` |

The word error rate is computed over the whole corpus, so that a long meeting weighs more than a
short one, and reported with its substitutions, deletions and insertions.

## Statistics

- **95 % confidence intervals** by bootstrap over meetings (10,000 resamples, fixed seed). Meetings,
  not words, are resampled: errors within a meeting are correlated.
- **Paired comparisons**: every system is scored on the same meetings, and the difference between
  two systems is resampled meeting by meeting. "A is better than B" is claimed only when the
  interval of the difference excludes zero.

## Checking the scorer

A wrong scorer makes every number wrong, so it is checked four ways:

1. **Known answers.** `score.self_check()` runs before every benchmark and stops it on any wrong
   answer: a transcript against itself scores 0 %, an empty output 100 %, one substitution in six
   words 16.7 %, and normalization removes case, punctuation and hesitations but keeps French
   accents.
2. **Agreement with the reference library.** The `learn` command's word error rate is tested against
   jiwer. This check found that the previous implementation, based on `difflib`, counted up to
   twice the real errors in 8 % of random cases; it now uses jiwer.
3. **Published figures.** The [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
   scores Whisper models on AMI's test set, cut into short utterances. The same models are run on
   the same utterances with this scorer; a gap of more than one point means the normalization or the
   setup is wrong, and no result is published until it is explained. offrecord's own numbers come
   from whole meetings, which adds the errors of long-form transcription, so they are reported
   separately.
4. **Manual inspection.** The 20 segments with the most errors are read, to catch scoring artifacts
   (numbers, hesitations, spellings) rather than real recognition errors.

## Reproducing the results

Each published result comes with its raw outputs (transcripts and reports) and a `results.json`
holding the context listed in rule 2. Re-scoring the published outputs needs no model and no GPU,
so anyone can check that the published tables follow from them.

## Limits

These benchmarks do not measure accents and jargon beyond those of AMI and SUMM-RE, Macs other than
the ones results were submitted from, or how useful people find the reports. Numbers measured on
English meetings say little about French ones, and the reverse.
