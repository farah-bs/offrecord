#!/usr/bin/env python3
"""report_comparison.py: which local LLM writes the best meeting report from the same transcript?

Each model writes the report through meeting.summarize, exactly as `meeting.py --report` does,
and each report is scored without another model:

    facts      share of a checklist of facts (numbers, names) that appear in the report
    invented   numbers in the report that appear nowhere in the transcript
    sections   required report sections present, out of 10
    terms      glossary terms spelled exactly as in the dictionary
    french     share of French function words among French and English ones

Usage:
    python experiments/report_comparison.py --transcript Meetings/<dir>/transcript.md \\
        --facts experiments/meeting.facts.json --dictionary dictionary.json --free-disk

The facts file lists, for each fact, a label and regular expressions (any one matching counts,
case-insensitive):

    {"facts": [{"label": "178 publications", "patterns": ["178"]},
               {"label": "PRISMA method", "patterns": ["prisma"]}]}

Each run is recorded with the project's telemetry (telemetry/<date>_report_comparison_<id>/):
the reports, results.csv, outputs.jsonl and report_comparison.png, saved again after each model.
With --free-disk, a model downloaded for the comparison is removed from the Hugging Face cache
once it has been scored. After an interruption, --resume <that folder> skips the models already
scored and retries the failed ones; the new measurements go to a new telemetry run, and the
results stay together in the first folder.
"""

import argparse
import csv
import gc
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dictionary as dictionary_file  # noqa: E402
import meeting  # noqa: E402
from analyze_telemetry import SERIES  # noqa: E402
from telemetry import Telemetry, metric, span  # noqa: E402

# Instruction-tuned models of 7 to 9 B parameters, 4-bit, from mlx-community.
MODELS = (
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Ministral-8B-Instruct-2410-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Llama-3.1-Tulu-3-8B-4bit",
    "mlx-community/gemma-2-9b-it-4bit",
    "mlx-community/granite-3.3-8b-instruct-4bit",
    "mlx-community/OLMo-2-1124-7B-Instruct-4bit",
    "mlx-community/aya-expanse-8b-4bit",
    "mlx-community/Lucie-7B-Instruct-4bit",
    "mlx-community/internlm3-8b-instruct-4bit",
)

# The 10 report sections, with the French titles a model may translate them to.
SECTIONS = (
    ("Summary", r"summary|résumé|synthèse"),
    ("Scientific points", r"scientific|scientifiques"),
    ("Advice", r"advice|recommendation|conseils|recommandations"),
    ("Research directions", r"research directions|directions de recherche|pistes"),
    ("References", r"references|références|ressources"),
    ("Decisions", r"decisions|décisions"),
    ("Milestones", r"milestones|deadlines|jalons|échéances"),
    ("TODO list", r"todo|à faire|tâches"),
    ("Open questions", r"open questions|questions ouvertes"),
    ("Uncertain passages", r"uncertain|incertains"),
)

COLUMNS = ["model", "facts", "invented", "sections", "terms", "french", "words", "total_s", "output_tokens",
           "generation_tps", "prefill_tps", "mlx_peak_gb", "truncated", "telemetry_run", "error"]

NUMBER = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
FRENCH = {"le", "la", "les", "des", "est", "une", "dans", "pour", "que", "qui", "sur", "avec", "pas"}
ENGLISH = {"the", "and", "is", "of", "to", "in", "that", "with", "for", "are", "this", "on", "not"}


def numbers(text: str) -> set[str]:
    """Numbers of at least 2 characters, normalized ("10 000" -> "10000", "18,6" -> "18.6")."""
    found = {re.sub(r"[   ]", "", n).replace(",", ".") for n in NUMBER.findall(text)}
    return {n for n in found if len(n) >= 2}


def spelled(term: str, text: str) -> bool:
    """The term, spelled exactly, as whole words; a version number may follow ("AlphaGeometry2")."""
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?![^\W\d])", text) is not None


def score(report: str, transcript: str, facts: list[dict], terms: list[str]) -> dict:
    headings = [h.lower() for h in re.findall(r"^#{1,4}\s*(.+)$", report, flags=re.MULTILINE)]
    words = re.findall(r"\w+", report.lower())
    fr, en = sum(w in FRENCH for w in words), sum(w in ENGLISH for w in words)
    covered = [f["label"] for f in facts
               if any(re.search(p, report, flags=re.IGNORECASE) for p in f["patterns"])]
    invented = sorted(numbers(report) - numbers(transcript))
    return {
        "facts": len(covered) / len(facts),
        "facts_missing": [f["label"] for f in facts if f["label"] not in covered],
        "invented": len(invented),
        "invented_numbers": invented,
        "sections": sum(any(re.search(p, h) for h in headings) for _, p in SECTIONS),
        "terms": sum(spelled(t, report) for t in terms),
        "french": fr / max(fr + en, 1),
        "words": len(words),
    }


def llm_metrics(events_path: Path, span_id: str) -> dict:
    """Token counts and speeds logged by meeting._generate under one model's span."""
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    children = {e["span_id"] for e in events if e["type"] == "span" and e.get("parent_id") == span_id}
    grandchildren = {e["span_id"] for e in events if e["type"] == "span" and e.get("parent_id") in children}
    llm_spans = [e for e in events if e["type"] == "span" and e["span_id"] in children | grandchildren
                 and e["name"] in ("notes_llm", "generation_llm")]
    calls = {e["span_id"] for e in llm_spans}
    values = {}
    for e in events:
        if e["type"] == "metric" and e.get("span_id") in calls:
            values.setdefault(e["name"], []).append(e["value"])
    return {
        "output_tokens": sum(e["attrs"].get("output_tokens", 0) for e in llm_spans),
        "generation_tps": round(sum(values.get("generation_tokens_per_s", [0])) /
                                max(len(values.get("generation_tokens_per_s", [])), 1), 1),
        "prefill_tps": round(sum(values.get("prefill_tokens_per_s", [0])) /
                             max(len(values.get("prefill_tokens_per_s", [])), 1), 1),
        "mlx_peak_gb": round(max(values.get("mlx_peak_gb", [0])), 2),
        "truncated": sum(1 for e in events if e["type"] == "event" and e["name"] == "truncated_output"
                         and e.get("span_id") in calls),
    }


def cached(repo: str) -> bool:
    from huggingface_hub import scan_cache_dir
    return any(r.repo_id == repo for r in scan_cache_dir().repos)


def remove_from_cache(repo: str) -> None:
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    hashes = [rev.commit_hash for r in cache.repos if r.repo_id == repo for rev in r.revisions]
    if hashes:
        cache.delete_revisions(*hashes).execute()


def model_spans(events_path: Path) -> dict[str, dict]:
    """The last completed model_report span of each model in a run."""
    if not events_path.exists():
        return {}
    spans = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["type"] == "span" and e["name"] == "model_report" and e["status"] == "ok":
            spans[e["attrs"]["model"]] = e
    return spans


def previous_results(folder: Path, transcript: str, facts: list[dict], terms: list[str]) -> dict[str, dict]:
    """Results already in a comparison folder, to resume it. Models scored before outputs.jsonl was
    saved after each model are rebuilt from their saved report and the folder's telemetry."""
    results = {}
    outputs = folder / "outputs.jsonl"
    if outputs.exists():
        for line in outputs.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            results[row["model"]] = row
    events_path = folder / "events.jsonl"
    for model, sp in model_spans(events_path).items():
        report = folder / "reports" / (model.split("/")[-1] + ".md")
        if model in results and "error" not in results[model] or not report.exists():
            continue
        row = {"model": model, "telemetry_run": folder.name.rsplit("_", 1)[-1],
               "total_s": round(sp["duration_s"], 1)}
        row.update(score(report.read_text(encoding="utf-8"), transcript, facts, terms))
        row.update(llm_metrics(events_path, sp["span_id"]))
        results[model] = row
    print(f"Resuming {folder.name}: {sum('error' not in r for r in results.values())} model(s) already scored")
    return results


def save(folder: Path, results: list[dict]) -> None:
    with open(folder / "outputs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    with open(folder / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    if any("error" not in r for r in results):
        chart(results, folder / "report_comparison.png")


def chart(results: list[dict], path: Path) -> None:
    rows = sorted([r for r in results if "error" not in r], key=lambda r: r["facts"])
    names = [r["model"].split("/")[-1].replace("-4bit", "") for r in rows]
    panels = (("facts", "Facts covered (%)", 100), ("invented", "Invented numbers", 1),
              ("total_s", "Time to write (s)", 1), ("generation_tps", "Writing speed (tokens/s)", 1))
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 0.45 * len(rows) + 1.6), sharey=True)
    for ax, (key, title, factor) in zip(axes, panels):
        values = [r[key] * factor for r in rows]
        bars = ax.barh(range(len(rows)), values, 0.6, color=SERIES[0])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3, color="#333333")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.margins(x=0.18)
    axes[0].set_yticks(range(len(rows)), names, fontsize=9)
    fig.suptitle("Meeting report: same transcript, same prompt, one model per row",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare local LLMs on the meeting report")
    p.add_argument("--transcript", type=Path, required=True, help="a transcript.md")
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, help="dictionary.json, for the glossary and term scores")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--language", default="fr")
    p.add_argument("--report-context", default="")
    p.add_argument("--free-disk", action="store_true",
                   help="remove each model downloaded for the comparison from the cache once scored")
    p.add_argument("--resume", type=Path, metavar="FOLDER",
                   help="continue a comparison folder: skip models already scored, retry failed ones")
    p.add_argument("--telemetry-directory", type=Path, default=ROOT / "telemetry")
    args = p.parse_args()

    import mlx.core as mx

    transcript = args.transcript.read_text(encoding="utf-8")
    text = transcript.split("\n", 2)[-1]  # without the "# Transcript: ..." title, as in the pipeline
    facts = json.loads(args.facts.read_text(encoding="utf-8"))["facts"]
    for f in facts:
        if not any(re.search(pt, transcript, flags=re.IGNORECASE) for pt in f["patterns"]):
            print(f"⚠️  fact not found in the transcript itself: {f['label']}")
    d = dictionary_file.load(args.dictionary) if args.dictionary else {"terms": [], "corrections": {}}
    glossary = dictionary_file.glossary(d)
    terms = list(dict.fromkeys(d["terms"] + list(d["corrections"].values())))

    out = args.resume or None
    results = previous_results(out, transcript, facts, terms) if out else {}
    # The transcript, facts and dictionary may be private: only counts are tagged.
    with Telemetry("report_comparison", directory=args.telemetry_directory,
                   tags={"models": len(args.models), "facts": len(facts), "terms": len(terms),
                         "resumed": bool(out)},
                   command=["report_comparison.py", "<redacted>"]) as t:
        out = out or t.directory  # a resumed comparison keeps its results in its first folder
        (out / "reports").mkdir(exist_ok=True)
        if not (out / "transcript.md").exists():
            shutil.copy(args.transcript, out / "transcript.md")
        for model in args.models:
            if model in results and "error" not in results[model]:
                print(f"\n=== {model}: already scored, skipped")
                continue
            was_cached = cached(model)
            row = {"model": model, "telemetry_run": t.id}
            print(f"\n=== {model}{'' if was_cached else ' (download)'}")
            try:
                with span("model_report", agent="llm", model=model) as s:
                    t0 = time.time()
                    report = meeting.summarize(text, model, args.report_context, args.language, glossary)
                    row["total_s"] = round(time.time() - t0, 1)
                    s["report_chars"] = len(report)
                (out / "reports" / (model.split("/")[-1] + ".md")).write_text(report, encoding="utf-8")
                row.update(score(report, transcript, facts, terms))
                for name in ("facts", "invented", "sections", "terms", "french"):
                    metric(f"report_{name}", row[name], model=model)
                events_path = t.directory / "events.jsonl"
                row.update(llm_metrics(events_path, model_spans(events_path)[model]["span_id"]))
            except Exception as e:
                row["error"] = repr(e)
                traceback.print_exc()
            finally:
                gc.collect()
                (mx.clear_cache if hasattr(mx, "clear_cache") else mx.metal.clear_cache)()
                if args.free_disk and not was_cached:
                    remove_from_cache(model)
                    print(f"   removed {model} from the cache")
            results[model] = row
            save(out, [results[m] for m in args.models if m in results])  # an interruption loses one model at most
            if "error" not in row:
                print(f"   facts {row['facts']:.0%}  invented {row['invented']}  sections {row['sections']}/10"
                      f"  terms {row['terms']}/{len(terms)}  french {row['french']:.0%}  {row['total_s']} s")
        save(out, [results[m] for m in args.models if m in results])
    print(f"\n📁 {out}")

if __name__ == "__main__":
    main()
