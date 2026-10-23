#!/usr/bin/env python3
"""analyze_telemetry.py: aggregate telemetry runs into tables and charts.

Usage:
    python analyze_telemetry.py                 # report every run and global comparison
    python analyze_telemetry.py --name meeting  # only runs named meeting
    python analyze_telemetry.py --run 3f2a91bc   # report one run
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GO = 1024 ** 3
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # Implementation note.
TEXTE, TEXTE_2, GRILLE, STEP_BACKGROUND = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150,
    "axes.edgecolor": GRILLE, "axes.labelcolor": TEXTE_2, "text.color": TEXTE,
    "xtick.color": TEXTE_2, "ytick.color": TEXTE_2, "axes.grid": True, "grid.color": GRILLE,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
})


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Implementation note.
    return lines


def load_runs(directory: Path | str = "telemetry", name: str | None = None,
            run_id: str | None = None) -> dict[str, pd.DataFrame]:
    """Load runs from `directory`. With run_id, read only that run's directory."""
    pattern = f"*_{run_id}/run.json" if run_id else "*/run.json"  # Implementation note.
    runs, events, samples = [], [], []
    for d in sorted(Path(directory).glob(pattern)):
        meta = json.loads(d.read_text(encoding="utf-8"))
        if name and meta["name"] != name:
            continue
        rid = meta["id"]
        runs.append({"run_id": rid, "directory": str(d.parent), "name": meta["name"], "start": meta["start"],
                     "duration_s": meta.get("duration_s"), "status": meta["status"],
                     "git_commit": meta.get("git_commit"), "git_modified": meta.get("git_modified"),
                     "chip": meta["machine"]["chip"], "ram_go": meta["machine"]["ram"] / GO,
                     **{f"tag_{k}": v for k, v in meta.get("tags", {}).items()},
                     **{f"version_{k}": v for k, v in meta.get("packages", {}).items()}})
        events += [{"run_id": rid, "run_start": meta["start"], **e} for e in _jsonl(d.parent / "events.jsonl")]
        samples += [{"run_id": rid, **s, "t_rel": s["t"] - meta["start"]}
                    for s in _jsonl(d.parent / "samples.jsonl")]

    ev = pd.DataFrame(events)
    by_type = {t: ev[ev["type"] == t].dropna(axis=1, how="all").reset_index(drop=True)
                for t in ("span", "metric", "event")} if not ev.empty else {}
    spans = by_type.get("span", pd.DataFrame())
    if not spans.empty:
        attrs = pd.json_normalize(spans["attrs"].tolist()).add_prefix("attr_")
        spans = pd.concat([spans.drop(columns="attrs"), attrs], axis=1)
        spans["start_rel"] = spans["start"] - spans["run_start"]
        spans["end_rel"] = spans["end"] - spans["run_start"]
    metrics = by_type.get("metric", pd.DataFrame())
    if not metrics.empty and "tags" in metrics:
        metrics = pd.concat([metrics.drop(columns="tags"),
                               pd.json_normalize(metrics["tags"].tolist()).add_prefix("tag_")], axis=1)
    events = by_type.get("event", pd.DataFrame())
    if not events.empty and "data" in events:
        events = pd.concat([events.drop(columns="data"),
                                pd.json_normalize(events["data"].tolist())], axis=1)
    return {"runs": pd.DataFrame(runs), "spans": spans, "metrics": metrics,
            "events": events, "samples": pd.DataFrame(samples)}


def step_summary(spans: pd.DataFrame) -> pd.DataFrame:
    g = spans.groupby(["agent", "name"], dropna=False)
    r = g["duration_s"].agg(n="count", average_duration_s="mean", p50_duration_s="median",
                         p95_duration_s=lambda x: x.quantile(0.95), max_duration_s="max")
    r["average_cpu_pct"] = g["average_cpu_pct"].mean()
    r["rss_peak_gb"] = g["rss_peak"].max() / GO
    for col, output in (("gpu_memory_peak", "gpu_memory_peak_gb"), ("mlx_peak", "mlx_peak_gb")):
        if col in spans:
            r[output] = g[col].max() / GO
    if "average_gpu_util_pct" in spans:
        r["average_gpu_util_pct"] = g["average_gpu_util_pct"].mean()
    r["errors"] = g["status"].apply(lambda s: (s != "ok").sum())
    return r.round(3).reset_index().sort_values("average_duration_s", ascending=False)


# Implementation note.

def _step_bands(ax, spans_run: pd.DataFrame, labels: bool) -> None:
    roots = spans_run[spans_run["parent_id"].isna()] if "parent_id" in spans_run else spans_run
    total = spans_run["end_rel"].max() / 60
    for i, s in enumerate(roots.sort_values("start_rel").itertuples()):
        a, b = s.start_rel / 60, s.end_rel / 60
        ax.axvspan(a, b, color=STEP_BACKGROUND if i % 2 == 0 else "#e8e7e2", zorder=0, lw=0)
        if labels and (b - a) > 0.08 * total:  # Implementation note.
            ax.text((a + b) / 2, 1.02, s.name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXTE_2, clip_on=False)


def timeline(d: dict, run_id: str, output: Path) -> Path:
    smp = d["samples"][d["samples"]["run_id"] == run_id]
    spans_run = d["spans"][d["spans"]["run_id"] == run_id] if not d["spans"].empty else d["spans"]
    panels = [("Process CPU (cores)", [("process_cpu_pct", "process + children", 1 / 100)])]
    memory = [(c, lbl, 1 / GO) for c, lbl in (("rss", "process (RSS)"), ("gpu_mem", "GPU (machine)"),
                                              ("mlx_active", "MLX active")) if c in smp]
    panels.append(("Memory (Go)", memory))
    if "gpu_util_pct" in smp:
        panels.append(("GPU utilization (%)", [("gpu_util_pct", "GPU (machine)", 1)]))

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    for k, (ax, (title, series)) in enumerate(zip(axes, panels)):
        _step_bands(ax, spans_run, labels=k == 0) if not spans_run.empty else None
        for i, (col, lbl, scale) in enumerate(series):
            # Implementation note.
            pts = smp[smp[col].notna()]
            ax.plot(pts["t_rel"] / 60, pts[col] * scale, color=SERIES[i], label=lbl)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(bottom=0)
        if len(series) > 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axes[-1].set_xlabel("Time since start (min)")
    fig.suptitle(f"Resources during run {run_id}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path = output / "timeline.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _label(r) -> str:
    return f"{r['agent']} · {r['name']}" if pd.notna(r["agent"]) else r["name"]


def duration_by_step(spans: pd.DataFrame, output: Path) -> Path:
    spans = spans.assign(step=spans.apply(_label, axis=1))
    order = spans.groupby("step")["duration_s"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.2))
    y = range(len(order))
    ax.barh(y, order.varead, color=SERIES[0], height=0.55, zorder=2)
    for i, step in enumerate(order.index):
        v = spans.loc[spans["step"] == step, "duration_s"]
        ax.scatter(v, [i] * len(v), s=14, color=TEXTE, zorder=3, linewidths=0)
        ax.text(order[step], i, f"  {order[step]:.1f} s", va="center", fontsize=8, color=TEXTE_2)
    ax.set_yticks(list(y), order.index, fontsize=9)
    ax.set_xlabel("Duration (s): bar = average, point = one run")
    ax.grid(axis="y", visible=False)
    ax.set_title("Duration by step", loc="left", fontsize=11)
    fig.tight_layout()
    path = output / "duration_by_step.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def memory_by_step(spans: pd.DataFrame, output: Path) -> Path:
    spans = spans.assign(step=spans.apply(_label, axis=1))
    cols = [(c, lbl) for c, lbl in (("rss_peak", "process (RSS)"), ("mlx_peak", "MLX active"),
                                    ("gpu_memory_peak", "GPU (machine)")) if c in spans]
    peaks = spans.groupby("step")[[c for c, _ in cols]].max().div(GO)
    peaks = peaks.loc[peaks.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 0.25 * len(cols) * len(peaks) + 1.4))
    h = 0.8 / len(cols)
    for i, (c, lbl) in enumerate(cols):
        # Implementation note.
        pos = [k + ((len(cols) - 1) / 2 - i) * h for k in range(len(peaks))]
        ax.barh(pos, peaks[c], height=h - 0.06, color=SERIES[i], label=lbl, zorder=2)
    ax.set_yticks(range(len(peaks)), peaks.index, fontsize=9)
    ax.set_xlabel("Memory peak (Go)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title("Memory peak by step", loc="left", fontsize=11)
    fig.tight_layout()
    path = output / "memory_by_step.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def metric_charts(metrics: pd.DataFrame, output: Path) -> list[Path]:
    if metrics.empty or "step" not in metrics:
        return []
    with_step = metrics.dropna(subset=["step"])
    (output / "metrics").mkdir(exist_ok=True)
    paths = []
    for name, df in with_step.groupby("name"):
        runs = df["run_id"].unique()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for i, rid in enumerate(runs[-3:]):  # Implementation note.
            r = df[df["run_id"] == rid].sort_values("step")
            ax.plot(r["step"], r["valeur"], color=SERIES[i], label=rid, marker="o", markersize=4)
        if len(runs) > 1:
            ax.legend(fontsize=8, title="run", title_fontsize=8)
        ax.set_xlabel("step")
        ax.set_title(name, loc="left", fontsize=11)
        fig.tight_layout()
        path = output / "metrics" / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def _filtrer(d: dict, run_id: str) -> dict[str, pd.DataFrame]:
    return {k: df[df["run_id"] == run_id] if "run_id" in df else df for k, df in d.items()}


def _tables_et_charts(d: dict, output: Path) -> pd.DataFrame | None:
    """CSV, step summaries, durations, memory, and metric charts for runs in `d`."""
    output.mkdir(parents=True, exist_ok=True)
    for name, df in d.items():
        df.to_csv(output / f"{name}.csv", index=False)
    r = None
    if not d["spans"].empty:
        r = step_summary(d["spans"])
        r.to_csv(output / "step_summary.csv", index=False)
        duration_by_step(d["spans"], output)
        memory_by_step(d["spans"], output)
    metric_charts(d["metrics"], output)
    return r


def run_report(d: dict, run_id: str) -> Path:
    """Report for one run, stored in its own directory: <run>/report/."""
    dr = _filtrer(d, run_id)
    output = Path(dr["runs"]["directory"].iloc[0]) / "report"
    _tables_et_charts(dr, output)
    if not dr["samples"].empty:
        timeline(dr, run_id, output)
    return output


def run_summary(d: dict) -> pd.DataFrame:
    """One line per run: duration, status, code version, and resource peaks."""
    runs = d["runs"].copy()
    runs["date"] = pd.to_datetime(runs["start"], unit="s").dt.tz_localize("UTC").dt.tz_convert(
        "Europe/Paris").dt.strftime("%Y-%m-%d %H:%M:%S")
    columns = ["run_id", "date", "name", "status", "duration_s", "git_commit", "git_modified"]
    r = runs[[c for c in columns if c in runs]].set_index("run_id")
    smp, spans = d["samples"], d["spans"]
    for col, output in (("rss", "rss_peak_gb"), ("mlx_active", "mlx_peak_gb"), ("gpu_mem", "gpu_memory_peak_gb")):
        if col in smp:
            r[output] = smp.groupby("run_id")[col].max() / GO
    if "process_cpu_pct" in smp:
        r["average_cpu_cores"] = smp.groupby("run_id")["process_cpu_pct"].mean() / 100
    if not spans.empty:
        r["nb_spans"] = spans.groupby("run_id").size()
        r["error_spans"] = spans[spans["status"] != "ok"].groupby("run_id").size()
        r["error_spans"] = r["error_spans"].fillna(0).astype(int)
    return r.round(3).reset_index().sort_values("date")


def evolution_by_run(spans: pd.DataFrame, runs: pd.DataFrame, output: Path) -> Path | None:
    """Duration of each step from one run to the next (identify regressions)."""
    path = output / "evolution_by_run.png"
    if runs["run_id"].nunique() < 2:
        path.unlink(missing_ok=True)  # Implementation note.
        return None
    spans = spans.assign(step=spans.apply(_label, axis=1))
    # Implementation note.
    roots = spans[spans["parent_id"].isna()] if "parent_id" in spans else spans
    top = roots.groupby("step")["duration_s"].mean().nlargest(8).index
    order = runs.sort_values("start")["run_id"].tolist()
    table = (roots[roots["step"].isin(top)].groupby(["run_id", "step"])["duration_s"].sum()
             .unstack().reindex(order))
    palette = SERIES + ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(order) + 3), 4))
    for i, step in enumerate(top):
        ax.plot(range(len(order)), table[step], color=palette[i], label=step, marker="o", markersize=5)
    starts = runs.set_index("run_id")["start"]
    labels = [pd.to_datetime(starts[r], unit="s").tz_localize("UTC").tz_convert("Europe/Paris")
                  .strftime("%m-%d %H:%M") + f"\n{r}" for r in order]
    ax.set_xticks(range(len(order)), labels, fontsize=8)
    ax.set_xlabel("run (chronological order)")
    ax.set_ylabel("Duration (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title("Duration of steps from one run to the next", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def comparison(d: dict, output: Path) -> pd.DataFrame | None:
    """All runs together: which steps cost what, and how they change."""
    r = _tables_et_charts(d, output)
    run_summary(d).to_csv(output / "run_summary.csv", index=False)
    if not d["spans"].empty:
        evolution_by_run(d["spans"], d["runs"], output)
    return r


def main() -> None:
    p = argparse.ArgumentParser(description="Tables and charts from ./telemetry")
    p.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "telemetry")
    p.add_argument("--name", help="filter by run name (e.g. meeting)")
    p.add_argument("--run", help="run ID: rebuild only its report")
    args = p.parse_args()

    if args.run:
        d = load_runs(args.directory, run_id=args.run)
        if d["runs"].empty:
            raise SystemExit(f"Run {args.run} not found in {args.directory}")
        print(f"📁 Report : {run_report(d, args.run)}")
        return

    d = load_runs(args.directory, args.name)
    if d["runs"].empty:
        raise SystemExit(f"No run in {args.directory}")

    for run_id in d["runs"].sort_values("start")["run_id"]:
        print(f"📁 {run_report(d, run_id)}")
    r = comparison(d, args.directory / "_comparison")
    print(f"\n{len(d['runs'])} run(s), {len(d['spans'])} spans, {len(d['metrics'])} metrics")
    if r is not None:
        print("\n" + r.to_string(index=False))
    print(f"\n📁 Comparison: {args.directory / '_comparison'}")


if __name__ == "__main__":
    main()
