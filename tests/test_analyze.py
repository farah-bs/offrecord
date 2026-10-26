from analyze_telemetry import comparison, load_runs, run_report


def test_run_report_writes_tables_and_charts(telemetry_run):
    d = load_runs(telemetry_run.directory.parent, run_id=telemetry_run.id)
    assert list(d["runs"]["run_id"]) == [telemetry_run.id]

    report = run_report(d, telemetry_run.id)
    for name in ("spans.csv", "step_summary.csv", "timeline.png", "duration_by_step.png",
                 "memory_by_step.png", "metrics/loss.png"):
        assert (report / name).exists(), name


def test_comparison_includes_run_summary(telemetry_run, tmp_path):
    d = load_runs(telemetry_run.directory.parent)
    summary = comparison(d, tmp_path / "_comparison")
    assert set(summary["name"]) == {"parent", "child", "broken"}
    assert summary.loc[summary["name"] == "broken", "errors"].item() == 1
    assert (tmp_path / "_comparison" / "run_summary.csv").exists()
