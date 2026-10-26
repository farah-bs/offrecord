import json

from conftest import read_jsonl

import telemetry
from telemetry import metric, span


def test_run_metadata_and_status(telemetry_run):
    meta = json.loads((telemetry_run.directory / "run.json").read_text())
    assert meta["status"] == "ok"
    assert meta["duration_s"] > 0
    assert meta["machine"]["ram"] > 0


def test_spans_are_nested_and_measured(telemetry_run):
    spans = {e["name"]: e for e in read_jsonl(telemetry_run.directory / "events.jsonl") if e["type"] == "span"}
    assert spans["child"]["parent_id"] == spans["parent"]["span_id"]
    assert spans["child"]["agent"] == "planner"  # inherited from the parent span
    assert spans["child"]["attrs"] == {"iteration": 0, "items": 3}
    assert spans["child"]["duration_s"] >= 0.3
    assert spans["child"]["rss_peak"] > 0
    assert spans["broken"]["status"] == "error" and "boom" in spans["broken"]["error"]


def test_metrics_are_attached_to_current_span(telemetry_run):
    events = read_jsonl(telemetry_run.directory / "events.jsonl")
    parent_id = next(e["span_id"] for e in events if e["type"] == "span" and e["name"] == "parent")
    losses = [e for e in events if e["type"] == "metric" and e["name"] == "loss"]
    assert [m["step"] for m in losses] == [0, 1, 2]
    assert all(m["span_id"] == parent_id for m in losses)


def test_sampler_process_writes_samples(telemetry_run):
    samples = [s for s in read_jsonl(telemetry_run.directory / "samples.jsonl") if "rss" in s]
    assert samples, "the sampler subprocess wrote no CPU/memory samples"
    assert telemetry_run._sampler.poll() is not None, "the sampler subprocess is still running"


def test_api_is_a_no_op_outside_a_run():
    assert telemetry.active() is None
    with span("ignored") as attrs:
        attrs["x"] = 1
    metric("ignored", 1.0)
