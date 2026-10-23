"""
telemetry.py: measurement of resources (CPU, memory, Apple GPU) and metric logging
for today's meeting pipeline and tomorrow's multi-agent system.

Usage:

    from telemetry import Telemetry, span, trace, metric, event

    with Telemetry("meeting", tags={"phase": "dev"}):
        with span("transcript", agent="whisper", model="large-v3-turbo") as s:
            ...
            s["segments"] = 284                      # attribute added during execution
            metric("rtf", 0.11)                    # attached to the current span and agent

        for i in range(5):                           # feedback / training loop
            with span("critique", agent="relecteur", iteration=i):
                metric("score", 0.7 + i / 20, step=i)
                event("feedback", iteration=i, accepte=i == 4)

    @trace(agent="planner")                    # same behavior as a decorator (sync or async)
    def planifier(...): ...

Outside a `with Telemetry(...)` block, span/trace/metric/event are no-ops:
instrumented code runs normally without telemetry.

Each run creates telemetry/<date>_<name>_<id>/ :
    run.json        machine, package versions, git commit, command line, status, duration
    samples.jsonl   samples every `interval` seconds (CPU, memory, GPU, MLX/MPS)
    events.jsonl    spans (timed steps + resources), metrics, events

CPU / memory / GPU sampling runs in a separate process: MLX and PyTorch keep
the GIL during long computations, and a Python thread would miss entire minutes.
Only MLX / MPS memory, readable only from the process, is sampled
by a thread (it may have gaps; spans also measure it at their boundaries).

Known limitations :
- CPU and memory are measured for the process (and its children, e.g. ffmpeg), not per thread.
  If two agents run in parallel in the same process, their spans overlap and
  see the same resources: to distinguish them, launch agents in separate processes
  separately, or read the timeline in samples.jsonl.
- RSS does not count MLX / MPS tensors (unified memory allocated through Metal): inspect
  mlx_active / mps_allocated for the process, gpu_mem for the whole machine.
- GPU: global machine GPU utilization and memory (ioreg, without sudo).
- Energy : non measured (powermetrics exige sudo).
- In a manually created thread, the parent span is not propagated automatically:
  use contextvars.copy_context().run(...) or a ThreadPoolExecutor through asyncio.
"""

import asyncio
import contextvars
import datetime as dt
import functools
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import psutil

TRACKED_PACKAGES = ("mlx", "mlx-lm", "mlx-whisper", "torch", "pyannote.audio", "numpy", "transformers")

_active: "Telemetry | None" = None
_current_span: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)

_IOREG = re.compile(r'"(Device Utilization %|In use system memory)"=(\d+)')


def _command(*cmd: str, cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _apple_gpu() -> dict:
    """Global GPU usage and memory (Apple Silicon), read in ioreg without sudo."""
    output = _command("ioreg", "-r", "-d", "1", "-c", "IOAccelerator") or ""
    values = dict(_IOREG.findall(output))
    r = {}
    if "Device Utilization %" in values:
        r["gpu_util_pct"] = int(values["Device Utilization %"])
    if "In use system memory" in values:
        r["gpu_mem"] = int(values["In use system memory"])
    return r


def _framework_memory() -> dict:
    """Memory allocated by MLX / PyTorch MPS, only when already imported."""
    r = {}
    if "mlx.core" in sys.modules:
        try:
            mx = sys.modules["mlx.core"]
            r["mlx_active"] = mx.get_active_memory()
            r["mlx_cache"] = mx.get_cache_memory()
        except Exception:
            pass
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            if torch.backends.mps.is_available():
                r["mps_allocated"] = torch.mps.current_allocated_memory()
        except Exception:
            pass
    return r


class _ProcessProbe:
    """CPU and memory for a process and its descendants (excluding excluded processes)."""

    def __init__(self, pid: int, excluded: set[int] | None = None):
        self.proc = psutil.Process(pid)
        self.excluded = excluded or set()
        self._children: dict[int, psutil.Process] = {}
        self.proc.cpu_percent(None)

    def process(self) -> list[psutil.Process]:
        # Implementation note.
        try:
            live = {c.pid: c for c in self.proc.children(recursive=True) if c.pid not in self.excluded}
        except psutil.Error:
            live = {}
        for pid in list(self._children):
            if pid not in live:
                del self._children[pid]
        for pid, c in live.items():
            if pid not in self._children:
                self._children[pid] = c
                try:
                    c.cpu_percent(None)
                except psutil.Error:
                    pass
        return [self.proc, *self._children.varead()]

    def cpu_et_rss(self) -> tuple[float, int]:
        cpu, rss = 0.0, 0
        for p in self.process():
            try:
                cpu += p.cpu_percent(None)
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        return cpu, rss

    def rss(self) -> int:
        total = 0
        for p in self.process():
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total

    def temps_cpu(self) -> float:
        """Cumulative CPU time (user + sys) for the process and its descendants, in seconds."""
        total = 0.0
        for p in self.process():
            try:
                t = p.cpu_times()
                total += t.user + t.system
                if p is self.proc:
                    total += t.children_user + t.children_system  # Implementation note.
            except psutil.Error:
                pass
        return total


def _sampler(pid: int, path: str, interval: float) -> None:
    """Process loop d'sampling : runs until the observed process ends."""
    probe = _ProcessProbe(pid, excluded={os.getpid()})
    psutil.cpu_percent(None)
    with open(path, "a", encoding="utf-8") as f:
        while probe.proc.is_running():
            time.sleep(interval)
            try:
                cpu, rss = probe.cpu_et_rss()
                vm = psutil.virtual_memory()
                e = {
                    "t": round(time.time(), 3),
                    "process_cpu_pct": round(cpu, 1),           # Implementation note.
                    "system_cpu_pct": psutil.cpu_percent(None),  # Implementation note.
                    "rss": rss,
                    "system_memory_used": vm.total - vm.available,
                    "swap_used": psutil.swap_memory().used,
                    **_apple_gpu(),
                }
            except psutil.NoSuchProcess:
                break
            f.write(json.dumps(e) + "\n")
            f.flush()


class Telemetry:
    def __init__(self, name: str, directory: Path | str = "telemetry", interval: float = 1.0,
                 tags: dict | None = None):
        self.name = name
        self.interval = interval
        self.tags = tags or {}
        self.id = uuid.uuid4().hex[:8]
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%Hh%M%S")
        self.directory = Path(directory) / f"{timestamp}_{name}_{self.id}"
        self._lock = threading.Lock()
        self._samples: list[dict] = []  # Implementation note.
        self._offset = 0
        self._stop = threading.Event()
        self._previous = None

    # Implementation note.

    def __enter__(self):
        global _active
        self.directory.mkdir(parents=True, exist_ok=True)
        self._samples_path = self.directory / "samples.jsonl"
        self._events_file = open(self.directory / "events.jsonl", "a", encoding="utf-8")
        self._samples_file = open(self._samples_path, "a", encoding="utf-8")
        self.start = time.time()
        self.meta = self._metadata()
        self._write_metadata(status="in_progress")
        self._sampler = subprocess.Popen(
            [sys.executable, "-I", __file__, "--sampler", str(os.getpid()),
             str(self._samples_path), str(self.interval)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        self._probe = _ProcessProbe(os.getpid(), excluded={self._sampler.pid})
        self._thread = threading.Thread(target=self._framework_loop, name="telemetry", daemon=True)
        self._thread.start()
        self._previous, _active = _active, self
        return self

    def __exit__(self, typ, exc, tb):
        global _active
        self._stop.set()
        self._thread.join(timeout=5)
        self._sampler.terminate()
        try:
            self._sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._sampler.kill()
        _active = self._previous
        status = "ok" if typ is None else ("interrompu" if typ is KeyboardInterrupt else "error")
        self._write_metadata(status=status, error=repr(exc) if exc else None)
        self._events_file.close()
        self._samples_file.close()
        return False

    def _metadata(self) -> dict:
        root = Path(__file__).resolve().parent
        versions = {}
        for p in TRACKED_PACKAGES:
            try:
                versions[p] = metadata.version(p)
            except metadata.PackageNotFoundError:
                pass
        git_status = _command("git", "status", "--porcelain", cwd=root)
        return {
            "id": self.id,
            "name": self.name,
            "tags": self.tags,
            "commande": sys.argv,
            "git_commit": _command("git", "rev-parse", "--short", "HEAD", cwd=root),
            "git_modified": bool(git_status) if git_status is not None else None,
            "machine": {
                "hote": platform.node(),
                "chip": _command("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
                "os": platform.platform(),
                "ram": psutil.virtual_memory().total,
                "coeurs": psutil.cpu_count(),
            },
            "python": sys.version.split()[0],
            "packages": versions,
            "interval_s": self.interval,
        }

    def _write_metadata(self, status: str, error: str | None = None) -> None:
        end = time.time()
        meta = {**self.meta, "start": self.start, "end": end if status != "in_progress" else None,
                "duration_s": round(end - self.start, 3) if status != "in_progress" else None,
                "status": status, "error": error}
        tmp = self.directory / "run.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.directory / "run.json")

    # Implementation note.

    def _framework_loop(self) -> None:
        while not self._stop.wait(self.interval):
            mem = _framework_memory()
            if mem:
                with self._lock:
                    self._samples_file.write(json.dumps({"t": round(time.time(), 3), **mem}) + "\n")
                    self._samples_file.flush()

    def _reload_samples(self) -> None:
        """Adds complete lines written since the last read."""
        with self._lock:
            with open(self._samples_path, "rb") as f:
                f.seek(self._offset)
                block = f.read()
            end = block.rfind(b"\n") + 1
            self._offset += end
            for line in block[:end].splitlines():
                try:
                    self._samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Implementation note.

    def _write(self, evt: dict) -> None:
        current = _current_span.get()
        evt.setdefault("t", round(time.time(), 3))
        evt.setdefault("span_id", current[0] if current else None)
        evt.setdefault("agent", current[1] if current else None)
        with self._lock:
            self._events_file.write(json.dumps(evt, ensure_ascii=False, default=str) + "\n")
            self._events_file.flush()

    def metric(self, name: str, valeur: float, step: int | None = None, **tags) -> None:
        self._write({"type": "metric", "name": name, "valeur": valeur, "step": step, "tags": tags})

    def event(self, name: str, **data) -> None:
        self._write({"type": "event", "name": name, "data": data})

    @contextmanager
    def span(self, name: str, agent: str | None = None, **attrs):
        parent = _current_span.get()
        sid = uuid.uuid4().hex[:12]
        agent = agent or (parent[1] if parent else None)
        jeton = _current_span.set((sid, agent))
        start = time.time()
        cpu0 = self._probe.temps_cpu()
        rss0 = self._probe.rss()
        mem0 = _framework_memory()
        status, error = "ok", None
        try:
            yield attrs
        except BaseException as e:
            status, error = ("interrompu" if isinstance(e, KeyboardInterrupt) else "error"), repr(e)
            raise
        finally:
            end = time.time()
            duration = end - start
            cpu_s = self._probe.temps_cpu() - cpu0
            rss1 = self._probe.rss()
            mem1 = _framework_memory()
            _current_span.reset(jeton)
            self._reload_samples()
            fenetre = [s for s in self._samples if start <= s["t"] <= end]

            def peak(cle, *bornes):
                v = [s[cle] for s in fenetre if cle in s] + [b for b in bornes if b is not None]
                return max(v) if v else None

            def average(cle):
                v = [s[cle] for s in fenetre if cle in s]
                return round(sum(v) / len(v), 1) if v else None

            self._write({
                "type": "span", "t": round(end, 3), "span_id": sid,
                "parent_id": parent[0] if parent else None, "agent": agent, "name": name,
                "start": round(start, 3), "end": round(end, 3), "duration_s": round(duration, 3),
                "cpu_s": round(cpu_s, 3),
                "average_cpu_pct": round(100 * cpu_s / duration, 1) if duration > 0 else None,
                "rss_start": rss0, "rss_end": rss1, "rss_peak": peak("rss", rss0, rss1),
                "gpu_memory_peak": peak("gpu_mem"),
                "average_gpu_util_pct": average("gpu_util_pct"), "gpu_util_peak_pct": peak("gpu_util_pct"),
                "mlx_peak": peak("mlx_active", mem0.get("mlx_active"), mem1.get("mlx_active")),
                "mps_peak": peak("mps_allocated", mem0.get("mps_allocated"), mem1.get("mps_allocated")),
                "sample_count": sum(1 for s in fenetre if "rss" in s),
                "thread": threading.current_thread().name,
                "status": status, "error": error, "attrs": attrs,
            })


# Implementation note.

@contextmanager
def _empty_span(attrs):
    yield attrs


def span(name: str, agent: str | None = None, **attrs):
    return _active.span(name, agent, **attrs) if _active else _empty_span(attrs)


def metric(name: str, valeur: float, step: int | None = None, **tags) -> None:
    if _active:
        _active.metric(name, valeur, step, **tags)


def event(name: str, **data) -> None:
    if _active:
        _active.event(name, **data)


def active() -> "Telemetry | None":
    return _active


def trace(name: str | None = None, agent: str | None = None, **attrs):
    """Decorator: each function call becomes a span."""
    def deco(f):
        n = name or f.__qualname__
        if asyncio.iscoroutinefunction(f):
            @functools.wraps(f)
            async def enveloppe_async(*a, **k):
                with span(n, agent, **attrs):
                    return await f(*a, **k)
            return enveloppe_async

        @functools.wraps(f)
        def enveloppe(*a, **k):
            with span(n, agent, **attrs):
                return f(*a, **k)
        return enveloppe
    return deco


if __name__ == "__main__" and sys.argv[1:2] == ["--sampler"]:
    _sampler(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
