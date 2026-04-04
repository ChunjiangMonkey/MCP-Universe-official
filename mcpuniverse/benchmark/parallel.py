"""
Parallel benchmark runner for MCP-Universe.

Orchestrates concurrent execution of benchmark tasks by splitting a multi-task
YAML config into single-task configs and running each in a separate subprocess.

Usage:
    # Orchestrator mode — single config
    python -m mcpuniverse.benchmark.parallel config.yaml --concurrency 20

    # Orchestrator mode — multiple configs pooled into one run
    python -m mcpuniverse.benchmark.parallel a.yaml b.yaml c.yaml --concurrency 20

    # Orchestrator mode — sequential runs from settings file
    python -m mcpuniverse.benchmark.parallel config.yaml --settings-file settings.yaml

    # Worker mode (called internally by orchestrator)
    python -m mcpuniverse.benchmark.parallel --worker config.yaml --output result.json --log-file trace.log
"""
import argparse
import asyncio
import copy
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque

import psutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from mcpuniverse.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
)
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks
from mcpuniverse.evaluator import EvaluationResult
from mcpuniverse.tracer.collectors.file import FileCollector


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorCategory(Enum):
    """Categories for classifying worker subprocess errors."""
    RATE_LIMIT = auto()
    TIMEOUT = auto()
    CONNECTION = auto()
    AUTH = auto()
    UNKNOWN = auto()


_ERROR_PATTERNS: List[Tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"ratelimiterror", re.IGNORECASE), ErrorCategory.RATE_LIMIT),
    (re.compile(r"\b429\b"), ErrorCategory.RATE_LIMIT),
    (re.compile(r"rate.?limit", re.IGNORECASE), ErrorCategory.RATE_LIMIT),
    (re.compile(r"too many requests", re.IGNORECASE), ErrorCategory.RATE_LIMIT),
    (re.compile(r"quota.?exceeded", re.IGNORECASE), ErrorCategory.RATE_LIMIT),
    (re.compile(r"timeout", re.IGNORECASE), ErrorCategory.TIMEOUT),
    (re.compile(r"timed?\s*out", re.IGNORECASE), ErrorCategory.TIMEOUT),
    (re.compile(r"connectionerror", re.IGNORECASE), ErrorCategory.CONNECTION),
    (re.compile(r"connection.?refused", re.IGNORECASE), ErrorCategory.CONNECTION),
    (re.compile(r"connection.?reset", re.IGNORECASE), ErrorCategory.CONNECTION),
    (re.compile(r"ECONNREFUSED", re.IGNORECASE), ErrorCategory.CONNECTION),
    (re.compile(r"\b401\b"), ErrorCategory.AUTH),
    (re.compile(r"\b403\b"), ErrorCategory.AUTH),
    (re.compile(r"unauthorized", re.IGNORECASE), ErrorCategory.AUTH),
    (re.compile(r"forbidden", re.IGNORECASE), ErrorCategory.AUTH),
    (re.compile(r"invalid.?api.?key", re.IGNORECASE), ErrorCategory.AUTH),
]


def classify_error(stderr: str) -> ErrorCategory:
    """Classify a worker's stderr output into an error category.

    Scans *stderr* against known patterns and returns the first match.
    Returns ``ErrorCategory.UNKNOWN`` when no pattern matches.
    """
    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(stderr):
            return category
    return ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Worker result
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """Outcome of a single worker subprocess."""
    task_path: str
    success: bool
    evaluation_results: Optional[List[Dict]] = None
    category: str = ""
    temp_yaml: str = ""
    trace_id: str = ""
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    stderr: str = ""
    stdout: str = ""
    return_code: int = -1
    duration_seconds: float = 0.0
    attempt: int = 1
    start_time: str = ""
    end_time: str = ""
    error_message: str = ""
    error_type: str = ""
    error_traceback: str = ""
    output_json: str = ""
    log_path: str = ""


def _truncate_text(text: str, limit: int = 2000) -> str:
    """Return *text* truncated to *limit* chars while preserving the tail."""
    if not text or len(text) <= limit:
        return text
    head = max(limit // 2, 1)
    tail = max(limit - head - len("\n...\n"), 1)
    return f"{text[:head]}\n...\n{text[-tail:]}"


def _best_effort_task_path_from_config(config_path: str) -> str:
    """Best-effort extraction of the single task path from a worker config."""
    try:
        docs = _parse_config_documents(config_path)
    except Exception:
        return ""

    for doc in docs:
        if doc.get("kind", "").lower() != "benchmark":
            continue
        tasks = doc.get("spec", {}).get("tasks", [])
        if tasks:
            return str(tasks[0])
    return ""


def _build_worker_error_payload(
    *,
    config_path: str,
    output_path: str,
    log_file: str,
    start_time: str,
    end_time: str,
    exc: Exception,
) -> Dict[str, Any]:
    """Build a structured worker error payload for the result JSON."""
    return {
        "task_path": _best_effort_task_path_from_config(config_path),
        "error": str(exc),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_traceback": traceback.format_exc(),
        "config_path": config_path,
        "output_path": output_path,
        "log_file": log_file,
        "start_time": start_time,
        "end_time": end_time,
    }


def _worker_failure_details(wr: WorkerResult) -> Dict[str, Any]:
    """Convert a failed worker result into a compact JSON-serializable dict."""
    return {
        "error_category": wr.error_category.name,
        "error_type": wr.error_type,
        "error_message": wr.error_message,
        "error_traceback": wr.error_traceback,
        "return_code": wr.return_code,
        "attempt": wr.attempt,
        "duration_seconds": round(wr.duration_seconds, 2),
        "start_time": wr.start_time,
        "end_time": wr.end_time,
        "output_json": wr.output_json,
        "log_path": wr.log_path,
        "stderr_excerpt": _truncate_text(wr.stderr),
        "stdout_excerpt": _truncate_text(wr.stdout),
    }


@dataclass
class RunSetting:
    """Single sequential benchmark setting."""
    name: str
    model_name: str
    llm_type: Optional[str] = None
    agent_type: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    concurrency: Optional[int] = None
    use_custom_tools: Optional[bool] = None
    github_tokens: Optional[str] = None
    cleanup_github_repos_after_run: Optional[bool] = None


@dataclass(frozen=True)
class TaskItem:
    """Single benchmark task prepared for pooled execution."""

    task_path: str
    temp_yaml: str
    category: str
    source_config: str


# ---------------------------------------------------------------------------
# GitHub token loading
# ---------------------------------------------------------------------------

def _load_github_tokens(csv_path: str) -> List[Tuple[str, str]]:
    """Load GitHub accounts from a CSV file (username,token per line).

    Returns a list of ``(username, token)`` tuples.
    Raises ``ValueError`` if the file contains no valid entries.
    """
    accounts: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace("\r", "").strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                continue
            accounts.append((parts[0].strip(), parts[1].strip()))
    if not accounts:
        raise ValueError(f"No valid GitHub accounts found in {csv_path}")
    return accounts


def _cleanup_github_repos(github_tokens: Optional[str] = None) -> None:
    """Best-effort cleanup of repos created during a benchmark run."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(repo_root, "cleanup_github_repos.sh")
    cmd = ["bash", script_path]
    if github_tokens:
        cmd.append(github_tokens)

    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[cleanup] Completed GitHub repo cleanup using {os.path.basename(script_path)}")
        if completed.stdout.strip():
            print(completed.stdout.strip())
    except FileNotFoundError:
        print(f"[cleanup] Skipped: script not found at {script_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[cleanup] Failed with exit code {exc.returncode}")
        if exc.stdout.strip():
            print(exc.stdout.strip())
        if exc.stderr.strip():
            print(exc.stderr.strip())


# ---------------------------------------------------------------------------
# Adaptive semaphore
# ---------------------------------------------------------------------------

class AdaptiveSemaphore:
    """An asyncio semaphore whose concurrency limit can be changed at runtime.

    Built on :class:`asyncio.Condition` so that :meth:`set_limit` can wake up
    waiters when the limit is *increased*, and simply block new acquisitions
    when the limit is *decreased* (no pre-emption of already-running tasks).
    """

    def __init__(self, initial_limit: int, min_limit: int = 1) -> None:
        if initial_limit < 1:
            raise ValueError("initial_limit must be >= 1")
        if min_limit < 1:
            raise ValueError("min_limit must be >= 1")
        self._limit = initial_limit
        self._min_limit = min_limit
        self._current = 0  # number of currently acquired slots
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def current(self) -> int:
        return self._current

    async def set_limit(self, new_limit: int) -> None:
        """Change the concurrency limit at runtime.

        If *new_limit* is larger than the old limit, waiting acquirers are
        woken up.  If smaller, no running tasks are interrupted — the
        semaphore simply blocks new acquisitions until enough slots are freed.
        """
        new_limit = max(new_limit, self._min_limit)
        async with self._condition:
            old = self._limit
            self._limit = new_limit
            if new_limit > old:
                self._condition.notify_all()

    async def acquire(self) -> None:
        async with self._condition:
            while self._current >= self._limit:
                await self._condition.wait()
            self._current += 1

    async def release(self) -> None:
        async with self._condition:
            self._current -= 1
            self._condition.notify()

    async def __aenter__(self) -> "AdaptiveSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        await self.release()


# ---------------------------------------------------------------------------
# Feedback controller
# ---------------------------------------------------------------------------

class FeedbackController:
    """Adaptive delay *and* concurrency controller driven by recent worker
    error signals.

    Maintains a sliding window of the most recent *window_size* results and
    computes a cooldown period based on the frequency of each error category.

    Additionally applies an **AIMD** (Additive Increase / Multiplicative
    Decrease) strategy to dynamically adjust the concurrency limit exposed
    via the :attr:`semaphore` property:

    * **RATE_LIMIT** error  → halve concurrency (``current // 2``)
    * **CONNECTION** error  → reduce concurrency by 25 % (``current * 3 // 4``)
    * Other error categories → no concurrency change (handled by delay only)
    * Consecutive successes ≥ ``window_size // 2`` → increase by 1 (up to
      *initial_concurrency*)
    * Memory usage exceeds *memory_threshold* → halve concurrency (``current // 2``)
    """

    def __init__(
        self,
        window_size: int = 20,
        max_cooldown: float = 60.0,
        initial_concurrency: int = 10,
        min_concurrency: int = 1,
        memory_threshold: float = 0.8,
    ):
        self._window: deque[WorkerResult] = deque(maxlen=window_size)
        self._window_size = window_size
        self._max_cooldown = max_cooldown
        self._lock = asyncio.Lock()

        self._initial_concurrency = initial_concurrency
        self._min_concurrency = min_concurrency
        self._memory_threshold = memory_threshold
        self._semaphore = AdaptiveSemaphore(
            initial_limit=initial_concurrency,
            min_limit=min_concurrency,
        )
        self._success_streak: int = 0

    @property
    def semaphore(self) -> AdaptiveSemaphore:
        """The adaptive semaphore used to gate concurrent workers."""
        return self._semaphore

    async def record(self, result: WorkerResult) -> None:
        """Record a worker result and apply AIMD concurrency adjustment."""
        self._window.append(result)
        await self._aimd_adjust(result)

    def _compute_cooldown(self) -> float:
        if not self._window:
            return 0.0

        rate_count = sum(1 for r in self._window if r.error_category == ErrorCategory.RATE_LIMIT)
        timeout_count = sum(1 for r in self._window if r.error_category == ErrorCategory.TIMEOUT)
        connection_count = sum(1 for r in self._window if r.error_category == ErrorCategory.CONNECTION)

        cooldown = rate_count * 5.0 + timeout_count * 2.0 + connection_count * 3.0

        # Recovery: halve cooldown when success rate >= 80%
        success_count = sum(1 for r in self._window if r.success)
        if len(self._window) > 0 and (success_count / len(self._window)) >= 0.80:
            cooldown /= 2.0

        return min(cooldown, self._max_cooldown)

    async def _aimd_adjust(self, result: WorkerResult) -> None:
        """Apply AIMD concurrency adjustment based on *result*."""
        cur = self._semaphore.limit

        if not result.success:
            self._success_streak = 0
            if result.error_category == ErrorCategory.RATE_LIMIT:
                new = max(cur // 2, self._min_concurrency)
            elif result.error_category == ErrorCategory.CONNECTION:
                new = max(cur * 3 // 4, self._min_concurrency)
            else:
                return  # TIMEOUT / AUTH / UNKNOWN — delay only
            if new != cur:
                await self._semaphore.set_limit(new)
        else:
            self._success_streak += 1
            if (
                self._success_streak >= self._window_size // 2
                and cur < self._initial_concurrency
            ):
                await self._semaphore.set_limit(min(cur + 1, self._initial_concurrency))
                self._success_streak = 0

        # Memory pressure check
        mem = psutil.virtual_memory()
        if mem.percent > self._memory_threshold * 100:
            cur = self._semaphore.limit  # re-read after possible AIMD change
            new = max(cur // 2, self._min_concurrency)
            if new != cur:
                await self._semaphore.set_limit(new)

    async def wait_before_launch(self) -> float:
        """Sleep for the computed cooldown period. Returns the actual delay."""
        async with self._lock:
            delay = self._compute_cooldown()
        if delay > 0:
            await asyncio.sleep(delay)
        return delay


# ---------------------------------------------------------------------------
# YAML splitting
# ---------------------------------------------------------------------------

def _parse_config_documents(config_path: str) -> List[dict]:
    """Read a multi-document YAML file and return the list of documents."""
    with open(config_path, "r", encoding="utf-8") as f:
        return list(yaml.safe_load_all(f))


def _resolve_config_path(config_path: str) -> str:
    """Resolve a config path using benchmark default folder fallback."""
    if os.path.exists(config_path):
        return config_path
    default_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
    candidate = os.path.join(default_folder, config_path)
    if os.path.exists(candidate):
        return candidate
    raise ValueError(f"Cannot find config file: {config_path}")


def _safe_name(name: str, fallback: str) -> str:
    """Sanitize arbitrary names for filesystem usage."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or fallback


def _task_namespace(task_path: str) -> str:
    """Extract the top-level task namespace from a benchmark task path."""
    normalized = str(task_path).replace("\\", "/").lstrip("/")
    if normalized.startswith("mcpuniverse/"):
        normalized = normalized[len("mcpuniverse/"):]
    parts = [part for part in normalized.split("/") if part]
    return parts[0] if parts else ""


def _config_category(config_path: str, docs: Optional[List[dict]] = None) -> str:
    """Infer the logical benchmark category for a config file."""
    if docs is None:
        docs = _parse_config_documents(config_path)

    namespaces = set()
    for doc in docs:
        if doc.get("kind", "").lower() != "benchmark":
            continue
        tasks = doc.get("spec", {}).get("tasks", [])
        for task_path in tasks:
            namespace = _task_namespace(task_path)
            if namespace and namespace != "multi_server":
                namespaces.add(namespace)

    if len(namespaces) == 1:
        return namespaces.pop()

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return _safe_name(config_name, "uncategorized")


def _task_output_relpath(task_path: str, category: Optional[str] = None) -> str:
    """Convert a benchmark task path to a stable output-relative file path."""
    normalized = str(task_path).replace("\\", "/").lstrip("/")
    if normalized.startswith("mcpuniverse/"):
        normalized = normalized[len("mcpuniverse/"):]
    if not normalized:
        normalized = "unknown_task.json"
    if category:
        filename = os.path.basename(normalized) or "unknown_task.json"
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        return os.path.join(_safe_name(category, "uncategorized"), filename)
    if not normalized.endswith(".json"):
        normalized = f"{normalized}.json"
    return normalized


def _task_output_paths(
    output_dir: str,
    task_path: str,
    attempt: int = 1,
    category: Optional[str] = None,
) -> Tuple[str, str]:
    """Build output JSON and trace log paths for a task and attempt."""
    rel_json = _task_output_relpath(task_path, category=category)
    if attempt > 1:
        stem, ext = os.path.splitext(rel_json)
        rel_json = f"{stem}_r{attempt - 1}{ext}"

    output_json = os.path.join(output_dir, rel_json)
    trace_stem, _ = os.path.splitext(rel_json)
    log_path = os.path.join(output_dir, f"{trace_stem}.trace.log")
    return output_json, log_path


def _coerce_bool(value: Any, key: str) -> bool:
    """Parse bool/str bool values from settings YAML."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"`{key}` must be bool or bool-like string, got: {value!r}")


def _load_settings_payload(settings_path: str) -> Union[Dict[str, Any], List[Any]]:
    """Load the raw YAML payload for a settings file."""
    with open(settings_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_settings_path(path_value: str, settings_path: str) -> str:
    """Resolve a settings-relative file path to an absolute path."""
    if os.path.isabs(path_value):
        return path_value
    settings_dir = os.path.dirname(os.path.abspath(settings_path))
    return os.path.normpath(os.path.join(settings_dir, path_value))


def _load_run_settings(settings_path: str) -> List[RunSetting]:
    """Load sequential run settings from YAML."""
    payload = _load_settings_payload(settings_path)

    if isinstance(payload, list):
        raw_settings = payload
    elif isinstance(payload, dict):
        raw_settings = payload.get("settings", payload.get("runs", payload.get("experiments", [])))
    else:
        raise ValueError("Settings file must be a YAML list or object containing `settings`")

    if not isinstance(raw_settings, list) or not raw_settings:
        raise ValueError("Settings file must define at least one setting")

    settings: List[RunSetting] = []
    for idx, item in enumerate(raw_settings):
        if not isinstance(item, dict):
            raise ValueError(f"settings[{idx}] must be an object")
        model_name = item.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"settings[{idx}].model_name is required and must be a non-empty string")

        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"run_{idx:02d}_{model_name}"

        concurrency = item.get("concurrency")
        if concurrency is not None:
            try:
                concurrency = int(concurrency)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].concurrency must be an integer") from exc
            if concurrency < 1:
                raise ValueError(f"settings[{idx}].concurrency must be >= 1")

        llm_type = item.get("type", item.get("llm_type"))
        if llm_type is not None:
            if not isinstance(llm_type, str) or not llm_type.strip():
                raise ValueError(f"settings[{idx}].type must be a non-empty string")
            llm_type = llm_type.strip()

        agent_type = item.get("agent_type")
        if agent_type is not None:
            if not isinstance(agent_type, str) or not agent_type.strip():
                raise ValueError(f"settings[{idx}].agent_type must be a non-empty string")
            agent_type = agent_type.strip()

        base_url = item.get("base_url", None)
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError(f"settings[{idx}].base_url must be a string")

        temperature = item.get("temperature")
        if temperature is not None:
            try:
                temperature = float(temperature)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].temperature must be a number") from exc

        top_p = item.get("top_p")
        if top_p is not None:
            try:
                top_p = float(top_p)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].top_p must be a number") from exc

        top_k = item.get("top_k")
        if top_k is not None:
            try:
                top_k = int(top_k)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].top_k must be an integer") from exc

        min_p = item.get("min_p")
        if min_p is not None:
            try:
                min_p = float(min_p)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].min_p must be a number") from exc

        presence_penalty = item.get("presence_penalty")
        if presence_penalty is not None:
            try:
                presence_penalty = float(presence_penalty)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].presence_penalty must be a number") from exc

        repetition_penalty = item.get("repetition_penalty")
        if repetition_penalty is not None:
            try:
                repetition_penalty = float(repetition_penalty)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"settings[{idx}].repetition_penalty must be a number") from exc

        use_custom_tools = item.get("use_custom_tools")
        if use_custom_tools is not None:
            use_custom_tools = _coerce_bool(use_custom_tools, f"settings[{idx}].use_custom_tools")

        github_tokens = item.get(
            "github_tokens",
            item.get("github_token_file", item.get("github_token_path")),
        )
        if github_tokens is not None:
            if not isinstance(github_tokens, str) or not github_tokens.strip():
                raise ValueError(
                    f"settings[{idx}].github_tokens must be a non-empty string"
                )
            github_tokens = _resolve_settings_path(github_tokens.strip(), settings_path)

        cleanup_github_repos_after_run = item.get("cleanup_github_repos_after_run")
        if cleanup_github_repos_after_run is not None:
            cleanup_github_repos_after_run = _coerce_bool(
                cleanup_github_repos_after_run,
                f"settings[{idx}].cleanup_github_repos_after_run",
            )

        settings.append(RunSetting(
            name=name.strip(),
            model_name=model_name.strip(),
            llm_type=llm_type,
            agent_type=agent_type,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            concurrency=concurrency,
            use_custom_tools=use_custom_tools,
            github_tokens=github_tokens,
            cleanup_github_repos_after_run=cleanup_github_repos_after_run,
        ))

    return settings


def _write_overridden_config(
    source_config_path: str,
    output_config_path: str,
    setting: RunSetting,
) -> None:
    """Write a temp config with model/type/base_url/sampling/use_custom_tools overrides."""
    source_path = _resolve_config_path(source_config_path)
    docs = _parse_config_documents(source_path)

    llm_updates = 0
    agent_updates = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        kind = str(doc.get("kind", "")).lower()
        spec = doc.setdefault("spec", {})
        if not isinstance(spec, dict):
            continue
        config = spec.setdefault("config", {})
        if not isinstance(config, dict):
            continue

        if kind == "llm":
            if setting.llm_type is not None:
                spec["type"] = setting.llm_type
            config["model_name"] = setting.model_name
            if setting.base_url is not None:
                config["base_url"] = setting.base_url
            if setting.temperature is not None:
                config["temperature"] = setting.temperature
            if setting.top_p is not None:
                config["top_p"] = setting.top_p
            if setting.top_k is not None:
                config["top_k"] = setting.top_k
            if setting.min_p is not None:
                config["min_p"] = setting.min_p
            if setting.presence_penalty is not None:
                config["presence_penalty"] = setting.presence_penalty
            if setting.repetition_penalty is not None:
                config["repetition_penalty"] = setting.repetition_penalty
            llm_updates += 1

        if kind == "agent":
            if setting.agent_type is not None:
                spec["type"] = setting.agent_type
            if setting.use_custom_tools is not None:
                config["use_custom_tools"] = setting.use_custom_tools
            if setting.agent_type is not None or setting.use_custom_tools is not None:
                agent_updates += 1

    if llm_updates == 0:
        raise ValueError(
            f"No `kind: llm` document found in config {source_config_path}; cannot apply model_name override"
        )
    has_agent_overrides = setting.agent_type is not None or setting.use_custom_tools is not None
    if has_agent_overrides and agent_updates == 0:
        print(
            f"[warning] No `kind: agent` document found in {source_config_path}; "
            "agent_type/use_custom_tools override skipped."
        )

    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        yaml.dump_all(docs, f, default_flow_style=False, allow_unicode=True)


async def _run_settings_suite(
    settings_file: str,
    config_paths: List[str],
    default_concurrency: int,
    output_dir: Optional[str],
    max_retries: int,
    github_tokens: Optional[str],
    min_concurrency: int,
    memory_threshold: float,
) -> None:
    """Run multiple settings sequentially; each setting executes in parallel per task."""
    settings = _load_run_settings(settings_file)

    suite_base = output_dir or os.path.join(
        "results",
        f"settings_{os.path.splitext(os.path.basename(settings_file))[0]}",
    )
    suite_dir = f"{suite_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(suite_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="mcpu_settings_")
    summary_runs: List[Dict[str, Any]] = []
    try:
        for idx, setting in enumerate(settings):
            setting_name = _safe_name(setting.name, f"setting_{idx:02d}")
            setting_tmp_dir = os.path.join(temp_dir, f"{idx:02d}_{setting_name}")
            os.makedirs(setting_tmp_dir, exist_ok=True)

            materialized_configs: List[str] = []
            for cfg_idx, cfg_path in enumerate(config_paths):
                output_cfg = os.path.join(setting_tmp_dir, f"config_{cfg_idx:02d}.yaml")
                _write_overridden_config(
                    source_config_path=cfg_path,
                    output_config_path=output_cfg,
                    setting=setting,
                )
                materialized_configs.append(output_cfg)

            run_concurrency = setting.concurrency or default_concurrency
            run_github_tokens = github_tokens or setting.github_tokens
            run_cleanup_github_repos = setting.cleanup_github_repos_after_run
            if run_cleanup_github_repos is None:
                run_cleanup_github_repos = True
            run_output_dir = os.path.join(suite_dir, f"{idx:02d}_{setting_name}")
            print(
                f"\n=== Setting {idx + 1}/{len(settings)}: {setting.name} "
                f"(type={setting.llm_type}, model={setting.model_name}, concurrency={run_concurrency}, "
                f"use_custom_tools={setting.use_custom_tools}, github_tokens={run_github_tokens}, "
                f"cleanup_github_repos_after_run={run_cleanup_github_repos}) ==="
            )

            runner = ParallelBenchmarkRunner(
                config=materialized_configs,
                concurrency=run_concurrency,
                output_dir=run_output_dir,
                max_retries=max_retries,
                github_tokens=run_github_tokens,
                min_concurrency=min_concurrency,
                memory_threshold=memory_threshold,
            )
            try:
                await runner.run()
            finally:
                if run_cleanup_github_repos:
                    _cleanup_github_repos(run_github_tokens)
                else:
                    print("[cleanup] Skipped: cleanup_github_repos_after_run=false in benchmark config")

            merged_path = os.path.join(runner._output_dir, "merged_results.json")
            merged_summary: Dict[str, Any] = {}
            if os.path.isfile(merged_path):
                with open(merged_path, "r", encoding="utf-8") as f:
                    merged_summary = (json.load(f) or {}).get("summary", {})

            summary_runs.append({
                "name": setting.name,
                "type": setting.llm_type,
                "model_name": setting.model_name,
                "base_url": setting.base_url,
                "use_custom_tools": setting.use_custom_tools,
                "concurrency": run_concurrency,
                "github_tokens": run_github_tokens,
                "cleanup_github_repos_after_run": run_cleanup_github_repos,
                "output_dir": runner._output_dir,
                "merged_results": merged_path,
                "summary": merged_summary,
            })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    suite_summary = {
        "settings_file": settings_file,
        "base_configs": config_paths,
        "generated_at": datetime.now().isoformat(),
        "runs": summary_runs,
    }
    suite_summary_path = os.path.join(suite_dir, "suite_summary.json")
    with open(suite_summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSuite summary written to {suite_summary_path}")


def _split_into_single_task_yamls(
    config_path: str,
    temp_dir: str,
) -> List[TaskItem]:
    """Split a multi-task benchmark YAML into per-task YAML files.

    Returns a list of task items carrying their logical output category.
    """
    docs = _parse_config_documents(config_path)
    category = _config_category(config_path, docs=docs)

    non_benchmark_docs: List[dict] = []
    benchmark_docs: List[dict] = []
    for doc in docs:
        if doc.get("kind", "").lower() == "benchmark":
            benchmark_docs.append(doc)
        else:
            non_benchmark_docs.append(doc)

    os.makedirs(temp_dir, exist_ok=True)
    result: List[TaskItem] = []
    task_index = 0

    for bench_doc in benchmark_docs:
        tasks = bench_doc.get("spec", {}).get("tasks", [])
        for task_path in tasks:
            single = copy.deepcopy(bench_doc)
            single["spec"]["tasks"] = [task_path]
            temp_yaml = os.path.join(temp_dir, f"task_{task_index:04d}.yaml")
            with open(temp_yaml, "w", encoding="utf-8") as f:
                yaml.dump_all(
                    non_benchmark_docs + [single],
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            result.append(TaskItem(
                task_path=task_path,
                temp_yaml=temp_yaml,
                category=category,
                source_config=config_path,
            ))
            task_index += 1

    return result


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

async def _run_worker(config_path: str, output_path: str, log_file: str) -> None:
    """Execute a single-task benchmark and write results to *output_path*."""
    start_time = datetime.now().isoformat()
    try:
        collector = FileCollector(log_file=log_file)
        runner = BenchmarkRunner(config_path)
        results: List[BenchmarkResult] = await runner.run(
            trace_collector=collector,
            callbacks=get_vprint_callbacks(),
        )
        end_time = datetime.now().isoformat()

        # Collect evaluation results from all benchmarks (typically 1)
        all_eval: List[Dict] = []
        trace_id = ""
        for br in results:
            for task_path, task_data in br.task_results.items():
                eval_results = task_data.get("evaluation_results", [])
                all_eval.extend(
                    r.model_dump(mode="json") if isinstance(r, EvaluationResult) else r
                    for r in eval_results
                )
            for tp, tid in br.task_trace_ids.items():
                trace_id = tid  # last one wins (single-task config)

        payload = {
            "task_path": list(results[0].task_results.keys())[0] if results else "",
            "evaluation_results": all_eval,
            "trace_id": trace_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    except Exception as exc:
        end_time = datetime.now().isoformat()
        payload = _build_worker_error_payload(
            config_path=config_path,
            output_path=output_path,
            log_file=log_file,
            start_time=start_time,
            end_time=end_time,
            exc=exc,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Parallel orchestrator
# ---------------------------------------------------------------------------

class ParallelBenchmarkRunner:
    """Run benchmark tasks concurrently using subprocess workers.

    *config* accepts a single YAML path (str) or a list of paths.  When
    multiple configs are provided their tasks are pooled into one shared
    concurrency pool while each task keeps its own llm/agent definition.
    """

    def __init__(
        self,
        config: Union[str, List[str]],
        concurrency: int = 10,
        output_dir: Optional[str] = None,
        max_retries: int = 2,
        github_tokens: Optional[str] = None,
        min_concurrency: int = 1,
        memory_threshold: float = 0.6,
    ):
        if isinstance(config, str):
            self._configs: List[str] = [config]
        else:
            self._configs = list(config)
        self._concurrency = concurrency
        self._output_dir = output_dir
        self._max_retries = max_retries
        self._feedback = FeedbackController(
            initial_concurrency=concurrency,
            min_concurrency=min_concurrency,
            memory_threshold=memory_threshold,
        )
        self._github_accounts: Optional[List[Tuple[str, str]]] = None
        if github_tokens:
            self._github_accounts = _load_github_tokens(github_tokens)

    async def run(self) -> List[BenchmarkResult]:
        """Split config(s), run workers concurrently, merge and return results."""
        if not self._output_dir:
            if len(self._configs) == 1:
                config_name = os.path.splitext(os.path.basename(self._configs[0]))[0]
            else:
                config_name = "multi_benchmark"
            self._output_dir = os.path.join("results", config_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = f"{self._output_dir}_{timestamp}"

        temp_dir = tempfile.mkdtemp(prefix="mcpu_parallel_")
        try:
            task_items: List[TaskItem] = []
            for cfg_idx, cfg_path in enumerate(self._configs):
                sub_dir = os.path.join(temp_dir, f"config_{cfg_idx}")
                task_items.extend(_split_into_single_task_yamls(cfg_path, sub_dir))
            if not task_items:
                print("No tasks found in config(s).")
                return []
            random.shuffle(task_items)

            # Round-robin assign GitHub accounts to tasks
            self._task_account_map: Dict[str, Tuple[str, str]] = {}
            if self._github_accounts:
                for idx, task_item in enumerate(task_items):
                    account = self._github_accounts[idx % len(self._github_accounts)]
                    self._task_account_map[task_item.temp_yaml] = account
                    print(f"  [github] {task_item.task_path} -> {account[0]}")

            os.makedirs(self._output_dir, exist_ok=True)
            semaphore = self._feedback.semaphore

            # First pass
            coros = []
            for task_item in task_items:
                out_json, log_path = _task_output_paths(
                    self._output_dir,
                    task_item.task_path,
                    attempt=1,
                    category=task_item.category,
                )
                coros.append(
                    self._launch_worker(task_item, out_json, log_path, semaphore, attempt=1)
                )

            worker_results: List[WorkerResult] = await asyncio.gather(*coros)

            # Retry loop
            for retry_round in range(1, self._max_retries + 1):
                failed = [
                    wr for wr in worker_results
                    if not wr.success and wr.error_category != ErrorCategory.AUTH
                ]
                if not failed:
                    break
                print(f"\n--- Retry round {retry_round}: {len(failed)} failed tasks ---")

                # Rebuild temp yamls for failed tasks
                retry_coros = []
                for wr in failed:
                    if not wr.temp_yaml:
                        continue
                    out_json, log_path = _task_output_paths(
                        self._output_dir,
                        wr.task_path,
                        attempt=retry_round + 1,
                        category=wr.category,
                    )
                    retry_coros.append(
                        self._launch_worker(
                            TaskItem(
                                task_path=wr.task_path,
                                temp_yaml=wr.temp_yaml,
                                category=wr.category or "uncategorized",
                                source_config="",
                            ),
                            out_json,
                            log_path,
                            semaphore,
                            attempt=retry_round + 1,
                        )
                    )

                retry_results = await asyncio.gather(*retry_coros)

                # Replace failed results with retry results
                retry_map = {rr.task_path: rr for rr in retry_results}
                worker_results = [
                    retry_map.get(wr.task_path, wr)
                    if not wr.success and wr.error_category != ErrorCategory.AUTH
                    else wr
                    for wr in worker_results
                ]

            merged = self._merge_results(self._configs, worker_results)
            self._print_summary(worker_results)
            self._print_evaluation_summary(merged)

            # Write merged results with pass-rate summary
            merged_path = os.path.join(self._output_dir, "merged_results.json")
            summary = self._build_summary(merged, worker_results)
            merged_payload = {
                "summary": summary,
                "benchmarks": [br.model_dump(mode="json") for br in merged],
            }
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump(merged_payload, f, indent=2, default=str)
            print(f"\nMerged results written to {merged_path}")

            return merged
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _launch_worker(
        self,
        task_item: TaskItem,
        output_json: str,
        log_path: str,
        semaphore: AdaptiveSemaphore,
        attempt: int = 1,
    ) -> WorkerResult:
        """Launch a single worker subprocess under semaphore control."""
        delay = await self._feedback.wait_before_launch()
        if delay > 0:
            print(f"  [feedback] Delayed {delay:.1f}s before launching: {task_item.task_path}")

        async with semaphore:

            print(f"  [{attempt}] Starting: {task_item.task_path}")
            start = time.monotonic()
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            worker_env = None
            if hasattr(self, "_task_account_map") and task_item.temp_yaml in self._task_account_map:
                username, token = self._task_account_map[task_item.temp_yaml]
                worker_env = {**os.environ,
                              "GITHUB_PERSONAL_ACCESS_TOKEN": token,
                              "GITHUB_PERSONAL_ACCOUNT_NAME": username}

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mcpuniverse.benchmark.parallel",
                "--worker", task_item.temp_yaml,
                "--output", output_json,
                "--log-file", log_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=worker_env,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            duration = time.monotonic() - start
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            if proc.returncode == 0 and os.path.isfile(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "error" in data:
                        error_text = "\n".join(
                            part for part in [
                                data.get("error_type", ""),
                                data.get("error_message", ""),
                                data.get("error_traceback", ""),
                                stderr_text,
                            ] if part
                        )
                        error_cat = classify_error(error_text)
                        wr = WorkerResult(
                            task_path=task_item.task_path, success=False,
                            category=task_item.category,
                            temp_yaml=task_item.temp_yaml,
                            error_category=error_cat,
                            stderr=stderr_text,
                            stdout=stdout_text,
                            return_code=proc.returncode, duration_seconds=duration,
                            attempt=attempt,
                            start_time=data.get("start_time", ""),
                            end_time=data.get("end_time", ""),
                            error_message=data.get("error_message", data.get("error", "")),
                            error_type=data.get("error_type", ""),
                            error_traceback=data.get("error_traceback", ""),
                            output_json=output_json,
                            log_path=data.get("log_file", log_path),
                        )
                    else:
                        wr = WorkerResult(
                            task_path=task_item.task_path, success=True,
                            evaluation_results=data.get("evaluation_results", []),
                            category=task_item.category,
                            temp_yaml=task_item.temp_yaml,
                            trace_id=data.get("trace_id", ""),
                            stdout=stdout_text,
                            return_code=0, duration_seconds=duration,
                            attempt=attempt,
                            start_time=data.get("start_time", ""),
                            end_time=data.get("end_time", ""),
                            output_json=output_json,
                            log_path=log_path,
                        )
                except (json.JSONDecodeError, KeyError) as exc:
                    wr = WorkerResult(
                        task_path=task_item.task_path, success=False,
                        category=task_item.category,
                        temp_yaml=task_item.temp_yaml,
                        error_category=ErrorCategory.UNKNOWN,
                        stderr=f"JSON parse error: {exc}\n{stderr_text}",
                        stdout=stdout_text,
                        return_code=proc.returncode, duration_seconds=duration,
                        attempt=attempt,
                        error_message=f"Failed to parse worker output JSON: {exc}",
                        error_type=type(exc).__name__,
                        error_traceback=traceback.format_exc(),
                        output_json=output_json,
                        log_path=log_path,
                    )
            else:
                error_text = "\n".join(
                    part for part in [stderr_text, stdout_text] if part
                )
                error_cat = classify_error(error_text)
                wr = WorkerResult(
                    task_path=task_item.task_path, success=False,
                    category=task_item.category,
                    temp_yaml=task_item.temp_yaml,
                    error_category=error_cat,
                    stderr=stderr_text,
                    stdout=stdout_text,
                    return_code=proc.returncode or 1, duration_seconds=duration,
                    attempt=attempt,
                    error_message=(
                        f"Worker exited with code {proc.returncode or 1} without a valid result file"
                    ),
                    error_type="WorkerProcessError",
                    output_json=output_json,
                    log_path=log_path,
                )

            status = "\033[32mOK\033[0m" if wr.success else f"\033[31mFAIL ({wr.error_category.name})\033[0m"
            print(f"  [{attempt}] Finished: {task_item.task_path} — {status} ({duration:.1f}s)")
            if not wr.success:
                detail = wr.error_message or _truncate_text(wr.stderr, limit=240)
                if detail:
                    print(f"      error: {_truncate_text(detail, limit=240)}")
                print(f"      output: {output_json}")
                print(f"      trace: {log_path}")

            await self._feedback.record(wr)
            return wr

    @staticmethod
    def _merge_results(
        config_paths: List[str],
        worker_results: List[WorkerResult],
    ) -> List[BenchmarkResult]:
        """Merge worker results into BenchmarkResult objects compatible with the serial runner."""
        # Index worker results by task_path
        wr_map: Dict[str, WorkerResult] = {}
        for wr in worker_results:
            wr_map[wr.task_path] = wr

        merged: List[BenchmarkResult] = []
        for config_path in config_paths:
            docs = _parse_config_documents(config_path)
            benchmark_docs = [
                d for d in docs if d.get("kind", "").lower() == "benchmark"
            ]

            for bench_doc in benchmark_docs:
                bench_config = BenchmarkConfig.model_validate(bench_doc["spec"])
                task_results: Dict[str, Dict] = {}
                task_trace_ids: Dict[str, str] = {}

                for task_path in bench_config.tasks:
                    wr = wr_map.get(task_path)
                    if wr and wr.success and wr.evaluation_results is not None:
                        eval_objs = [
                            EvaluationResult.model_validate(e) for e in wr.evaluation_results
                        ]
                        task_results[task_path] = {"evaluation_results": eval_objs}
                        task_trace_ids[task_path] = wr.trace_id
                    else:
                        task_results[task_path] = {
                            "evaluation_results": [],
                            "worker_error": _worker_failure_details(wr) if wr else None,
                        }
                        task_trace_ids[task_path] = ""

                merged.append(BenchmarkResult(
                    benchmark=bench_config,
                    task_results=task_results,
                    task_trace_ids=task_trace_ids,
                ))

        return merged

    @staticmethod
    def _print_summary(worker_results: List[WorkerResult]) -> None:
        """Print a coloured pass/fail summary table."""
        total = len(worker_results)
        passed = sum(1 for r in worker_results if r.success)
        failed = total - passed

        print("\n" + "=" * 70)
        print("PARALLEL BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Task':<60} {'Status':<10}")
        print("-" * 70)

        for wr in worker_results:
            if wr.success:
                status = "\033[32mPASS\033[0m"
            else:
                status = f"\033[31mFAIL\033[0m ({wr.error_category.name})"
            # Truncate long paths
            display_path = wr.task_path if len(wr.task_path) <= 58 else "..." + wr.task_path[-55:]
            print(f"{display_path:<60} {status}")

        print("-" * 70)
        print(
            f"Total: {total}  |  "
            f"\033[32mPassed: {passed}\033[0m  |  "
            f"\033[31mFailed: {failed}\033[0m"
        )
        print("=" * 70)

    @staticmethod
    def _print_evaluation_summary(merged: List[BenchmarkResult]) -> None:
        """Print an evaluation-level pass/fail summary across all benchmarks."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"{'Task':<50} {'Evals':>6} {'Pass':>6} {'Fail':>6} {'Score':>7}")
        print("-" * 70)

        total_evals = 0
        total_passed = 0
        total_failed = 0

        for br in merged:
            for task_path, task_data in br.task_results.items():
                eval_results = task_data.get("evaluation_results", [])
                if not eval_results:
                    display = task_path if len(task_path) <= 48 else "..." + task_path[-45:]
                    print(f"{display:<50} {'—':>6} {'—':>6} {'—':>6} {'N/A':>7}")
                    continue

                n_passed = sum(
                    1 for e in eval_results
                    if (e.passed if isinstance(e, EvaluationResult) else e.get("passed", False))
                )
                n_total = len(eval_results)
                n_failed = n_total - n_passed
                score = n_passed / n_total if n_total > 0 else 0.0

                total_evals += n_total
                total_passed += n_passed
                total_failed += n_failed

                display = task_path if len(task_path) <= 48 else "..." + task_path[-45:]
                score_color = "\033[32m" if score >= 1.0 else "\033[33m" if score > 0 else "\033[31m"
                print(
                    f"{display:<50} {n_total:>6} "
                    f"\033[32m{n_passed:>6}\033[0m "
                    f"\033[31m{n_failed:>6}\033[0m "
                    f"{score_color}{score:>6.0%}\033[0m"
                )

        print("-" * 70)
        overall = total_passed / total_evals if total_evals > 0 else 0.0
        print(
            f"{'TOTAL':<50} {total_evals:>6} "
            f"\033[32m{total_passed:>6}\033[0m "
            f"\033[31m{total_failed:>6}\033[0m "
            f"{overall:>6.0%}"
        )
        print("=" * 70)

    @staticmethod
    def _build_summary(
        merged: List[BenchmarkResult],
        worker_results: List[WorkerResult],
    ) -> Dict:
        """Build a summary dict with per-task, per-category and overall pass rates."""
        task_summaries: List[Dict] = []
        category_summaries: Dict[str, Dict[str, Union[int, float, str]]] = {}
        total_evals = 0
        total_evals_passed = 0
        total_tasks = 0
        total_tasks_passed = 0
        worker_result_map = {wr.task_path: wr for wr in worker_results}
        task_categories = {
            wr.task_path: (wr.category or "uncategorized")
            for wr in worker_results
        }

        for br in merged:
            for task_path, task_data in br.task_results.items():
                eval_results = task_data.get("evaluation_results", [])
                total_tasks += 1
                category = task_categories.get(task_path, "uncategorized")
                category_summary = category_summaries.setdefault(
                    category,
                    {
                        "category": category,
                        "eval_total": 0,
                        "eval_passed": 0,
                        "eval_failed": 0,
                        "eval_pass_rate": 0.0,
                        "task_total": 0,
                        "task_passed": 0,
                        "task_failed": 0,
                        "task_pass_rate": 0.0,
                    },
                )
                category_summary["task_total"] += 1

                if not eval_results:
                    task_summary = {
                        "category": category,
                        "task": task_path,
                        "evals_total": 0,
                        "evals_passed": 0,
                        "evals_failed": 0,
                        "task_passed": False,
                    }
                    wr = worker_result_map.get(task_path)
                    if wr and not wr.success:
                        task_summary["worker_error"] = _worker_failure_details(wr)
                    task_summaries.append(task_summary)
                    continue

                n_passed = sum(
                    1 for e in eval_results
                    if (e.passed if isinstance(e, EvaluationResult) else e.get("passed", False))
                )
                n_total = len(eval_results)
                n_failed = n_total - n_passed
                task_passed = n_failed == 0

                total_evals += n_total
                total_evals_passed += n_passed
                if task_passed:
                    total_tasks_passed += 1
                    category_summary["task_passed"] += 1

                task_summaries.append({
                    "category": category,
                    "task": task_path,
                    "evals_total": n_total,
                    "evals_passed": n_passed,
                    "evals_failed": n_failed,
                    "task_passed": task_passed,
                })
                category_summary["eval_total"] += n_total
                category_summary["eval_passed"] += n_passed

        for category_summary in category_summaries.values():
            category_summary["eval_failed"] = (
                category_summary["eval_total"] - category_summary["eval_passed"]
            )
            category_summary["task_failed"] = (
                category_summary["task_total"] - category_summary["task_passed"]
            )
            category_summary["eval_pass_rate"] = round(
                (
                    category_summary["eval_passed"] / category_summary["eval_total"]
                    if category_summary["eval_total"] > 0
                    else 0.0
                ),
                4,
            )
            category_summary["task_pass_rate"] = round(
                (
                    category_summary["task_passed"] / category_summary["task_total"]
                    if category_summary["task_total"] > 0
                    else 0.0
                ),
                4,
            )

        eval_pass_rate = total_evals_passed / total_evals if total_evals > 0 else 0.0
        task_pass_rate = total_tasks_passed / total_tasks if total_tasks > 0 else 0.0

        # Compute total wall-clock duration from worker timestamps
        start_times: List[datetime] = []
        end_times: List[datetime] = []
        for wr in worker_results:
            if wr.start_time:
                try:
                    start_times.append(datetime.fromisoformat(wr.start_time))
                except ValueError:
                    pass
            if wr.end_time:
                try:
                    end_times.append(datetime.fromisoformat(wr.end_time))
                except ValueError:
                    pass

        duration_seconds: Optional[float] = None
        earliest_start: Optional[str] = None
        latest_end: Optional[str] = None
        if start_times and end_times:
            earliest = min(start_times)
            latest = max(end_times)
            duration_seconds = round((latest - earliest).total_seconds(), 2)
            earliest_start = earliest.isoformat()
            latest_end = latest.isoformat()

        return {
            "total_duration_seconds": duration_seconds,
            "earliest_start": earliest_start,
            "latest_end": latest_end,
            "eval_pass_rate": round(eval_pass_rate, 4),
            "eval_total": total_evals,
            "eval_passed": total_evals_passed,
            "eval_failed": total_evals - total_evals_passed,
            "task_pass_rate": round(task_pass_rate, 4),
            "task_total": total_tasks,
            "task_passed": total_tasks_passed,
            "task_failed": total_tasks - total_tasks_passed,
            "categories": {
                name: category_summaries[name]
                for name in sorted(category_summaries)
            },
            "tasks": task_summaries,
            "worker_failures": [
                {
                    "category": wr.category or "uncategorized",
                    "task": wr.task_path,
                    **_worker_failure_details(wr),
                }
                for wr in worker_results
                if not wr.success
            ],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel benchmark runner for MCP-Universe",
    )
    parser.add_argument(
        "--worker",
        metavar="CONFIG",
        help="Run in worker mode: execute a single-task config",
    )
    parser.add_argument("--output", help="(worker) Output JSON path")
    parser.add_argument("--log-file", help="(worker) Trace log file path")

    parser.add_argument(
        "config",
        nargs="*",
        help="(orchestrator) Benchmark config YAML path(s); multiple configs are pooled into one run",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="(orchestrator) Max concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="(orchestrator) Output directory (default: results/<config_name>)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="(orchestrator) Max retry rounds for failed tasks (default: 2)",
    )
    parser.add_argument(
        "--min-concurrency", type=int, default=1,
        help="(orchestrator) Minimum concurrency when AIMD scales down (default: 1)",
    )
    parser.add_argument(
        "--memory-threshold", type=float, default=0.6,
        help="(orchestrator) Memory usage threshold (0-1) to halve concurrency (default: 0.6)",
    )
    parser.add_argument(
        "--github-tokens", default=None,
        help="(orchestrator) CSV file with GitHub accounts (username,token) for round-robin distribution",
    )
    parser.add_argument(
        "--settings-file",
        default=None,
        help=(
            "(orchestrator) YAML file defining sequential settings. "
            "Each setting can override type, model_name, base_url, concurrency, use_custom_tools."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.worker:
        # Worker mode
        if not args.output:
            parser.error("--output is required in worker mode")
        if not args.log_file:
            parser.error("--log-file is required in worker mode")
        asyncio.run(_run_worker(args.worker, args.output, args.log_file))
    elif args.settings_file:
        if not args.config:
            parser.error("Provide at least one config path when using --settings-file")
        asyncio.run(_run_settings_suite(
            settings_file=args.settings_file,
            config_paths=args.config,
            default_concurrency=args.concurrency,
            output_dir=args.output_dir,
            max_retries=args.max_retries,
            github_tokens=args.github_tokens,
            min_concurrency=args.min_concurrency,
            memory_threshold=args.memory_threshold,
        ))
    elif args.config:
        # Orchestrator mode — single config or multiple configs pooled
        runner = ParallelBenchmarkRunner(
            config=args.config,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
            max_retries=args.max_retries,
            github_tokens=args.github_tokens,
            min_concurrency=args.min_concurrency,
            memory_threshold=args.memory_threshold,
        )
        try:
            asyncio.run(runner.run())
        finally:
            _cleanup_github_repos(args.github_tokens)
    else:
        parser.error("Provide config path(s) or use --worker mode")


if __name__ == "__main__":
    main()
