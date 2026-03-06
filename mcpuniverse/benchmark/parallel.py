"""
Parallel benchmark runner for MCP-Universe.

Orchestrates concurrent execution of benchmark tasks by splitting a multi-task
YAML config into single-task configs and running each in a separate subprocess.

Usage:
    # Orchestrator mode — single config
    python -m mcpuniverse.benchmark.parallel config.yaml --concurrency 20

    # Orchestrator mode — multiple configs pooled into one run
    python -m mcpuniverse.benchmark.parallel a.yaml b.yaml c.yaml --concurrency 20

    # Worker mode (called internally by orchestrator)
    python -m mcpuniverse.benchmark.parallel --worker config.yaml --output result.json --log-file trace.log
"""
import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import deque

import psutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

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
    trace_id: str = ""
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    stderr: str = ""
    return_code: int = -1
    duration_seconds: float = 0.0
    attempt: int = 1
    start_time: str = ""
    end_time: str = ""


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
        max_cooldown: float = 300.0,
        initial_concurrency: int = 10,
        min_concurrency: int = 1,
        memory_threshold: float = 0.6,
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


def _split_into_single_task_yamls(
    config_path: str,
    temp_dir: str,
) -> List[Tuple[str, str]]:
    """Split a multi-task benchmark YAML into per-task YAML files.

    Returns a list of ``(task_path, temp_yaml_path)`` tuples.
    """
    docs = _parse_config_documents(config_path)

    non_benchmark_docs: List[dict] = []
    benchmark_docs: List[dict] = []
    for doc in docs:
        if doc.get("kind", "").lower() == "benchmark":
            benchmark_docs.append(doc)
        else:
            non_benchmark_docs.append(doc)

    os.makedirs(temp_dir, exist_ok=True)
    result: List[Tuple[str, str]] = []
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
            result.append((task_path, temp_yaml))
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
        payload = {
            "error": str(exc),
            "start_time": start_time,
            "end_time": end_time,
        }
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
            task_items: List[Tuple[str, str]] = []
            for cfg_idx, cfg_path in enumerate(self._configs):
                sub_dir = os.path.join(temp_dir, f"config_{cfg_idx}")
                task_items.extend(_split_into_single_task_yamls(cfg_path, sub_dir))
            if not task_items:
                print("No tasks found in config(s).")
                return []

            # Round-robin assign GitHub accounts to tasks
            self._task_account_map: Dict[str, Tuple[str, str]] = {}
            if self._github_accounts:
                for idx, (task_path, _) in enumerate(task_items):
                    account = self._github_accounts[idx % len(self._github_accounts)]
                    self._task_account_map[task_path] = account
                    print(f"  [github] {task_path} -> {account[0]}")

            os.makedirs(self._output_dir, exist_ok=True)
            semaphore = self._feedback.semaphore

            # First pass
            coros = []
            for idx, (task_path, temp_yaml) in enumerate(task_items):
                out_json = os.path.join(self._output_dir, f"result_{idx:04d}.json")
                log_path = os.path.join(self._output_dir, f"trace_{idx:04d}.log")
                coros.append(
                    self._launch_worker(task_path, temp_yaml, out_json, log_path, semaphore, attempt=1)
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
                    # Find the original temp yaml for this task
                    matching = [
                        (tp, ty) for tp, ty in task_items if tp == wr.task_path
                    ]
                    if not matching:
                        continue
                    task_path, temp_yaml = matching[0]
                    idx_str = os.path.basename(temp_yaml).replace("task_", "").replace(".yaml", "")
                    out_json = os.path.join(
                        self._output_dir, f"result_{idx_str}_r{retry_round}.json"
                    )
                    log_path = os.path.join(
                        self._output_dir, f"trace_{idx_str}_r{retry_round}.log"
                    )
                    retry_coros.append(
                        self._launch_worker(
                            task_path, temp_yaml, out_json, log_path, semaphore,
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
        task_path: str,
        temp_yaml: str,
        output_json: str,
        log_path: str,
        semaphore: AdaptiveSemaphore,
        attempt: int = 1,
    ) -> WorkerResult:
        """Launch a single worker subprocess under semaphore control."""
        async with semaphore:
            delay = await self._feedback.wait_before_launch()
            if delay > 0:
                print(f"  [feedback] Delayed {delay:.1f}s before launching: {task_path}")

            print(f"  [{attempt}] Starting: {task_path}")
            start = time.monotonic()

            worker_env = None
            if hasattr(self, "_task_account_map") and task_path in self._task_account_map:
                username, token = self._task_account_map[task_path]
                worker_env = {**os.environ,
                              "GITHUB_PERSONAL_ACCESS_TOKEN": token,
                              "GITHUB_PERSONAL_ACCOUNT_NAME": username}

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "mcpuniverse.benchmark.parallel",
                "--worker", temp_yaml,
                "--output", output_json,
                "--log-file", log_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=worker_env,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            duration = time.monotonic() - start
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            if proc.returncode == 0 and os.path.isfile(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "error" in data:
                        error_cat = classify_error(data["error"] + stderr_text)
                        wr = WorkerResult(
                            task_path=task_path, success=False,
                            error_category=error_cat, stderr=stderr_text,
                            return_code=proc.returncode, duration_seconds=duration,
                            attempt=attempt,
                            start_time=data.get("start_time", ""),
                            end_time=data.get("end_time", ""),
                        )
                    else:
                        wr = WorkerResult(
                            task_path=task_path, success=True,
                            evaluation_results=data.get("evaluation_results", []),
                            trace_id=data.get("trace_id", ""),
                            return_code=0, duration_seconds=duration,
                            attempt=attempt,
                            start_time=data.get("start_time", ""),
                            end_time=data.get("end_time", ""),
                        )
                except (json.JSONDecodeError, KeyError) as exc:
                    wr = WorkerResult(
                        task_path=task_path, success=False,
                        error_category=ErrorCategory.UNKNOWN,
                        stderr=f"JSON parse error: {exc}\n{stderr_text}",
                        return_code=proc.returncode, duration_seconds=duration,
                        attempt=attempt,
                    )
            else:
                error_cat = classify_error(stderr_text)
                wr = WorkerResult(
                    task_path=task_path, success=False,
                    error_category=error_cat, stderr=stderr_text,
                    return_code=proc.returncode or 1, duration_seconds=duration,
                    attempt=attempt,
                )

            status = "\033[32mOK\033[0m" if wr.success else f"\033[31mFAIL ({wr.error_category.name})\033[0m"
            print(f"  [{attempt}] Finished: {task_path} — {status} ({duration:.1f}s)")

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
                        task_results[task_path] = {"evaluation_results": []}
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
        """Build a summary dict with per-task and overall pass rates."""
        task_summaries: List[Dict] = []
        total_evals = 0
        total_evals_passed = 0
        total_tasks = 0
        total_tasks_passed = 0

        for br in merged:
            for task_path, task_data in br.task_results.items():
                eval_results = task_data.get("evaluation_results", [])
                total_tasks += 1

                if not eval_results:
                    task_summaries.append({
                        "task": task_path,
                        "evals_total": 0,
                        "evals_passed": 0,
                        "evals_failed": 0,
                        "task_passed": False,
                    })
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

                task_summaries.append({
                    "task": task_path,
                    "evals_total": n_total,
                    "evals_passed": n_passed,
                    "evals_failed": n_failed,
                    "task_passed": task_passed,
                })

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
            "tasks": task_summaries,
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
        asyncio.run(runner.run())
    else:
        parser.error("Provide config path(s) or use --worker mode")


if __name__ == "__main__":
    main()
