"""
Tests for the parallel benchmark runner.
"""
import asyncio
import os
import tempfile
import unittest

import pytest
import yaml

from mcpuniverse.benchmark.parallel import (
    AdaptiveSemaphore,
    ErrorCategory,
    FeedbackController,
    WorkerResult,
    _parse_config_documents,
    _split_into_single_task_yamls,
    classify_error,
)


class TestErrorClassifier(unittest.TestCase):
    """Unit tests for classify_error()."""

    def test_rate_limit_429(self):
        self.assertEqual(
            classify_error("HTTP 429 Too Many Requests"),
            ErrorCategory.RATE_LIMIT,
        )

    def test_rate_limit_error_class(self):
        self.assertEqual(
            classify_error("RateLimitError: exceeded quota"),
            ErrorCategory.RATE_LIMIT,
        )

    def test_rate_limit_phrase(self):
        self.assertEqual(
            classify_error("rate limit exceeded for model"),
            ErrorCategory.RATE_LIMIT,
        )

    def test_rate_limit_too_many_requests(self):
        self.assertEqual(
            classify_error("too many requests, please retry"),
            ErrorCategory.RATE_LIMIT,
        )

    def test_rate_limit_quota_exceeded(self):
        self.assertEqual(
            classify_error("API quota exceeded"),
            ErrorCategory.RATE_LIMIT,
        )

    def test_timeout(self):
        self.assertEqual(
            classify_error("TimeoutError: operation timed out"),
            ErrorCategory.TIMEOUT,
        )

    def test_timed_out(self):
        self.assertEqual(
            classify_error("request timed out after 30s"),
            ErrorCategory.TIMEOUT,
        )

    def test_connection_error(self):
        self.assertEqual(
            classify_error("ConnectionError: failed to connect"),
            ErrorCategory.CONNECTION,
        )

    def test_connection_refused(self):
        self.assertEqual(
            classify_error("connection refused on port 8080"),
            ErrorCategory.CONNECTION,
        )

    def test_connection_reset(self):
        self.assertEqual(
            classify_error("connection reset by peer"),
            ErrorCategory.CONNECTION,
        )

    def test_econnrefused(self):
        self.assertEqual(
            classify_error("Error: ECONNREFUSED 127.0.0.1:3000"),
            ErrorCategory.CONNECTION,
        )

    def test_auth_401(self):
        self.assertEqual(
            classify_error("HTTP 401 Unauthorized"),
            ErrorCategory.AUTH,
        )

    def test_auth_403(self):
        self.assertEqual(
            classify_error("HTTP 403 Forbidden"),
            ErrorCategory.AUTH,
        )

    def test_unauthorized(self):
        self.assertEqual(
            classify_error("Unauthorized access to resource"),
            ErrorCategory.AUTH,
        )

    def test_invalid_api_key(self):
        self.assertEqual(
            classify_error("invalid api key provided"),
            ErrorCategory.AUTH,
        )

    def test_unknown_error(self):
        self.assertEqual(
            classify_error("some random error happened"),
            ErrorCategory.UNKNOWN,
        )

    def test_empty_string(self):
        self.assertEqual(classify_error(""), ErrorCategory.UNKNOWN)

    def test_priority_rate_limit_over_timeout(self):
        # When both patterns match, the first match (rate limit) wins
        self.assertEqual(
            classify_error("429 timeout"),
            ErrorCategory.RATE_LIMIT,
        )


class TestFeedbackController(unittest.IsolatedAsyncioTestCase):
    """Unit tests for FeedbackController cooldown logic."""

    @staticmethod
    def _make_result(success: bool, category: ErrorCategory = ErrorCategory.UNKNOWN) -> WorkerResult:
        return WorkerResult(
            task_path="test/task.json",
            success=success,
            error_category=category if not success else ErrorCategory.UNKNOWN,
        )

    def test_empty_window_zero_cooldown(self):
        fc = FeedbackController(window_size=10)
        self.assertEqual(fc._compute_cooldown(), 0.0)

    async def test_all_success_zero_cooldown(self):
        fc = FeedbackController(window_size=10)
        for _ in range(5):
            await fc.record(self._make_result(success=True))
        self.assertEqual(fc._compute_cooldown(), 0.0)

    async def test_rate_limit_increases_cooldown(self):
        fc = FeedbackController(window_size=10)
        await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        # 1 rate_limit * 5.0 = 5.0, no recovery (0% success)
        self.assertAlmostEqual(fc._compute_cooldown(), 5.0)

    async def test_multiple_error_types(self):
        fc = FeedbackController(window_size=10)
        await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        await fc.record(self._make_result(success=False, category=ErrorCategory.TIMEOUT))
        await fc.record(self._make_result(success=False, category=ErrorCategory.CONNECTION))
        # 5 + 2 + 3 = 10.0
        self.assertAlmostEqual(fc._compute_cooldown(), 10.0)

    async def test_max_cooldown_cap(self):
        fc = FeedbackController(window_size=100, max_cooldown=30.0)
        for _ in range(20):
            await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        # 20 * 5.0 = 100.0, capped at 30.0
        self.assertAlmostEqual(fc._compute_cooldown(), 30.0)

    async def test_recovery_halves_cooldown(self):
        fc = FeedbackController(window_size=10)
        # 2 rate limit errors + 8 successes = 80% success
        for _ in range(2):
            await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        for _ in range(8):
            await fc.record(self._make_result(success=True))
        # 2 * 5.0 = 10.0, halved due to 80% success => 5.0
        self.assertAlmostEqual(fc._compute_cooldown(), 5.0)

    async def test_below_recovery_threshold_no_halving(self):
        fc = FeedbackController(window_size=10)
        # 3 rate limit errors + 7 successes = 70% success (below 80%)
        for _ in range(3):
            await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        for _ in range(7):
            await fc.record(self._make_result(success=True))
        # 3 * 5.0 = 15.0, no halving
        self.assertAlmostEqual(fc._compute_cooldown(), 15.0)

    async def test_sliding_window_eviction(self):
        fc = FeedbackController(window_size=5)
        # Fill window with errors
        for _ in range(5):
            await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        self.assertAlmostEqual(fc._compute_cooldown(), 25.0)
        # Push out all errors with successes
        for _ in range(5):
            await fc.record(self._make_result(success=True))
        self.assertAlmostEqual(fc._compute_cooldown(), 0.0)

    async def test_wait_before_launch_no_delay(self):
        fc = FeedbackController(window_size=10)
        delay = await fc.wait_before_launch()
        self.assertEqual(delay, 0.0)


class TestYamlSplitting(unittest.TestCase):
    """Tests for YAML config splitting."""

    def test_split_creates_correct_count(self):
        # Create a temporary multi-task YAML
        docs = [
            {"kind": "llm", "spec": {"name": "llm-1", "type": "openrouter", "config": {"model_name": "test"}}},
            {"kind": "agent", "spec": {"name": "agent-1", "type": "function-call", "config": {"llm": "llm-1"}}},
            {"kind": "benchmark", "spec": {
                "description": "test benchmark",
                "agent": "agent-1",
                "tasks": ["task_a.json", "task_b.json", "task_c.json"],
            }},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump_all(docs, f, default_flow_style=False)
            config_path = f.name

        temp_dir = tempfile.mkdtemp()
        try:
            result = _split_into_single_task_yamls(config_path, temp_dir)
            self.assertEqual(len(result), 3)

            # Verify each split file has exactly one task
            for task_path, yaml_path in result:
                split_docs = _parse_config_documents(yaml_path)
                bench_docs = [d for d in split_docs if d.get("kind", "").lower() == "benchmark"]
                self.assertEqual(len(bench_docs), 1)
                self.assertEqual(len(bench_docs[0]["spec"]["tasks"]), 1)
                self.assertEqual(bench_docs[0]["spec"]["tasks"][0], task_path)

                # Non-benchmark docs preserved
                non_bench = [d for d in split_docs if d.get("kind", "").lower() != "benchmark"]
                self.assertEqual(len(non_bench), 2)
        finally:
            os.unlink(config_path)
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_split_multiple_benchmarks(self):
        docs = [
            {"kind": "llm", "spec": {"name": "llm-1", "type": "test", "config": {}}},
            {"kind": "benchmark", "spec": {
                "description": "bench 1", "agent": "a", "tasks": ["t1.json", "t2.json"],
            }},
            {"kind": "benchmark", "spec": {
                "description": "bench 2", "agent": "b", "tasks": ["t3.json"],
            }},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump_all(docs, f, default_flow_style=False)
            config_path = f.name

        temp_dir = tempfile.mkdtemp()
        try:
            result = _split_into_single_task_yamls(config_path, temp_dir)
            self.assertEqual(len(result), 3)
            task_paths = [tp for tp, _ in result]
            self.assertEqual(task_paths, ["t1.json", "t2.json", "t3.json"])
        finally:
            os.unlink(config_path)
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_parse_config_documents(self):
        docs = [
            {"kind": "llm", "spec": {"name": "llm-1"}},
            {"kind": "benchmark", "spec": {"tasks": []}},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump_all(docs, f, default_flow_style=False)
            config_path = f.name

        try:
            parsed = _parse_config_documents(config_path)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["kind"], "llm")
            self.assertEqual(parsed[1]["kind"], "benchmark")
        finally:
            os.unlink(config_path)


class TestAdaptiveSemaphore(unittest.IsolatedAsyncioTestCase):
    """Unit tests for AdaptiveSemaphore."""

    async def test_basic_acquire_release(self):
        sem = AdaptiveSemaphore(initial_limit=2)
        await sem.acquire()
        self.assertEqual(sem.current, 1)
        await sem.release()
        self.assertEqual(sem.current, 0)

    async def test_context_manager(self):
        sem = AdaptiveSemaphore(initial_limit=2)
        async with sem:
            self.assertEqual(sem.current, 1)
        self.assertEqual(sem.current, 0)

    async def test_blocks_at_limit(self):
        sem = AdaptiveSemaphore(initial_limit=1)
        await sem.acquire()

        acquired = asyncio.Event()

        async def _try_acquire():
            await sem.acquire()
            acquired.set()

        task = asyncio.create_task(_try_acquire())
        await asyncio.sleep(0.05)
        self.assertFalse(acquired.is_set())

        await sem.release()
        await asyncio.sleep(0.05)
        self.assertTrue(acquired.is_set())
        await sem.release()
        task.cancel()

    async def test_set_limit_increase_wakes_waiters(self):
        sem = AdaptiveSemaphore(initial_limit=1)
        await sem.acquire()  # slot full

        acquired = asyncio.Event()

        async def _try_acquire():
            await sem.acquire()
            acquired.set()

        task = asyncio.create_task(_try_acquire())
        await asyncio.sleep(0.05)
        self.assertFalse(acquired.is_set())

        await sem.set_limit(2)
        await asyncio.sleep(0.05)
        self.assertTrue(acquired.is_set())

        await sem.release()
        await sem.release()
        task.cancel()

    async def test_set_limit_decrease_no_preemption(self):
        sem = AdaptiveSemaphore(initial_limit=3)
        await sem.acquire()
        await sem.acquire()
        self.assertEqual(sem.current, 2)

        await sem.set_limit(1)
        # Already acquired slots still held, no error
        self.assertEqual(sem.current, 2)
        self.assertEqual(sem.limit, 1)

        await sem.release()
        await sem.release()

    async def test_min_limit_enforced(self):
        sem = AdaptiveSemaphore(initial_limit=5, min_limit=2)
        await sem.set_limit(1)
        self.assertEqual(sem.limit, 2)

    async def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            AdaptiveSemaphore(initial_limit=0)
        with self.assertRaises(ValueError):
            AdaptiveSemaphore(initial_limit=1, min_limit=0)


class TestFeedbackControllerAIMD(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the AIMD concurrency adjustment in FeedbackController."""

    @staticmethod
    def _make_result(success: bool, category: ErrorCategory = ErrorCategory.UNKNOWN) -> WorkerResult:
        return WorkerResult(
            task_path="test/task.json",
            success=success,
            error_category=category if not success else ErrorCategory.UNKNOWN,
        )

    async def test_rate_limit_halves_concurrency(self):
        fc = FeedbackController(window_size=20, initial_concurrency=10)
        await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        self.assertEqual(fc.semaphore.limit, 5)

    async def test_connection_reduces_by_quarter(self):
        fc = FeedbackController(window_size=20, initial_concurrency=8)
        await fc.record(self._make_result(success=False, category=ErrorCategory.CONNECTION))
        # 8 * 3 // 4 = 6
        self.assertEqual(fc.semaphore.limit, 6)

    async def test_timeout_no_concurrency_change(self):
        fc = FeedbackController(window_size=20, initial_concurrency=10)
        await fc.record(self._make_result(success=False, category=ErrorCategory.TIMEOUT))
        self.assertEqual(fc.semaphore.limit, 10)

    async def test_auth_no_concurrency_change(self):
        fc = FeedbackController(window_size=20, initial_concurrency=10)
        await fc.record(self._make_result(success=False, category=ErrorCategory.AUTH))
        self.assertEqual(fc.semaphore.limit, 10)

    async def test_unknown_no_concurrency_change(self):
        fc = FeedbackController(window_size=20, initial_concurrency=10)
        await fc.record(self._make_result(success=False, category=ErrorCategory.UNKNOWN))
        self.assertEqual(fc.semaphore.limit, 10)

    async def test_consecutive_success_recovers(self):
        fc = FeedbackController(window_size=20, initial_concurrency=10, min_concurrency=1)
        # First reduce concurrency
        await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        self.assertEqual(fc.semaphore.limit, 5)
        # Need window_size // 2 = 10 consecutive successes to increase by 1
        for _ in range(10):
            await fc.record(self._make_result(success=True))
        self.assertEqual(fc.semaphore.limit, 6)

    async def test_recovery_does_not_exceed_initial(self):
        fc = FeedbackController(window_size=4, initial_concurrency=3, min_concurrency=1)
        # Already at initial — streak of successes should not increase beyond it
        for _ in range(10):
            await fc.record(self._make_result(success=True))
        self.assertEqual(fc.semaphore.limit, 3)

    async def test_min_concurrency_floor(self):
        fc = FeedbackController(window_size=20, initial_concurrency=2, min_concurrency=2)
        await fc.record(self._make_result(success=False, category=ErrorCategory.RATE_LIMIT))
        # 2 // 2 = 1, but min is 2
        self.assertEqual(fc.semaphore.limit, 2)

    async def test_semaphore_property(self):
        fc = FeedbackController(initial_concurrency=5, min_concurrency=2)
        self.assertIsInstance(fc.semaphore, AdaptiveSemaphore)
        self.assertEqual(fc.semaphore.limit, 5)


@pytest.mark.skip(reason="Integration test: requires LLM and MCP services")
class TestParallelBenchmarkRunner(unittest.IsolatedAsyncioTestCase):
    """Integration tests for ParallelBenchmarkRunner."""

    async def test_small_scale_run(self):
        from mcpuniverse.benchmark.parallel import ParallelBenchmarkRunner

        runner = ParallelBenchmarkRunner(
            config="mcpuniverse/benchmark/configs/mcpuniverse/financial_analysis.yaml",
            concurrency=3,
            output_dir="/tmp/test_parallel_results",
            max_retries=1,
        )
        results = await runner.run()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
