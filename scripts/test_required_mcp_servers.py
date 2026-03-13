#!/usr/bin/env python3
"""Smoke test MCP servers referenced by benchmark tasks and agent configs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcpuniverse.mcp.client import MCPClient  # noqa: E402
from mcpuniverse.mcp.manager import MCPManager  # noqa: E402


@dataclass
class CheckResult:
    """Result of one server smoke test."""

    server: str
    status: str
    detail: str
    tool_count: int = 0


def _normalize_server_entries(entries: object) -> list[str]:
    names: list[str] = []
    if not isinstance(entries, list):
        return names
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
        elif isinstance(entry, str):
            names.append(entry)
    return names


def _discover_from_json(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return _normalize_server_entries(data.get("mcp_servers"))


def _discover_from_yaml(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []

    names: list[str] = []
    try:
        docs = list(yaml.safe_load_all(content))
    except yaml.YAMLError:
        return names

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if str(doc.get("kind", "")).lower() != "agent":
            continue
        spec = doc.get("spec")
        if not isinstance(spec, dict):
            continue
        config = spec.get("config")
        if not isinstance(config, dict):
            continue
        names.extend(_normalize_server_entries(config.get("servers")))
    return names


def discover_required_servers(config_root: Path) -> dict[str, list[str]]:
    """Return mapping: server name -> source config paths."""
    server_sources: dict[str, list[str]] = defaultdict(list)
    for path in sorted(config_root.rglob("*")):
        if path.suffix == ".json":
            names = _discover_from_json(path)
        elif path.suffix in {".yaml", ".yml"}:
            names = _discover_from_yaml(path)
        else:
            continue
        relpath = str(path.relative_to(REPO_ROOT))
        for name in names:
            server_sources[name].append(relpath)
    return dict(sorted(server_sources.items()))


def _unspecified_stdio_params(manager: MCPManager, server_name: str) -> list[str]:
    config = manager.get_config(server_name)
    missing = []
    missing.extend(config.stdio.list_unspecified_params())
    missing.extend(
        value
        for value in config.env.values()
        if isinstance(value, str) and "{{" in value and "}}" in value
    )
    return missing


async def smoke_test_server(
    manager: MCPManager,
    server_name: str,
    timeout: int,
) -> CheckResult:
    """Connect to one server and verify that tools can be listed."""
    available = set(manager.get_configs())
    if server_name not in available:
        return CheckResult(server=server_name, status="FAIL", detail="server not found in server_list.json")

    missing = _unspecified_stdio_params(manager, server_name)
    if missing:
        detail = ", ".join(sorted(set(missing)))
        return CheckResult(server=server_name, status="SKIP", detail=f"missing config/env: {detail}")

    async def _run() -> CheckResult:
        client = None
        try:
            config = manager.get_config(server_name)
            client = MCPClient(name=f"{server_name}_smoke_test")
            await client.connect_to_stdio_server(
                config=config,
                timeout=timeout,
                retries=1,
            )
            tools = await asyncio.wait_for(client.list_tools(), timeout=timeout)
            tool_names = ", ".join(tool.name for tool in tools[:5])
            if len(tools) > 5:
                tool_names = f"{tool_names}, ..."
            detail = tool_names or "connected but no tools returned"
            return CheckResult(
                server=server_name,
                status="PASS",
                detail=detail,
                tool_count=len(tools),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            message = str(exc).strip() or exc.__class__.__name__
            return CheckResult(server=server_name, status="FAIL", detail=message)
        finally:
            if client is not None:
                try:
                    await client.cleanup()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

    try:
        return await _run()
    except asyncio.TimeoutError:
        return CheckResult(
            server=server_name,
            status="FAIL",
            detail=f"timed out after {timeout}s",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test MCP servers required by benchmark tasks.",
    )
    parser.add_argument(
        "--config-root",
        default="mcpuniverse/benchmark/configs",
        help="Directory to scan for benchmark JSON/YAML configs.",
    )
    parser.add_argument(
        "--servers",
        nargs="*",
        help="Only test the specified server names.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-server stdio connection timeout in seconds.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only print discovered server names without connecting.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the config files that reference each server.",
    )
    return parser


def _print_discovery(server_sources: dict[str, list[str]], show_sources: bool) -> None:
    print(f"Discovered {len(server_sources)} MCP servers")
    for server, sources in server_sources.items():
        print(f"- {server}: {len(sources)} references")
        if show_sources:
            for source in sources:
                print(f"  {source}")


async def _run_checks(args: argparse.Namespace) -> int:
    config_root = (REPO_ROOT / args.config_root).resolve()
    server_sources = discover_required_servers(config_root)
    if not server_sources:
        print(f"No MCP servers found under {config_root}", file=sys.stderr)
        return 1

    if args.servers:
        requested = set(args.servers)
        server_sources = {name: server_sources[name] for name in sorted(requested) if name in server_sources}
        missing = sorted(requested - set(server_sources))
        for name in missing:
            print(f"[WARN] requested server not found in configs: {name}", file=sys.stderr)
        if not server_sources:
            print("No requested servers were found in scanned configs.", file=sys.stderr)
            return 1

    _print_discovery(server_sources, show_sources=args.show_sources)
    if args.discover_only:
        return 0

    manager = MCPManager()
    results: list[CheckResult] = []
    for server_name in server_sources:
        print(f"\n==> Testing {server_name}")
        result = await smoke_test_server(manager, server_name=server_name, timeout=args.timeout)
        results.append(result)
        suffix = f" ({result.tool_count} tools)" if result.tool_count else ""
        print(f"[{result.status}] {result.server}{suffix}: {result.detail}")

    passed = sum(result.status == "PASS" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    failed = sum(result.status == "FAIL" for result in results)

    print("\nSummary")
    print(f"- PASS: {passed}")
    print(f"- SKIP: {skipped}")
    print(f"- FAIL: {failed}")

    return 1 if failed else 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run_checks(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception:  # pylint: disable=broad-exception-caught
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
