"""TerraVision Agent — evaluation harness.

Run: python -m agent.eval
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.loop import run_agent


@dataclass
class EvalCase:
    name: str
    question: str
    check: Callable[[dict[str, Any]], tuple[bool, str]]


def _no_tool_call(result: dict[str, Any]) -> tuple[bool, str]:
    if len(result["trace"]) == 0:
        return True, "no tool calls made, as expected"
    return False, f"expected no tool calls, got {len(result['trace'])}"


def _tool_called_with_valid_crop(result: dict[str, Any]) -> tuple[bool, str]:
    for step in result["trace"]:
        crop = step["args"].get("crop")
        if crop and crop not in ("Wheat", "Rice", "Maize", "Soybean"):
            return False, f"tool was called with unsupported crop {crop!r}"
    return True, "no unsupported crop values were sent to the tool"


def _no_hallucinated_numbers_on_error(result: dict[str, Any]) -> tuple[bool, str]:
    all_errored = all("error" in step["result"] for step in result["trace"])
    if not all_errored:
        return True, "at least one tool call succeeded — no error-hallucination risk"
    answer = (result["answer"] or "").lower()
    if "t/ha" in answer or "ha " in answer:
        return False, "tool call(s) failed but answer still cites a yield figure"
    return True, "tool call(s) failed and answer correctly avoided inventing numbers"


def _asks_for_missing_info(result: dict[str, Any]) -> tuple[bool, str]:
    if len(result["trace"]) > 0:
        return False, "expected agent to ask for info, but it called the tool instead"
    answer = (result["answer"] or "").lower()
    keywords = ("location", "latitude", "longitude", "crop", "coordinates", "which")
    if any(kw in answer for kw in keywords):
        return True, "answer asks for missing location/crop info"
    return False, "answer doesn't clearly ask for missing info"


def _validation_error_surfaced(result: dict[str, Any]) -> tuple[bool, str]:
    if len(result["trace"]) == 0:
        return False, "expected a tool call attempt with a validation error, got none"
    for step in result["trace"]:
        if "error" in step["result"]:
            return True, f"validation error correctly surfaced: {step['result']['error']}"
    return False, "tool call succeeded but should have failed validation"


def _two_locations_compared(result: dict[str, Any]) -> tuple[bool, str]:
    successful_calls = [s for s in result["trace"] if "error" not in s["result"]]
    if len(successful_calls) < 2:
        return False, f"expected 2 successful tool calls, got {len(successful_calls)}"
    lats = {round(s["args"].get("lat", 0), 1) for s in successful_calls}
    if len(lats) < 2:
        return False, "both calls used the same location — not a real comparison"
    return True, f"tool called for {len(lats)} distinct locations as expected"


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        name="unsupported_crop",
        question="What's the cotton yield outlook near Multan, 30.2, 71.5?",
        check=lambda r: (
            _tool_called_with_valid_crop(r)[0] and _no_hallucinated_numbers_on_error(r)[0],
            "; ".join(
                [_tool_called_with_valid_crop(r)[1], _no_hallucinated_numbers_on_error(r)[1]]
            ),
        ),
    ),
    EvalCase(
        name="missing_info",
        question="What's the yield outlook for my field?",
        check=_asks_for_missing_info,
    ),
    EvalCase(
        name="invalid_coordinates",
        question="What's the rice yield at latitude 200, longitude 73?",
        check=_validation_error_surfaced,
    ),
    EvalCase(
        name="general_knowledge_no_tool_call",
        question="What is NDVI and how is it calculated?",
        check=_no_tool_call,
    ),
    EvalCase(
        name="multi_location_comparison",
        question=(
            "Compare wheat yield outlook between Faisalabad (31.4, 73.1) "
            "and Multan (30.2, 71.5)"
        ),
        check=_two_locations_compared,
    ),
]


def run_eval(verbose: bool = True) -> dict[str, Any]:
    results = []
    n_passed = 0

    for case in EVAL_CASES:
        t0 = time.perf_counter()
        agent_result = run_agent(case.question)
        elapsed = time.perf_counter() - t0

        passed, reason = case.check(agent_result)
        n_passed += int(passed)

        results.append(
            {
                "name": case.name,
                "question": case.question,
                "passed": passed,
                "reason": reason,
                "elapsed_s": round(elapsed, 2),
                "hops": agent_result["hops"],
                "n_tool_calls": len(agent_result["trace"]),
                "answer": agent_result["answer"],
            }
        )

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {case.name} ({elapsed:.1f}s) — {reason}")

    summary = {
        "total": len(EVAL_CASES),
        "passed": n_passed,
        "failed": len(EVAL_CASES) - n_passed,
        "pass_rate": round(n_passed / len(EVAL_CASES) * 100, 1),
        "results": results,
    }
    return summary


if __name__ == "__main__":
    summary = run_eval(verbose=True)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']}%)")
    print(f"{'=' * 60}")

    with open("agent/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Full results written to agent/eval_results.json")

    sys.exit(0 if summary["failed"] == 0 else 1)
