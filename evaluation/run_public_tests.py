"""Public test runner for Travel Itinerary Builder.

Runs all 14 test cases against the live /arena/run endpoint, pausing 3 seconds
between cases to respect rate limits. Outputs a summary table and writes
evaluation/results/public_test_results.json.
"""
import argparse
import json
from pathlib import Path
import time
import httpx

def main():
    parser = argparse.ArgumentParser(description="Run public evaluation cases")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the Agent Arena server")
    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    cases_file = eval_dir / "public_cases.json"
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "public_test_results.json"

    if not cases_file.exists():
        print(f"Error: {cases_file} not found.")
        raise SystemExit(1)

    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(cases)} test cases. Target URL: {args.url}\n")
    print(f"{'RESULT':<6} | {'ID':<38} | {'STATUS':<18} | {'STOP REASON':<24} | {'STEPS':<5} | {'LATENCY (ms)':<12}")
    print("-" * 115)

    passed_count = 0
    results_records = []

    for idx, case in enumerate(cases):
        case_id = case.get("id", f"case_{idx+1}")
        expected_status = case.get("expected_status")
        expected_stop_reason = case.get("expected_stop_reason")

        request_payload = case.get("request") or {
            "task": case["task"],
            "external_context": case.get("external_context", []),
            "arena_config": case.get("arena_config", {"max_steps": 6, "fault": "none"})
        }

        t0 = time.perf_counter()
        actual_status = "error"
        actual_stop_reason = "request_failed"
        actual_steps = 0
        latency_ms = 0.0
        passed = False
        error_detail = None

        try:
            with httpx.Client(timeout=75.0) as client:
                resp = client.post(f"{args.url.rstrip('/')}/arena/run", json=request_payload)
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                resp.raise_for_status()
                data = resp.json()

                actual_status = data.get("status")
                actual_stop_reason = data.get("stop_reason")
                actual_steps = data.get("steps", 0)
                if "metrics" in data and "latency_ms" in data["metrics"] and data["metrics"]["latency_ms"] > 0:
                    latency_ms = round(data["metrics"]["latency_ms"], 2)

                # Status check
                status_ok = (actual_status == expected_status)
                # Stop reason check (only if expected_stop_reason is specified)
                stop_reason_ok = True
                if expected_stop_reason:
                    stop_reason_ok = (actual_stop_reason == expected_stop_reason)

                passed = status_ok and stop_reason_ok

        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            error_detail = f"{type(exc).__name__}: {exc}"
            actual_stop_reason = type(exc).__name__

        if passed:
            passed_count += 1
            tag = "PASS"
        else:
            tag = "FAIL"

        print(f"{tag:<6} | {case_id:<38} | {str(actual_status):<18} | {str(actual_stop_reason):<24} | {actual_steps:<5} | {latency_ms:<12.1f}")
        if not passed and error_detail:
            print(f"       -> Error detail: {error_detail}")

        results_records.append({
            "id": case_id,
            "category": case.get("category", "unknown"),
            "expected_status": expected_status,
            "actual_status": actual_status,
            "expected_stop_reason": expected_stop_reason,
            "actual_stop_reason": actual_stop_reason,
            "steps": actual_steps,
            "latency_ms": latency_ms,
            "passed": passed,
            "error": error_detail
        })

        # Pause 3 seconds between cases as required
        if idx < len(cases) - 1:
            time.sleep(3.0)

    print("-" * 115)
    print(f"\nSummary: {passed_count}/{len(cases)} test cases passed.")

    output_summary = {
        "total_cases": len(cases),
        "passed_cases": passed_count,
        "failed_cases": len(cases) - passed_count,
        "pass_rate": f"{(passed_count / len(cases)) * 100:.1f}%",
        "results": results_records
    }

    results_file.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")
    print(f"Detailed results saved to {results_file}")

    raise SystemExit(0 if passed_count == len(cases) else 1)

if __name__ == "__main__":
    main()
