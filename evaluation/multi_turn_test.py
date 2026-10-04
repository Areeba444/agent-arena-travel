"""Multi-turn clarification test for Travel Itinerary Builder.

Exercises multi-turn conversational memory via /chat:
1. Turn 1 sends a flight booking with missing travel date -> expects needs_clarification.
2. Turn 2 provides the missing travel date -> expects completed.
3. DELETE /chat/{session_id} to clear session memory.
Saves the live transcript to evaluation/results/multi_turn_test.json.
"""
import argparse
import json
from pathlib import Path
import time
from uuid import uuid4
import httpx

def main():
    parser = argparse.ArgumentParser(description="Multi-turn clarification test")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the Agent Arena server")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "multi_turn_test.json"

    session_id = f"session-clarify-{uuid4().hex[:16]}"
    transcript = {
        "session_id": session_id,
        "turns": [],
        "success": False,
        "cleared": False
    }

    print(f"Starting Multi-Turn Clarification Test on session: {session_id}")
    print(f"Target URL: {args.url}\n")

    with httpx.Client(timeout=75.0) as client:
        # ─── TURN 1 ─────────────────────────────────────────────────
        task_1 = (
            "Please build an itinerary for my flight. Flight: Emirates EK-205 from Milan (MXP) "
            "to New York (JFK), departing at 14:00 and arriving at 17:30 (booking code EK-9912). "
            "Passenger: Elena Rostova."
        )
        print("Turn 1 Prompt: Flight provided with NO date.")
        t0 = time.perf_counter()
        resp_1 = client.post(
            f"{args.url.rstrip('/')}/chat",
            json={
                "session_id": session_id,
                "task": task_1,
                "external_context": [],
                "arena_config": {"max_steps": 6, "fault": "none"}
            }
        )
        resp_1.raise_for_status()
        data_1 = resp_1.json()
        latency_1 = round((time.perf_counter() - t0) * 1000, 2)

        status_1 = data_1.get("status")
        msg_1 = data_1.get("final_response")
        steps_1 = data_1.get("steps", 0)

        print(f"Turn 1 Response -> Status: {status_1} (Steps: {steps_1}, Latency: {latency_1}ms)")
        print(f"Turn 1 Message: {msg_1[:160]}...\n")

        transcript["turns"].append({
            "turn": 1,
            "task": task_1,
            "status": status_1,
            "stop_reason": data_1.get("stop_reason"),
            "steps": steps_1,
            "latency_ms": latency_1,
            "final_response": msg_1
        })

        # Pause 3 seconds between turns
        time.sleep(3.0)

        # ─── TURN 2 ─────────────────────────────────────────────────
        task_2 = "The departure date for flight EK-9912 is November 10, 2026 (2026-11-10)."
        print("Turn 2 Prompt: Providing the missing travel date (2026-11-10).")
        t0 = time.perf_counter()
        resp_2 = client.post(
            f"{args.url.rstrip('/')}/chat",
            json={
                "session_id": session_id,
                "task": task_2,
                "external_context": [],
                "arena_config": {"max_steps": 6, "fault": "none"}
            }
        )
        resp_2.raise_for_status()
        data_2 = resp_2.json()
        latency_2 = round((time.perf_counter() - t0) * 1000, 2)

        status_2 = data_2.get("status")
        msg_2 = data_2.get("final_response")
        steps_2 = data_2.get("steps", 0)

        print("Turn 2 Message:")
        try:
            print(msg_2)
        except Exception:
            print(msg_2.encode('ascii', errors='replace').decode())
        print()

        transcript["turns"].append({
            "turn": 2,
            "task": task_2,
            "status": status_2,
            "stop_reason": data_2.get("stop_reason"),
            "steps": steps_2,
            "latency_ms": latency_2,
            "final_response": msg_2
        })

        # ─── TURN 3: DELETE SESSION ─────────────────────────────────
        del_resp = client.delete(f"{args.url.rstrip('/')}/chat/{session_id}")
        if del_resp.status_code == 200:
            transcript["cleared"] = True
            print(f"Session {session_id} successfully cleared on server.")

        turn1_ok = (status_1 == "needs_clarification")
        turn2_ok = (status_2 == "completed")
        transcript["success"] = (turn1_ok and turn2_ok)

    out_file.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    print(f"\nMulti-turn evidence saved to {out_file}")

    if transcript["success"]:
        print("RESULT: PASS (Turn 1 correctly requested clarification, Turn 2 completed itinerary).")
        raise SystemExit(0)
    else:
        print(f"RESULT: FAIL (Turn 1: {status_1}, Turn 2: {status_2})")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
