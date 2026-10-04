import json
import time
import urllib.request

def test_case(fault: str):
    print("=" * 40)
    print(f"TEST CASE FAULT: {fault}")
    print("=" * 40)
    body = {
        "task": "Build my itinerary from the bookings provided.",
        "external_context": [{
            "source": "email",
            "content": "Flight PK123 Islamabad to Istanbul departs 2026-11-05 08:30, arrives 2026-11-05 12:10 local. Confirmation ABC123. Hotel Sultan Inn, Istanbul, check-in 2026-11-05 15:00, check-out 2026-11-08 11:00. Confirmation H778."
        }],
        "arena_config": {"max_steps": 6, "fault": fault}
    }
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/arena/run",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    elapsed = round(time.time() - t0, 2)
    print(f"Status: {data.get('status')}")
    print(f"Stop Reason: {data.get('stop_reason')}")
    print(f"Steps: {data.get('steps')}")
    print(f"Total Elapsed: {elapsed}s")
    print("Tool Calls:")
    for tc in data.get("tool_calls", []):
        print(f"  - {tc}")
    print("Events:")
    for ev in data.get("events", []):
        print(f"  - {ev}")
    clean_resp = data.get("final_response", "")[:400].encode("ascii", errors="replace").decode()
    print("Final Response snippet:")
    print(clean_resp)
    print()

if __name__ == "__main__":
    for f in ["none", "tool_timeout", "malformed_tool_output", "invalid_agent_decision"]:
        test_case(f)
