"""
Verification script for format_context_for_prompt() budget cap.
Run from src/legion/:
    python test_context_budget.py
"""
import asyncio
from world_model import SharedWorldModel, Event
from datetime import datetime


def make_event(i: int) -> Event:
    return Event(
        id=f"evt_{i:06d}",
        event_type="node_output",
        agent="planner",
        content=f"This is synthetic event number {i}. " + ("x" * 50),
        importance=0.5,
        timestamp=datetime.now().isoformat(),
        goal_id=None,
        tags=[],
    )


async def main():
    wm = SharedWorldModel("data/test_budget_wm.json")
    await wm.load()

    # Inject 25 events directly (no disk write needed for retrieval test)
    for i in range(25):
        wm.events.append(make_event(i))
    wm._event_counter = 25

    # Test 1: max_chars=500 should truncate (each event ~100 chars formatted)
    result = wm.format_context_for_prompt("test query", top_k=25, max_chars=500)
    assert len(result) <= 520, f"FAIL: output too long: {len(result)}"
    assert "[context truncated" in result, f"FAIL: no truncation notice in output"
    print(f"PASS: truncated result is {len(result)} chars")
    print(f"Truncation notice: {result[result.rfind('[context'):]}")

    # Test 2: large max_chars should NOT truncate
    result2 = wm.format_context_for_prompt("test query", top_k=5, max_chars=4000)
    assert "[context truncated" not in result2, "FAIL: spurious truncation on large budget"
    print(f"PASS: no truncation at 4000 chars (output: {len(result2)} chars)")

    # Test 3: default (no max_chars) should behave identically to old behavior
    result3 = wm.format_context_for_prompt("test query", top_k=5)
    assert "[context truncated" not in result3, "FAIL: default changed"
    print(f"PASS: default no-cap behavior preserved")

    # Cleanup
    import os
    if os.path.exists("data/test_budget_wm.json"):
        os.remove("data/test_budget_wm.json")

    print("\nAll tests passed.")


asyncio.run(main())
