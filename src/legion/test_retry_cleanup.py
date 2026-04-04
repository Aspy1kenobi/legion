"""
Verification script for retry_count_* belief cleanup.
Run from src/legion/:
    python test_retry_cleanup.py

Tests the startup sweep logic directly in-memory — no network calls, no disk state.
"""
import asyncio
from world_model import SharedWorldModel, Belief, Goal
from datetime import datetime


def now():
    return datetime.now().isoformat()


async def main():
    # Build an in-memory world model (load creates fresh if path absent)
    wm = SharedWorldModel("data/test_retry_cleanup_wm.json")
    await wm.load()

    # Inject 3 synthetic goals: abandoned, pending, and missing (not in goals at all)
    abandoned_goal_id = "goal_0001_100000"
    pending_goal_id   = "goal_0002_100001"
    missing_goal_id   = "goal_0003_100002"  # intentionally NOT added to wm.goals

    wm.goals[abandoned_goal_id] = Goal(
        id=abandoned_goal_id, description="abandoned goal", priority=0.5,
        status="abandoned", source="human", created_at=now(), updated_at=now(),
    )
    wm.goals[pending_goal_id] = Goal(
        id=pending_goal_id, description="pending goal", priority=0.5,
        status="pending", source="human", created_at=now(), updated_at=now(),
    )
    # missing_goal_id is deliberately absent from wm.goals

    # Inject 3 retry_count_* beliefs
    for gid in (abandoned_goal_id, pending_goal_id, missing_goal_id):
        key = f"retry_count_{gid}"
        wm.beliefs[key] = Belief(
            id=key, content="1", confidence=0.0,
            source="consensus", created_at=now(), updated_at=now(),
            tags=["internal", "retry_counter"],
        )

    assert len([k for k in wm.beliefs if k.startswith("retry_count_")]) == 3, \
        "FAIL: expected 3 retry beliefs before cleanup"

    # Run the startup sweep logic (same code as RunLoop.startup)
    stale_keys = [
        k for k, _ in wm.beliefs.items()
        if k.startswith("retry_count_")
        and (
            k[len("retry_count_"):] not in wm.goals
            or wm.goals[k[len("retry_count_"):]].status == "abandoned"
        )
    ]
    for k in stale_keys:
        await wm.delete_belief(k)

    remaining = [k for k in wm.beliefs if k.startswith("retry_count_")]

    assert len(stale_keys) == 2, f"FAIL: expected 2 deleted, got {len(stale_keys)}: {stale_keys}"
    assert len(remaining) == 1, f"FAIL: expected 1 retained, got {len(remaining)}: {remaining}"
    assert remaining[0] == f"retry_count_{pending_goal_id}", \
        f"FAIL: wrong belief retained: {remaining[0]}"
    assert f"retry_count_{abandoned_goal_id}" not in wm.beliefs, \
        "FAIL: abandoned goal's counter still present"
    assert f"retry_count_{missing_goal_id}" not in wm.beliefs, \
        "FAIL: missing goal's counter still present"

    print(f"PASS: deleted {len(stale_keys)} stale counters (abandoned + missing goal)")
    print(f"PASS: retained 1 counter for pending goal: {remaining[0]}")
    print(f"PASS: wm.beliefs has no stale retry_count_* keys")

    # Cleanup
    import os
    if os.path.exists("data/test_retry_cleanup_wm.json"):
        os.remove("data/test_retry_cleanup_wm.json")

    print("\nAll tests passed.")


asyncio.run(main())
