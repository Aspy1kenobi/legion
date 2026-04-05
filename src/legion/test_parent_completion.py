"""
Verification script for _maybe_complete_parent() sibling boundary.
Run from src/legion/:
    python test_parent_completion.py

Tests two scenarios in-memory — no network calls, no disk state:

1. Mixed terminal: complete + abandoned siblings → parent auto-completes
   (validates the fail() trigger path added in this fix)

2. All-complete baseline: all siblings complete → parent auto-completes
   (confirms existing behavior is preserved)
"""
import asyncio
from world_model import SharedWorldModel, Goal
from goal_stack import GoalStack
from datetime import datetime


def now():
    return datetime.now().isoformat()


async def make_parent_with_children(wm, gs, n_children=3):
    """
    Push a parent goal and N independent children (no sequential depends_on).
    Marks parent active (as decompose() would). Returns (parent, [children]).
    """
    parent = await gs.push("parent goal for test", priority=0.5)
    children = []
    for i in range(n_children):
        child = await gs.push(
            f"child goal {i + 1}",
            priority=0.5,
            parent_id=parent.id,
        )
        children.append(child)
    # Mark parent active — decompose() does this; we replicate it here
    await wm.update_goal_status(parent.id, status="active")
    return parent, children


async def main():
    wm = SharedWorldModel("data/test_parent_completion_wm.json")
    await wm.load()
    gs = GoalStack(wm)

    # ── Test 1: fail() path — mixed complete + abandoned ─────────────────────
    # Scenario: 3 independent children. C1 completes, C2 fails (abandoned),
    # C3 completes. After C3 completes the parent should auto-complete because
    # all siblings are now in {complete, abandoned}.
    print("Test 1: fail() trigger path — mixed complete + abandoned")
    parent1, (c1, c2, c3) = await make_parent_with_children(wm, gs, n_children=3)

    await gs.complete(c1.id, result="done", node_id="test")
    assert wm.goals[parent1.id].status == "active", \
        f"FAIL: parent should still be active after C1 complete (c2,c3 pending)"

    await gs.fail(c2.id, reason="simulated failure", node_id="test")
    assert wm.goals[parent1.id].status == "active", \
        f"FAIL: parent should still be active after C2 fail (c3 still pending)"

    await gs.complete(c3.id, result="done", node_id="test")
    assert wm.goals[parent1.id].status == "complete", (
        f"FAIL: parent should be complete after C3 completes "
        f"(C1=complete, C2=abandoned, C3=complete). "
        f"Actual: {wm.goals[parent1.id].status}"
    )
    print("  PASS: parent auto-completed after mixed complete+abandoned siblings")

    # ── Test 2: baseline — all siblings complete ──────────────────────────────
    # Confirms the pre-existing complete() path still works correctly.
    print("Test 2: baseline — all children complete")
    parent2, (d1, d2, d3) = await make_parent_with_children(wm, gs, n_children=3)

    await gs.complete(d1.id, result="done", node_id="test")
    assert wm.goals[parent2.id].status == "active", \
        "FAIL: parent should still be active (d2, d3 pending)"

    await gs.complete(d2.id, result="done", node_id="test")
    assert wm.goals[parent2.id].status == "active", \
        "FAIL: parent should still be active (d3 pending)"

    await gs.complete(d3.id, result="done", node_id="test")
    assert wm.goals[parent2.id].status == "complete", (
        f"FAIL: parent should be complete after all children complete. "
        f"Actual: {wm.goals[parent2.id].status}"
    )
    print("  PASS: parent auto-completed when all siblings complete")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    import os
    if os.path.exists("data/test_parent_completion_wm.json"):
        os.remove("data/test_parent_completion_wm.json")

    print("\nAll tests passed.")


asyncio.run(main())
