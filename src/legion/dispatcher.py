"""
Legion Dispatcher
=================
Matches goals to nodes and manages the claim/execute/release cycle.

Responsibilities:
    - Node registry: register/deregister LegionNodes at startup
    - Selection: given a goal, find a capable idle node
    - Claim: mark node busy + goal active atomically before dispatch
    - Execute: call node.fn(goal, wm), await result
    - Release: on success → complete goal, mark node idle
              on failure → fail goal, mark node idle, log event

Not responsible for:
    - Deciding WHAT to work on (GoalStack)
    - Deciding WHETHER a result is true (consensus.py)
    - The heartbeat / tick loop (run_loop.py)

Relationship to existing code:
    - debate_async.py called agents directly by name; Dispatcher replaces
      that hard-coded routing with capability-based selection
    - Nodes are thin LegionNode dataclasses; their .fn is the async callable
      that replaces the agent functions in agents.py

Async model:
    - dispatch_one() handles a single goal end-to-end
    - dispatch_all() runs all eligible goals concurrently (asyncio.gather)
    - Both are safe to call from run_loop.py's tick
"""

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world_model import SharedWorldModel, Goal
    from goal_stack import GoalStack


# ── Node definition ───────────────────────────────────────────────────────────

@dataclass
class LegionNode:
    """
    A Legion node: identity + capability declaration + async callable.

    The fn signature is:
        async def fn(goal: Goal, wm: SharedWorldModel) -> str
    It must return a result string and write any intermediate outputs
    to wm via add_event(). It must NOT update goal status — that's
    the dispatcher's job.

    role_type drives evaluative vs procedural behavior:
        "procedural"  — Planner, Engineer: need attribution scaffolding
        "evaluative"  — Skeptic, Ethicist: self-correct via memory retrieval
    """
    name:         str
    role_type:    str                    # "procedural" | "evaluative"
    capabilities: list[str]             # e.g. ["plan", "decompose"]
    fn:           Callable              # async (goal, wm) -> str

    def can_handle(self, goal: "Goal") -> bool:
        """
        True if this node has at least one capability relevant to the goal.

        Current implementation: keyword match against goal description.
        Future: embed goal + capabilities and use cosine similarity.
        """
        desc_lower = goal.description.lower()
        return any(cap.lower() in desc_lower for cap in self.capabilities)


# ── Dispatcher ────────────────────────────────────────────────────────────────

class Dispatcher:
    """
    Routes goals to nodes and manages the full execution lifecycle.

    Usage (from run_loop.py):
        dispatcher = Dispatcher(wm, gs)
        dispatcher.register(planner_node)
        dispatcher.register(skeptic_node)

        # In the tick loop:
        await dispatcher.dispatch_all()
    """

    def __init__(self, world_model: "SharedWorldModel", goal_stack: "GoalStack"):
        self.wm = world_model
        self.gs = goal_stack
        self.nodes: dict[str, LegionNode] = {}

    # ── Registry ──────────────────────────────────────────────────────────────

    def register(self, node: LegionNode) -> None:
        """
        Register a node with the dispatcher and world model.
        Safe to call multiple times — re-registration updates the record.
        Must be called before dispatch_all() for the node to receive work.
        """
        self.nodes[node.name] = node
        # Register synchronously in the node dict; wm registration is async
        # and will be finalized on first dispatch or via register_all()

    async def register_all(self) -> None:
        """Persist all registered nodes to wm. Call once at startup."""
        for node in self.nodes.values():
            await self.wm.register_node(
                name=node.name,
                role_type=node.role_type,
                capabilities=node.capabilities,
            )

    def deregister(self, node_name: str) -> None:
        """Remove a node from the active pool. Does not delete wm record."""
        self.nodes.pop(node_name, None)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _find_node(self, goal: "Goal") -> Optional[LegionNode]:
        """
        Find the best idle node for a goal.

        Selection order:
          1. Node must be registered and wm-tracked as "idle"
          2. Node must can_handle(goal)
          3. Among eligible nodes, prefer the one with fewest tasks_completed
             (load balancing — avoids always routing to the same node)

        Returns None if no eligible node exists.
        """
        eligible = []
        for node in self.nodes.values():
            wm_record = self.wm.nodes.get(node.name)
            if wm_record is None:
                continue
            if wm_record.status != "idle":
                continue
            if not node.can_handle(goal):
                continue
            eligible.append((node, wm_record.tasks_completed))

        if not eligible:
            return None

        # Least-loaded first
        eligible.sort(key=lambda x: x[1])
        return eligible[0][0]

    # ── Claim / Release ───────────────────────────────────────────────────────

    async def _claim(self, node: LegionNode, goal: "Goal") -> None:
        """Mark node busy and goal active before execution begins."""
        record = self.wm.nodes[node.name]
        record.status          = "busy"
        record.last_active     = datetime.now().isoformat()
        record.current_goal_id = goal.id
        await self.wm.update_goal_status(goal.id, status="active", assigned_to=node.name)
        await self.wm.save()

    async def _release(self, node: LegionNode, success: bool = True) -> None:
        """Return node to idle pool after execution completes or fails."""
        record = self.wm.nodes.get(node.name)
        if record is None:
            return
        record.status          = "idle"
        record.current_goal_id = None
        record.last_active     = datetime.now().isoformat()
        if success:
            record.tasks_completed += 1
        await self.wm.save()

    # ── Execution ─────────────────────────────────────────────────────────────

    async def dispatch_one(self, goal: "Goal", followon_budget: int = 0) -> tuple[Optional[bool], int]:
        """
        Full claim → execute → release cycle for a single goal.

        Returns:
            (True, n)   — goal executed successfully; n follow-on goals pushed
            (False, 0)  — goal executed but raised an exception
            (None, 0)   — no capable idle node found; goal stays pending (skipped)

        Never raises — all exceptions are caught, logged, and converted
        to goal failure events so run_loop.py stays alive.
        """
        node = self._find_node(goal)
        if node is None:
            # No capable idle node right now. Goal stays pending.
            # run_loop will retry on next tick.
            await self.wm.add_event(
                agent="dispatcher",
                event_type="dispatch_skipped",
                content=f"No capable idle node for: {goal.description}",
                importance=0.3,
                goal_id=goal.id,
            )
            return None, 0

        await self._claim(node, goal)

        try:
            result = await node.fn(goal, self.wm)

            # Defensive parse — works for planner JSON and plain-string outputs alike.
            # Non-JSON results (engineer, skeptic, ethicist) silently fall back to
            # plan_text=result, follow_on_goals=[], so non-planner nodes are unaffected.
            try:
                parsed = json.loads(result)
                if not isinstance(parsed, dict):
                    raise ValueError("not a dict")
                plan_text = parsed.get("plan") or result
                follow_on_goals = parsed.get("follow_on_goals", [])
                if not isinstance(follow_on_goals, list):
                    follow_on_goals = []
            except (json.JSONDecodeError, ValueError):
                plan_text = result
                follow_on_goals = []

            await self.gs.complete(goal.id, result=plan_text, node_id=node.name)
            await self._release(node, success=True)

            # Push follow-on goals up to the budget limit.
            followons_pushed = 0
            for desc in follow_on_goals:
                if followons_pushed >= followon_budget:
                    break
                await self.gs.push(desc, source="planner")
                followons_pushed += 1

            return True, followons_pushed

        except Exception as e:
            reason = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            await self.gs.fail(goal.id, reason=reason, node_id=node.name)
            await self._release(node, success=False)
            await self.wm.add_event(
                agent=node.name,
                event_type="node_error",
                content=reason,
                importance=0.8,
                goal_id=goal.id,
            )
            return False, 0

    async def dispatch_all(self, followon_budget: int = 0) -> dict:
        """
        Dispatch all currently eligible goals concurrently.

        Pulls all pending unblocked goals from GoalStack, finds a node
        for each, and runs them in parallel via asyncio.gather().

        Goals with no available node are skipped and retried next tick.
        Goals that would exceed available nodes are left pending.

        Returns a summary dict for run_loop.py logging:
            {"dispatched": int, "skipped": int, "failed": int, "followons_pushed": int}
        """
        # Snapshot the eligible queue — don't mutate while iterating
        candidates = [
            g for g in self.wm.get_pending_goals()
            if g.assigned_to is None
        ]

        if not candidates:
            return {"dispatched": 0, "skipped": 0, "failed": 0, "followons_pushed": 0}

        results = await asyncio.gather(
            *[self.dispatch_one(goal, followon_budget) for goal in candidates],
            return_exceptions=False,  # exceptions handled inside dispatch_one
        )

        dispatched       = sum(1 for r, _ in results if r is True)
        failed           = sum(1 for r, _ in results if r is False)
        skipped          = sum(1 for r, _ in results if r is None)
        followons_pushed = sum(n for _, n in results)

        return {
            "dispatched":       dispatched,
            "skipped":          skipped,
            "failed":           failed,
            "followons_pushed": followons_pushed,
        }

    # ── Introspection ─────────────────────────────────────────────────────────

    def status_report(self) -> str:
        """Terse node status for injection into run_loop logs."""
        lines = []
        for node in self.nodes.values():
            record = self.wm.nodes.get(node.name)
            if record is None:
                lines.append(f"  {node.name}: unregistered")
                continue
            goal_str = f" → {record.current_goal_id}" if record.current_goal_id else ""
            lines.append(
                f"  {node.name} [{record.status}]{goal_str} "
                f"(done: {record.tasks_completed})"
            )
        return "Nodes:\n" + "\n".join(lines) if lines else "Nodes: none registered"