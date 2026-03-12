"""
Legion GoalStack
================
Interface layer over SharedWorldModel.goals.

Responsibilities:
    - push / pop_next / peek — goal lifecycle management
    - Priority ordering — highest priority unblocked goal surfaces first
    - Decomposition — replace a goal with ordered child goals
    - Status reporting — formatted snapshot for agent context injection

Not responsible for:
    - Storage (that's SharedWorldModel)
    - Consensus (that's consensus.py)
    - Dispatch (that's dispatcher.py)

Design rule: GoalStack holds NO independent goal state.
All reads pull from wm.goals. All writes go through wm's async methods.
This eliminates drift between GoalStack and WorldModel.

Async boundary:
    - Write-path methods (push, pop_next, complete, fail, decompose) are async
    - Read-path methods (peek, status_report, get_tree) are sync
    - Callers in run_loop.py will be async — this matches the pattern in debate_async.py
"""

import asyncio
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world_model import SharedWorldModel, Goal


class GoalStack:
    """
    Priority queue interface over SharedWorldModel.goals.

    All nodes interact with goals through this class.
    Direct writes to wm.goals are discouraged outside of GoalStack.

    Usage:
        gs = GoalStack(world_model)

        goal = await gs.push("Implement consensus mechanism", priority=0.9)
        next_goal = await gs.pop_next("planner_node")
        await gs.complete(next_goal.id, result="ConsensusEngine built and tested")

        children = await gs.decompose(goal.id, [
            "Define voting protocol",
            "Implement quorum logic",
            "Write integration test",
        ])
    """

    def __init__(self, world_model: "SharedWorldModel"):
        self.wm = world_model

    # ── Write path (async) ────────────────────────────────────────────────────

    async def push(
        self,
        description: str,
        priority:    float = 0.5,
        source:      str   = "human",
        parent_id:   Optional[str] = None,
        depends_on:  Optional[list[str]] = None,
    ) -> "Goal":
        """
        Add a new goal to the collective's work queue.

        Args:
            description: What needs to be accomplished.
            priority:    0.0–1.0. Higher surfaces first in pop_next().
            source:      "human" or the node name that generated the goal.
                         Self-generated goals (source != "human") are what
                         make Legion proactive rather than reactive.
            parent_id:   Set by decompose(). Marks this as a subgoal.
            depends_on:  List of goal IDs that must be complete before this
                         goal becomes eligible for pop_next().

        Returns:
            The newly created Goal (from world_model.Goal).
        """
        goal = await self.wm.add_goal(
            description=description,
            priority=priority,
            source=source,
            parent_id=parent_id,
            depends_on=depends_on or [],
        )
        await self.wm.add_event(
            agent=source,
            event_type="goal_added",
            content=f"Goal added: {description}",
            importance=priority,
            goal_id=goal.id,
        )
        return goal

    async def pop_next(self, node_id: str) -> Optional["Goal"]:
        """
        Claim the highest-priority unblocked, unassigned goal for a node.

        Marks the goal as "active" and assigns it to node_id atomically.
        Returns None if no eligible goals exist.

        Eligibility rules (mirrors wm.get_pending_goals()):
            - status == "pending"
            - all depends_on goals are "complete"
            - not already assigned

        Args:
            node_id: The claiming node's name (e.g. "planner_node_1").

        Returns:
            The claimed Goal, or None.
        """
        candidates = self.wm.get_pending_goals()
        # Filter out goals already assigned to another node
        # (get_pending_goals returns status=="pending", assigned_to may still be None)
        unassigned = [g for g in candidates if g.assigned_to is None]

        if not unassigned:
            return None

        goal = unassigned[0]  # already sorted by priority desc
        await self.wm.update_goal_status(goal.id, status="active", assigned_to=node_id)
        await self.wm.add_event(
            agent=node_id,
            event_type="goal_claimed",
            content=f"Node {node_id} claimed goal: {goal.description}",
            importance=goal.priority,
            goal_id=goal.id,
        )
        # Return the updated goal object
        return self.wm.goals[goal.id]

    async def complete(self, goal_id: str, result: str, node_id: str = "unknown") -> None:
        """
        Mark a goal complete and record its result as a collective event.

        After marking complete, any goals that depend_on this goal_id
        become eligible for pop_next() automatically — no explicit unblocking
        needed because eligibility is computed dynamically in get_pending_goals().

        Args:
            goal_id: ID of the goal to complete.
            result:  The outcome or artifact produced. Stored as an event
                     so all nodes can retrieve it via wm.retrieve_context().
            node_id: The node reporting completion (for event attribution).
        """
        if goal_id not in self.wm.goals:
            raise KeyError(f"Goal '{goal_id}' not found in world model.")

        goal = self.wm.goals[goal_id]
        await self.wm.update_goal_status(goal_id, status="complete")
        await self.wm.add_event(
            agent=node_id,
            event_type="goal_complete",
            content=f"Goal complete: {goal.description}\nResult: {result}",
            importance=goal.priority,
            goal_id=goal_id,
        )

    async def fail(self, goal_id: str, reason: str, node_id: str = "unknown") -> None:
        """
        Mark a goal failed. Goals that depend on this goal remain blocked.

        The run_loop should detect orphaned blocked goals and decide whether
        to abandon them, re-route, or escalate to human.

        Args:
            goal_id: ID of the goal that failed.
            reason:  What went wrong. Recorded as an event.
            node_id: The node reporting the failure.
        """
        if goal_id not in self.wm.goals:
            raise KeyError(f"Goal '{goal_id}' not found in world model.")

        goal = self.wm.goals[goal_id]
        await self.wm.update_goal_status(goal_id, status="abandoned")
        await self.wm.add_event(
            agent=node_id,
            event_type="goal_failed",
            content=f"Goal failed: {goal.description}\nReason: {reason}",
            importance=goal.priority,
            goal_id=goal_id,
        )

    async def decompose(
        self,
        goal_id:  str,
        subgoals: list[str],
        source:   str = "planner",
    ) -> list["Goal"]:
        """
        Replace a goal with an ordered list of child goals.

        The parent goal is marked "active" (it's not complete until all
        children are). Children are created with sequential dependencies:
        child[n] depends_on child[n-1], so they execute in order by default.

        To allow parallel execution, the caller should push subgoals manually
        with explicit depends_on instead of using decompose().

        Args:
            goal_id:  The goal to decompose. Must exist and be "pending" or "active".
            subgoals: Ordered list of descriptions for child goals.
            source:   Node performing the decomposition (for event attribution).

        Returns:
            List of created child Goals.
        """
        if goal_id not in self.wm.goals:
            raise KeyError(f"Goal '{goal_id}' not found in world model.")

        parent = self.wm.goals[goal_id]

        # Inherit parent priority; children can be adjusted after creation
        priority = parent.priority
        created: list["Goal"] = []
        prev_id: Optional[str] = None

        for description in subgoals:
            child = await self.push(
                description=description,
                priority=priority,
                source=source,
                parent_id=goal_id,
                depends_on=[prev_id] if prev_id else [],
            )
            created.append(child)
            prev_id = child.id

        # Mark parent active — it tracks child completion, not direct execution
        await self.wm.update_goal_status(goal_id, status="active")
        await self.wm.add_event(
            agent=source,
            event_type="goal_decomposed",
            content=(
                f"Goal decomposed: {parent.description}\n"
                f"Into {len(subgoals)} subgoals: {', '.join(subgoals)}"
            ),
            importance=priority,
            goal_id=goal_id,
        )

        return created

    # ── Read path (sync) ──────────────────────────────────────────────────────

    def peek(self) -> Optional["Goal"]:
        """
        Return the highest-priority unblocked, unassigned goal without claiming it.
        Returns None if the queue is empty.
        """
        candidates = [g for g in self.wm.get_pending_goals() if g.assigned_to is None]
        return candidates[0] if candidates else None

    def get_tree(self, goal_id: str) -> dict:
        """
        Return a goal and all its descendants as a nested dict.
        Useful for run_loop introspection and status reporting.
        """
        if goal_id not in self.wm.goals:
            return {}

        goal = self.wm.goals[goal_id]
        children = [
            g for g in self.wm.goals.values()
            if g.parent_id == goal_id
        ]

        return {
            "id":          goal.id,
            "description": goal.description,
            "status":      goal.status,
            "priority":    goal.priority,
            "assigned_to": goal.assigned_to,
            "children":    [self.get_tree(c.id) for c in children],
        }

    def status_report(self) -> str:
        """
        Formatted snapshot of the current goal queue.
        Injected into node prompts so agents know what the collective is working on.

        Format is intentionally terse — this goes into a prompt context window.
        """
        goals = list(self.wm.goals.values())
        if not goals:
            return "Goal queue: empty"

        pending  = [g for g in goals if g.status == "pending"]
        active   = [g for g in goals if g.status == "active"]
        complete = [g for g in goals if g.status == "complete"]
        failed   = [g for g in goals if g.status == "abandoned"]

        lines = [f"Goals: {len(pending)} pending | {len(active)} active | "
                 f"{len(complete)} complete | {len(failed)} failed"]

        if active:
            lines.append("\nACTIVE:")
            for g in sorted(active, key=lambda x: x.priority, reverse=True):
                assignee = f" → {g.assigned_to}" if g.assigned_to else ""
                lines.append(f"  [{g.priority:.1f}] {g.description}{assignee}")

        if pending:
            lines.append("\nPENDING (next up):")
            for g in self.wm.get_pending_goals()[:3]:  # top 3 only
                lines.append(f"  [{g.priority:.1f}] {g.description}")
            if len(pending) > 3:
                lines.append(f"  ... and {len(pending) - 3} more")

        return "\n".join(lines)