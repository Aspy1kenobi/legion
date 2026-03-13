"""
Legion RunLoop
==============
The autonomous heartbeat that drives the hivemind.

This is the top-level entry point for Legion. It wires together:
    SharedWorldModel → GoalStack → Dispatcher → ConsensusEngine

and runs a continuous tick loop until interrupted or the goal queue
is exhausted.

Tick model: fixed interval with adaptive sleep.
    - TICK_INTERVAL_ACTIVE:  seconds between ticks when work is in progress
    - TICK_INTERVAL_IDLE:    seconds between ticks when queue is empty
    Two constants, one conditional. No notification infrastructure needed.
    Tune these experimentally once the loop is running — same approach as
    RESPONSE_TRUNCATE_CHARS / EXECUTION_MODE in debate_async.py.

Shutdown:
    - KeyboardInterrupt (Ctrl-C): clean shutdown, state persisted to wm
    - goal_queue exhausted + no active goals: optional auto-halt (configurable)
    - max_ticks reached: hard stop for experiment runs

Startup sequence:
    1. Load world model from disk (or create fresh)
    2. Register nodes with dispatcher and wm
    3. Optionally seed initial goals (human or from prior session)
    4. Enter tick loop

Usage:
    # Minimal — seed one goal and run
    asyncio.run(main(
        initial_goals=["Build and test the consensus mechanism"],
    ))

    # Experiment mode — fixed tick count, then halt
    asyncio.run(main(
        initial_goals=["Analyze the Legion architecture for gaps"],
        max_ticks=20,
    ))
"""

import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                          # src/legion/
sys.path.insert(0, os.path.dirname(_HERE))         # src/

import asyncio
import signal
from datetime import datetime
from typing import Optional

from world_model import SharedWorldModel
from goal_stack import GoalStack
from dispatcher import Dispatcher, LegionNode
from consensus import ConsensusEngine
from bootstrap_beliefs import bootstrap_beliefs

# ── Tick constants ────────────────────────────────────────────────────────────

TICK_INTERVAL_ACTIVE = 2.0   # seconds — work is in progress, check frequently
TICK_INTERVAL_IDLE   = 10.0  # seconds — queue empty, no point hammering wm

# ── Node definitions ──────────────────────────────────────────────────────────
# These are the Legion-native replacements for the agent functions in agents.py.
# Each fn signature: async (goal: Goal, wm: SharedWorldModel) -> str
#
# Phase 1 implementations are intentionally minimal — they call the LLM with
# goal context and return the response. Richer behavior (self-retrieval,
# multi-step reasoning) layers on top of this foundation.

MAX_DECOMPOSE_DEPTH = 2  # tunable — raise to allow deeper trees


def _goal_depth(goal, wm: SharedWorldModel) -> int:
    """
    Count hops from goal to root via parent_id chain.
    Root goal (no parent) = depth 0. Direct child = depth 1. Etc.
    Caps at MAX_DECOMPOSE_DEPTH + 1 to avoid unbounded traversal on corrupt data.
    """
    depth = 0
    current = goal
    while current.parent_id and depth <= MAX_DECOMPOSE_DEPTH:
        parent = wm.goals.get(current.parent_id)
        if parent is None:
            break
        current = parent
        depth += 1
    return depth


async def _planner_fn(goal, wm: SharedWorldModel) -> str:
    """
    Procedural node: decomposes goals into plans OR subgoal trees autonomously.

    Decision logic (single LLM call):
      - If goal is complex/multi-step AND depth < MAX_DECOMPOSE_DEPTH:
            → return JSON {"decompose": true, "subgoals": [...]}
      - Otherwise (atomic, directly answerable, or at depth limit):
            → return JSON {"decompose": false, "result": "..."}

    Depth limit (MAX_DECOMPOSE_DEPTH=2) prevents unbounded recursion.
    Goals at or beyond the limit are forced to atomic execution regardless
    of LLM preference. Limit is a module constant — raise it to allow
    deeper trees when legitimate multi-level decomposition is needed.

    Attribution scaffolding (JARVIS paper finding): collective beliefs and
    context are injected explicitly so the procedural node doesn't collapse.
    """
    import json
    from llm_client import call_llm
    from config import Config
    from goal_stack import GoalStack

    config     = Config()
    gs         = GoalStack(wm)
    depth      = _goal_depth(goal, wm)
    at_limit   = depth >= MAX_DECOMPOSE_DEPTH
    context    = wm.format_context_for_prompt(goal.description, top_k=4)
    beliefs    = wm.get_active_beliefs(min_confidence=0.5)
    belief_str = "\n".join(f"- {b.content}" for b in beliefs) or "(none yet)"

    if at_limit:
        # Force atomic — tell the LLM it must execute, not decompose
        system_content = (
            "You are a strategic planner in a multi-agent AI collective. "
            "Your output will be reviewed by an evaluative agent before being "
            "committed as a collective belief. Be specific and actionable.\n\n"
            "You must respond with a single JSON object and nothing else.\n\n"
            "This goal is a leaf node — execute it directly. Do NOT decompose.\n"
            'Respond with: {"decompose": false, "result": "<your full answer>"}'
        )
    else:
        system_content = (
            "You are a strategic planner in a multi-agent AI collective. "
            "Your output will be reviewed by an evaluative agent before being "
            "committed as a collective belief. Be specific and actionable.\n\n"
            "You must respond with a single JSON object and nothing else — "
            "no preamble, no explanation outside the JSON.\n\n"
            "If the goal is complex or multi-step (requires multiple distinct "
            "phases of work), respond with:\n"
            '{"decompose": true, "subgoals": ["<step 1>", "<step 2>", ...]}\n\n'
            "Subgoals must be concrete and specific. Each must be completable "
            "by a single agent. Use 2–5 subgoals maximum. Subgoal descriptions "
            "must contain capability keywords: 'analyze', 'design', 'plan', "
            "'implement', 'build', 'code', 'test', or 'evaluate'.\n\n"
            "If the goal is atomic or directly answerable, respond with:\n"
            '{"decompose": false, "result": "<your full answer>"}'
        )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"GOAL: {goal.description}\n\n"
                f"COLLECTIVE BELIEFS (committed truths):\n{belief_str}\n\n"
                f"RECENT COLLECTIVE ACTIVITY:\n{context or '(none)'}\n\n"
                + (
                    "Execute this goal directly. Respond with JSON only."
                    if at_limit else
                    "Decide: is this goal atomic (answer it directly) or complex "
                    "(decompose into subgoals)? Respond with JSON only."
                )
            ),
        },
    ]

    raw, _ = await call_llm(messages, config)

    # ── Parse LLM response ────────────────────────────────────────────────────
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(l for l in lines if not l.startswith("```")).strip()
        data = json.loads(cleaned)
    except Exception as e:
        # Parse failure → treat as atomic, use raw text as result
        await wm.add_event(
            agent="planner",
            event_type="node_output",
            content=f"[parse_failed: {e}] {raw}",
            importance=goal.priority,
            goal_id=goal.id,
            tags=["parse_error"],
        )
        return raw

    # ── Decompose path ────────────────────────────────────────────────────────
    if data.get("decompose") and data.get("subgoals") and not at_limit:
        subgoals = [str(s) for s in data["subgoals"] if str(s).strip()]
        if not subgoals:
            # Malformed — fall through to atomic
            data["decompose"] = False
            data["result"] = raw
        else:
            children = await gs.decompose(goal.id, subgoals, source="planner")
            summary = (
                f"Decomposed into {len(children)} subgoals:\n" +
                "\n".join(f"  {i+1}. {c.description}" for i, c in enumerate(children))
            )
            await wm.add_event(
                agent="planner",
                event_type="node_output",
                content=summary,
                importance=goal.priority,
                goal_id=goal.id,
                tags=["decomposed"],
            )
            return summary

    # ── Atomic path ───────────────────────────────────────────────────────────
    result = str(data.get("result", raw))
    await wm.add_event(
        agent="planner",
        event_type="node_output",
        content=result,
        importance=goal.priority,
        goal_id=goal.id,
    )
    return result


async def _skeptic_fn(goal, wm: SharedWorldModel) -> str:
    """
    Evaluative node: challenges outputs and identifies failure modes.
    Self-corrects via memory retrieval unprompted (JARVIS paper finding).
    Used by ConsensusEngine as the default challenger.
    """
    from llm_client import call_llm
    from config import Config

    config  = Config()
    # Self-retrieval: skeptic reads its own prior outputs — the mechanism
    # behind emergent self-correction per the JARVIS paper
    self_context  = wm.format_context_for_prompt(
        goal.description, top_k=3, agent_filter="skeptic"
    )
    group_context = wm.format_context_for_prompt(goal.description, top_k=4)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a rigorous skeptic in a multi-agent AI collective. "
                "Your job is to find what's wrong before it fails in practice. "
                "You are constructive: for every problem you identify, suggest "
                "what would need to be true for the plan to work."
            ),
        },
        {
            "role": "user",
            "content": (
                f"GOAL: {goal.description}\n\n"
                f"YOUR PRIOR ANALYSIS ON RELATED TOPICS:\n"
                f"{self_context or '(none)'}\n\n"
                f"COLLECTIVE ACTIVITY:\n{group_context or '(none)'}\n\n"
                "Identify the unstated assumptions, failure modes, and evidence "
                "gaps in how this goal is being approached. Be specific."
            ),
        },
    ]

    result, _ = await call_llm(messages, config)
    await wm.add_event(
        agent="skeptic",
        event_type="node_output",
        content=result,
        importance=goal.priority,
        goal_id=goal.id,
    )
    return result


async def _engineer_fn(goal, wm: SharedWorldModel) -> str:
    """
    Procedural node: handles implementation, coding, and testing subgoals.

    Phase 1 stub: logs intent and returns a structured acknowledgment.
    Exists to absorb implementation subgoals generated by planner decomposition
    so they don't pile up unhandled. Real implementation in Phase 2.
    """
    result = (
        f"[Engineer stub] Received implementation goal: {goal.description}\n"
        f"Phase 1: Logging intent. Full implementation pending Phase 2 engineer build.\n"
        f"Would produce: code, tests, and integration artifacts for this subgoal."
    )
    await wm.add_event(
        agent="engineer",
        event_type="node_output",
        content=result,
        importance=goal.priority,
        goal_id=goal.id,
        tags=["stub"],
    )
    return result


def _build_default_nodes() -> list[LegionNode]:
    """
    Construct the default Legion node pool.
    Extend this list as new node types are added.
    """
    return [
        LegionNode(
            name="planner",
            role_type="procedural",
            capabilities=["plan", "decompose", "analyze", "design", "build",
                          "implement", "create", "define", "structure"],
            fn=_planner_fn,
        ),
        LegionNode(
            name="engineer",
            role_type="procedural",
            capabilities=["implement", "build", "code", "test", "integrate",
                          "write", "develop", "prototype"],
            fn=_engineer_fn,
        ),
        LegionNode(
            name="skeptic",
            role_type="evaluative",
            capabilities=["review", "evaluate", "assess", "check", "test",
                          "verify", "challenge", "audit"],
            fn=_skeptic_fn,
        ),
    ]


# ── RunLoop ───────────────────────────────────────────────────────────────────

class RunLoop:
    """
    Legion's autonomous heartbeat.

    Wires the full stack together and drives the tick loop.
    All state lives in wm — RunLoop itself is stateless between ticks.

    Args:
        wm_path:       Path to world model JSON. Created if absent.
        max_ticks:     Hard stop after N ticks. None = run until interrupted.
        auto_halt:     Stop when goal queue is empty and no active goals.
        nodes:         Override default node pool (for testing or extension).
        initial_goals: Seed goals to push before the loop starts.
    """

    def __init__(
        self,
        wm_path:       str = "data/world_model.json",
        max_ticks:     Optional[int] = None,
        auto_halt:     bool = True,
        nodes:         Optional[list[LegionNode]] = None,
        initial_goals: Optional[list[str]] = None,
    ):
        self.wm_path       = wm_path
        self.max_ticks     = max_ticks
        self.auto_halt     = auto_halt
        self.node_list     = nodes or _build_default_nodes()
        self.initial_goals = initial_goals or []

        # Populated during startup()
        self.wm:         Optional[SharedWorldModel] = None
        self.gs:         Optional[GoalStack]        = None
        self.dispatcher: Optional[Dispatcher]       = None
        self.consensus:  Optional[ConsensusEngine]  = None

        self._running    = False
        self._tick_count = 0

    # ── Startup ───────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Initialize all components and seed initial goals."""
        from config import Config
        config = Config()

        # 1. World model
        self.wm = SharedWorldModel(self.wm_path)
        await self.wm.load()
        belief_count = await bootstrap_beliefs(self.wm)
        self._log(f"Bootstrap complete", f"{belief_count} beliefs seeded")


        # 2. Goal stack
        self.gs = GoalStack(self.wm)

        # 3. Dispatcher
        self.dispatcher = Dispatcher(self.wm, self.gs)
        for node in self.node_list:
            self.dispatcher.register(node)
        await self.dispatcher.register_all()
        self._log("Nodes registered", list(self.dispatcher.nodes.keys()))

        # 4. Consensus engine — evaluative nodes only
        evaluative = [n for n in self.node_list if n.role_type == "evaluative"]
        self.consensus = ConsensusEngine(self.wm, config, evaluative, max_retries=3)

        # 5. Seed initial goals
        for description in self.initial_goals:
            goal = await self.gs.push(description, priority=0.7, source="human")
            self._log(f"Seeded goal: {goal.id}", description)

        self._running = True
        self._log("Legion startup complete")

    # ── Tick ──────────────────────────────────────────────────────────────────

    async def tick(self) -> float:
        """
        One tick: dispatch eligible goals, run consensus on completions.
        Returns the sleep interval to use before the next tick.
        """
        self._tick_count += 1
        self._log(f"── Tick {self._tick_count} ──────────────────────")

        # Dispatch all eligible goals concurrently
        dispatch_result = await self.dispatcher.dispatch_all()
        self._log("Dispatch", dispatch_result)

        # Run consensus on any goals that completed this tick
        # (goals whose status just became "complete" but have no belief yet)
        await self._run_consensus_on_completions()

        # Abandon goals whose dependencies were abandoned (prevents infinite pending)
        await self._abandon_orphaned_goals()

        # Status snapshot
        wm_status = self.wm.status()
        self._log("World model", wm_status)

        # Adaptive sleep: idle if nothing is happening
        has_active = wm_status["goals_active"] > 0
        has_pending = wm_status["goals_pending"] > 0
        if has_active or has_pending:
            return TICK_INTERVAL_ACTIVE
        return TICK_INTERVAL_IDLE

    async def _run_consensus_on_completions(self) -> None:
        """
        Find recently completed goals that haven't been through consensus yet
        and run the challenge/accept protocol on them.

        Detection: a goal is "needs consensus" if status=="complete" and
        no belief with id belief_{goal.id} exists in wm.beliefs.
        """
        from consensus import _belief_id

        needs_consensus = [
            g for g in self.wm.goals.values()
            if g.status == "complete"
            and _belief_id(g) not in self.wm.beliefs
        ]

        for goal in needs_consensus:
            # Find the most recent node_output event for this goal
            relevant_events = [
                e for e in reversed(self.wm.events)
                if e.goal_id == goal.id and e.event_type == "node_output"
            ]
            if not relevant_events:
                continue

            result    = relevant_events[0].content
            producer  = relevant_events[0].agent

            self._log(f"Consensus: evaluating {goal.id}")
            committed = await self.consensus.evaluate(goal, result, producer)
            self._log(f"Consensus: {'committed' if committed else 'rejected'}", goal.id)

    async def _abandon_orphaned_goals(self) -> None:
        """
        Abandon any pending goal whose dependency was abandoned.
        Prevents orphaned goals from holding the queue open indefinitely.
        Recursive via repeated tick calls — each pass clears one level of depth.
        """
        for goal in list(self.wm.goals.values()):
            if goal.status != "pending":
                continue
            for dep_id in goal.depends_on:
                dep = self.wm.goals.get(dep_id)
                if dep and dep.status == "abandoned":
                    await self.wm.update_goal_status(goal.id, status="abandoned")
                    await self.wm.add_event(
                        agent="run_loop",
                        event_type="goal_escalated",
                        content=(
                            f"Goal abandoned: dependency {dep_id} was abandoned.\n"
                            f"Blocked goal: {goal.description}"
                        ),
                        importance=0.9,
                        goal_id=goal.id,
                        tags=["orphaned", "escalation"],
                    )
                    self._log(f"Orphan abandoned: {goal.id}", goal.description[:60])
                    break  # one abandoned dep is enough; move to next goal

    # ── Halt detection ────────────────────────────────────────────────────────

    def _should_halt(self) -> tuple[bool, str]:
        """
        Check whether the loop should stop.
        Returns (should_halt, reason).
        """
        if self.max_ticks and self._tick_count >= self.max_ticks:
            return True, f"max_ticks={self.max_ticks} reached"

        if self.auto_halt:
            s = self.wm.status()
            if s["goals_pending"] == 0 and s["goals_active"] == 0:
                return True, "goal queue exhausted"

        return False, ""

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start Legion and run until halted."""
        await self.startup()

        # Register clean shutdown on SIGINT/SIGTERM
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        try:
            while self._running:
                sleep_interval = await self.tick()

                halt, reason = self._should_halt()
                if halt:
                    self._log(f"Halting: {reason}")
                    break

                self._log(f"Sleeping {sleep_interval}s")
                await asyncio.sleep(sleep_interval)

        except Exception as e:
            self._log(f"RunLoop error: {e}")
            raise
        finally:
            await self.shutdown()

    def _request_stop(self) -> None:
        """Signal handler — sets flag for clean loop exit."""
        self._log("Stop requested")
        self._running = False

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Persist state and log final status."""
        if self.wm:
            await self.wm.save()
            self._log("Shutdown complete", self.wm.status())

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, detail=None) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        detail_str = f" {detail}" if detail is not None else ""
        print(f"[Legion {ts}] {msg}{detail_str}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(
    initial_goals: list[str] = None,
    wm_path:       str = "data/world_model.json",
    max_ticks:     Optional[int] = None,
    auto_halt:     bool = True,
):
    loop = RunLoop(
        wm_path=wm_path,
        max_ticks=max_ticks,
        auto_halt=auto_halt,
        initial_goals=initial_goals or [],
    )
    await loop.run()


if __name__ == "__main__":
    # Default run: seed one goal and let Legion work autonomously
    asyncio.run(main(
        initial_goals=[
            "Design and implement a self-monitoring capability for Legion: "
            "analyze current gaps, design a health-check mechanism, and "
            "build a prototype that reports node status and goal queue depth."
        ],
        max_ticks=20,
    ))