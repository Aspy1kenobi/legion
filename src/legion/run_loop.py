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

async def _planner_fn(goal, wm: SharedWorldModel) -> str:
    """
    Procedural node: decomposes goals into plans.

    Routing:
        If the goal description contains a DECOMPOSE_TRIGGER keyword
        (e.g. "decompose", "break down"), the planner calls
        GoalStack.llm_decompose() and returns the confirmation string.
        The goal's subgoals are pushed to the stack inside llm_decompose();
        _planner_fn just passes the summary back as its result.

        Otherwise, the planner produces a standard plan string.

    Attribution scaffolding required (JARVIS paper finding) — context
    injection via wm.format_context_for_prompt() provides this.
    """
    from llm_client import call_llm
    from config import Config
    from goal_stack import GoalStack, DECOMPOSE_TRIGGERS

    config = Config()

    # ── Decompose route ───────────────────────────────────────────────────────
    desc_lower = goal.description.lower()
    if any(trigger in desc_lower for trigger in DECOMPOSE_TRIGGERS):
        gs = GoalStack(wm)
        result = await gs.llm_decompose(goal.id, config, source="planner")
        await wm.add_event(
            agent="planner",
            event_type="node_output",
            content=result,
            importance=goal.priority,
            goal_id=goal.id,
        )
        return result

    # ── Plan route ────────────────────────────────────────────────────────────
    # Episodic context: recent events relevant to this goal
    event_context = wm.format_context_for_prompt(goal.description, top_k=4)

    # Belief context: committed facts filtered by relevance to the goal.
    # top_k=5 to cover cross-cutting goals that touch multiple subsystems.
    # Uses retrieve_context against beliefs converted to pseudo-events so the
    # same recency+importance+relevance scoring applies.
    # Simpler path: filter active beliefs by keyword match against goal description.
    desc_words = set(goal.description.lower().split())
    all_beliefs = wm.get_active_beliefs(min_confidence=0.5)
    # Score each belief by word overlap with goal description
    def _belief_relevance(b):
        b_words = set(b.content.lower().split())
        return len(b_words & desc_words)
    ranked_beliefs = sorted(all_beliefs, key=_belief_relevance, reverse=True)
    top_beliefs = ranked_beliefs[:5]
    belief_str = "\n".join(f"- {b.content}" for b in top_beliefs) or "(none yet)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strategic planner embedded inside a running multi-agent "
                "AI system called Legion. You have direct access to the collective's "
                "committed beliefs and event history — you are not an external "
                "consultant. When analyzing system components, reason about them "
                "directly using the context provided. Do not ask for access to "
                "information already present in the context below. "
                "Your output will be reviewed by an evaluative agent before being "
                "committed as a collective belief. Be specific and actionable."
            ),
        },
        {
            "role": "user",
            "content": (
                f"GOAL: {goal.description}\n\n"
                f"RELEVANT COLLECTIVE BELIEFS (committed facts about this system):\n"
                f"{belief_str}\n\n"
                f"RECENT COLLECTIVE ACTIVITY:\n{event_context or '(none yet)'}\n\n"
                "Produce a concrete analysis or plan to accomplish this goal. "
                "Reason directly from the beliefs and context above. "
                "Identify key findings, dependencies, and the single most "
                "important action to take first."
            ),
        },
    ]

    result, _ = await call_llm(messages, config)
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
        self.consensus = ConsensusEngine(self.wm, config, evaluative)

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
    # Standard plan path test: goal contains no DECOMPOSE_TRIGGERS keywords,
    # so _planner_fn routes to the plan path, not llm_decompose().
    # Expected: single goal completed, no child goals pushed, planner output
    # is specifically about the dispatcher rather than generic planning boilerplate.
    asyncio.run(main(
        initial_goals=[
            "Analyze the dispatcher's node selection logic and identify whether "
            "load balancing across multiple nodes of the same role_type would "
            "improve throughput."
        ],
        max_ticks=10,
    ))