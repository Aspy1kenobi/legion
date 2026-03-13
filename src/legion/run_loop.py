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

Tick sequence (order matters — do not reorder without reading comments):
    1. dispatch_all()               — claim + execute pending goals
    2. _run_consensus_on_completions() — challenge/accept completed goals
    3. _run_strategist()            — push gap goals if queue is empty
    4. _should_halt()               — NOW safe to check: strategist has had
                                      its chance to refill the queue

The strategist MUST fire before _should_halt() or it never runs in short
sessions. When the initial goal completes and is committed, the queue hits
zero. If halt fires first, the strategist never pushes gap goals and auto-halt
triggers on an empty queue that the strategist would have refilled.

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

# ── Node functions ────────────────────────────────────────────────────────────

async def _planner_fn(goal, wm: SharedWorldModel) -> str:
    """
    Procedural node: decomposes goals into plans.
    Reads collective context, produces a structured plan.
    Attribution scaffolding required (JARVIS paper finding) — context
    injection via wm.format_context_for_prompt() provides this.
    """
    from llm_client import call_llm
    from config import Config

    config     = Config()
    context    = wm.format_context_for_prompt(goal.description, top_k=4)
    beliefs    = wm.get_active_beliefs(min_confidence=0.5)
    belief_str = "\n".join(f"- {b.content}" for b in beliefs) or "(none yet)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strategic planner in a multi-agent AI collective. "
                "Your output will be reviewed by an evaluative agent before being "
                "committed as a collective belief. Be specific and actionable. "
                "Connect your plan to what the collective already knows."
            ),
        },
        {
            "role": "user",
            "content": (
                f"GOAL: {goal.description}\n\n"
                f"COLLECTIVE BELIEFS (committed truths):\n{belief_str}\n\n"
                f"RECENT COLLECTIVE ACTIVITY:\n{context or '(none)'}\n\n"
                "Produce a concrete plan to accomplish this goal. "
                "Identify the key steps, dependencies, and the single most "
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


async def _engineer_fn(goal, wm: SharedWorldModel) -> str:
    """
    Procedural node: produces concrete implementations, code, and prototypes.
    Claims goals containing implement/build/code/write/develop/prototype/create.
    Where the planner produces plans, the engineer produces artifacts.
    """
    from llm_client import call_llm
    from config import Config

    config     = Config()
    context    = wm.format_context_for_prompt(goal.description, top_k=4)
    beliefs    = wm.get_active_beliefs(min_confidence=0.5)
    belief_str = "\n".join(f"- {b.content}" for b in beliefs) or "(none yet)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a pragmatic engineer in a multi-agent AI collective. "
                "Your output will be reviewed by an evaluative agent before being "
                "committed as a collective belief. Produce concrete, specific "
                "artifacts: code, pseudocode, data structures, or step-by-step "
                "implementation instructions. Do not produce abstract plans — "
                "produce things that can be directly built or evaluated.\n\n"
                "CODEBASE CONTEXT (read before producing any code):\n"
                "- Language: Python 3.11, asyncio, no new dependencies without explicit justification\n"
                "- LLM calls: use call_llm(messages, config) from llm_client.py — do NOT import torch, "
                "transformers, or any ML library\n"
                "- Node functions signature: async def fn(goal, wm: SharedWorldModel) -> str\n"
                "- All persistence goes through wm.add_event() — do not write files directly\n"
                "- Existing patterns are in run_loop.py: _planner_fn is the canonical example to follow"
            ),
        },
        {
            "role": "user",
            "content": (
                f"GOAL: {goal.description}\n\n"
                f"COLLECTIVE BELIEFS (committed truths):\n{belief_str}\n\n"
                f"RECENT COLLECTIVE ACTIVITY:\n{context or '(none)'}\n\n"
                "Produce a concrete implementation or artifact to accomplish this goal. "
                "Follow the codebase context above exactly. "
                "If the goal involves Python code, write code that fits directly into the "
                "existing codebase without new dependencies."
            ),
        },
    ]

    result, _ = await call_llm(messages, config)
    await wm.add_event(
        agent="engineer",
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

    config = Config()
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

    Capability vocabulary is partitioned deliberately:
        planner   — strategic/structural verbs: plan, decompose, analyze, design, define, structure
        engineer  — execution verbs: implement, build, code, test, integrate, write, develop,
                    prototype, create
        skeptic   — evaluative verbs: review, evaluate, assess, check, verify, challenge, audit

    Overlap between planner and engineer was the routing bug from the previous session.
    Keep these lists disjoint. If a new verb could belong to either, assign it to engineer
    (engineer is the executor; planner is the strategist).
    """
    return [
        LegionNode(
            name="planner",
            role_type="procedural",
            capabilities=["plan", "decompose", "analyze", "design", "define", "structure"],
            fn=_planner_fn,
        ),
        LegionNode(
            name="engineer",
            role_type="procedural",
            capabilities=["implement", "build", "code", "test", "integrate",
                          "write", "develop", "prototype", "create"],
            fn=_engineer_fn,
        ),
        LegionNode(
            name="skeptic",
            role_type="evaluative",
            capabilities=["review", "evaluate", "assess", "check",
                          "verify", "challenge", "audit"],
            fn=_skeptic_fn,
        ),
    ]


# ── Strategist ────────────────────────────────────────────────────────────────

async def _run_strategist(wm: SharedWorldModel, gs: GoalStack) -> int:
    """
    Survey committed beliefs for known gaps and push actionable follow-up goals.

    Called every tick, BEFORE _should_halt(), so it can refill the queue
    before the halt condition is evaluated. This is the fix for the ordering
    bug where auto-halt fired before the strategist had a chance to run.

    Returns the number of new goals pushed this tick (0 if queue was not empty
    or no actionable gaps were found).

    Design rules:
      - Only fires when the goal queue is fully empty (pending == 0 AND active == 0).
        If there is still work to do, the strategist stays silent.
      - Each gap belief maps to exactly one concrete, narrow goal. The goal must
        be specific enough for the skeptic to evaluate it against factual criteria,
        not just opinion. "Implement X" beats "Think about X."
      - Goals are not duplicated: before pushing, check whether a goal with
        matching description already exists (any status). This prevents the
        strategist from re-pushing goals that were already attempted and failed.
      - Pushed at priority 0.6 (below human-seeded 0.7 — human intent leads).
    """
    s = wm.status()
    if s["goals_pending"] > 0 or s["goals_active"] > 0:
        return 0  # Queue has work — strategist is silent

    # Build a set of existing goal descriptions (normalised) for dedup
    existing_descriptions = {
        g.description.strip().lower()
        for g in wm.goals.values()
    }

    # ── Gap → goal mapping ────────────────────────────────────────────────────
    # Each entry: (belief_id_that_signals_gap, goal_description_to_push)
    # Descriptions must:
    #   - contain a keyword from engineer's or planner's capability list
    #   - be narrow enough for the skeptic to return accept/reject on factual grounds
    #   - produce a concrete artifact (code, spec, data structure) not an opinion
    #
    # gap_llm_decomposition is intentionally absent here — it is the seed goal
    # at the entry point. Including it would cause the strategist to push a
    # duplicate on the same tick the seed goal commits.
    GAP_GOALS: list[tuple[str, str]] = [
        (
            "gap_can_handle_keyword",
            "Implement a cosine-similarity alternative to keyword matching in "
            "LegionNode.can_handle() using sentence embeddings, with a fallback "
            "to the existing keyword path when embeddings are unavailable.",
        ),
        (
            "gap_engineer_node_missing",
            # Goal description kept narrow: produce only the async function body
            # and the LegionNode registration line — no ML model classes, no
            # new dependencies. The implementation target is run_loop.py in a
            # Python 3.11 asyncio codebase that calls call_llm() from llm_client.py.
            "Write the async _engineer_fn(goal, wm) function body for run_loop.py "
            "that calls call_llm() with an engineer system prompt and returns the "
            "result string, following the same pattern as _planner_fn. "
            "Also write the LegionNode(...) registration line for _build_default_nodes().",
        ),
        (
            "gap_ethicist_node_missing",
            "Design the system prompt and evaluative criteria for an Ethicist node "
            "that can serve as a second evaluator in ConsensusEngine alongside Skeptic.",
        ),
    ]

    pushed = 0
    for belief_id, goal_description in GAP_GOALS:
        # Only act on beliefs that are actually committed (confidence >= 0.5)
        belief = wm.beliefs.get(belief_id)
        if belief is None or belief.confidence < 0.5:
            continue

        # Skip if a goal with this description was already pushed (any status)
        if goal_description.strip().lower() in existing_descriptions:
            continue

        await gs.push(
            description=goal_description,
            priority=0.6,
            source="strategist",
        )
        pushed += 1

    return pushed


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
        self._log("Bootstrap complete", f"{belief_count} beliefs seeded")

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
        One tick: dispatch → consensus → strategist → halt-check.

        Tick order is load-bearing:
          1. dispatch_all()                  — do the work
          2. _run_consensus_on_completions() — commit results
          3. _run_strategist()               — refill queue from gaps
          4. _should_halt() (in run())       — halt only after strategist fires

        The strategist must run before the halt check. If halt fires on an
        empty queue before the strategist has a chance to push gap goals,
        Legion shuts down after the first goal completes instead of pursuing
        its known architectural gaps autonomously.
        """
        self._tick_count += 1
        self._log(f"── Tick {self._tick_count} ──────────────────────")

        # 1. Dispatch all eligible goals concurrently
        dispatch_result = await self.dispatcher.dispatch_all()
        self._log("Dispatch", dispatch_result)

        # 2. Consensus: challenge/accept any newly completed goals
        await self._run_consensus_on_completions()

        # 3. Strategist: push gap goals if queue is now empty
        #    Must happen BEFORE _should_halt() is evaluated in run()
        new_goals = await _run_strategist(self.wm, self.gs)
        if new_goals:
            self._log("Strategist", f"pushed {new_goals} gap goal(s)")

        # 4. Status snapshot (read AFTER strategist so counts are accurate)
        wm_status = self.wm.status()
        self._log("World model", wm_status)

        # Adaptive sleep
        has_work = wm_status["goals_active"] > 0 or wm_status["goals_pending"] > 0
        return TICK_INTERVAL_ACTIVE if has_work else TICK_INTERVAL_IDLE

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

            result   = relevant_events[0].content
            producer = relevant_events[0].agent

            self._log(f"Consensus: evaluating {goal.id}")
            committed = await self.consensus.evaluate(goal, result, producer)
            self._log(f"Consensus: {'committed' if committed else 'rejected'}", goal.id)

    # ── Halt detection ────────────────────────────────────────────────────────

    def _should_halt(self) -> tuple[bool, str]:
        """
        Check whether the loop should stop.
        Returns (should_halt, reason).

        Called AFTER tick() (which calls the strategist), so the strategist
        has already had its chance to push new goals before this evaluates.
        """
        if self.max_ticks and self._tick_count >= self.max_ticks:
            return True, f"max_ticks={self.max_ticks} reached"

        if self.auto_halt:
            s = self.wm.status()
            if s["goals_pending"] == 0 and s["goals_active"] == 0:
                return True, "goal queue exhausted (strategist found no new gaps)"

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

                # _should_halt() is safe here: tick() ran the strategist first
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
    # Seed goal is intentionally narrow and factual so the skeptic can commit it.
    #
    # REJECTED (too broad/speculative — produces opinion, not committable fact):
    #   "Analyze the Legion architecture and identify the most critical missing capability"
    #
    # ACCEPTED (narrow, concrete, produces an artifact the skeptic can evaluate):
    #   "Design the function signature for LLM-driven goal decomposition"
    #
    # The strategist will push the remaining gap goals automatically once the
    # first goal commits and the queue empties. No need to seed them all here.
    asyncio.run(main(
        initial_goals=[
            "Design the data contract and function signature for LLM-driven "
            "goal decomposition in GoalStack.decompose(), including input format, "
            "output schema, and fallback behaviour when the LLM is unavailable.",
        ],
        max_ticks=10,
    ))