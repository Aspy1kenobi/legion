"""
Legion ConsensusEngine
======================
Implements the challenge/accept commit protocol for Legion beliefs.

Mechanism (Option B — Challenge/Accept):
    1. A procedural node produces an output (result string)
    2. ConsensusEngine selects an evaluative node to challenge it
    3. Evaluative node receives the output + world model context and returns
       a structured verdict: {"verdict": "accept"|"reject", "reason": "...",
                               "confidence": 0.0-1.0}
    4. If verdict == "accept" (or evaluator is unavailable/malformed):
       → result is committed as a Belief in wm
    5. If verdict == "reject":
       → result is NOT committed; rejection reason is logged as an event
       → goal is returned to "pending" for retry or human escalation

Design decisions:
    - Structured signal, not parsed language. Evaluator must return JSON.
      Malformed response → accept with confidence=0.3 and a warning event.
      A broken evaluator must not permanently block a goal.
    - One challenge round maximum. This is not a debate loop. The evaluative
      node gets one pass. If it rejects, the goal retries — it doesn't enter
      an infinite challenge cycle.
    - Evaluative node selection: prefer nodes with role_type=="evaluative"
      that are currently idle. If none available, skip challenge and commit
      with reduced confidence (logged). Cost constraint makes this correct —
      blocking commits until an evaluator is free would stall the system.
    - Confidence propagates: accepted beliefs carry the evaluator's stated
      confidence. Unchallenged beliefs carry a default of 0.6 (committed,
      but flagged as unreviewed).

Relationship to existing code:
    - Evaluative agent prompts (skeptic, ethicist) in prompts.py are the
      template for challenge prompts. Legion challenge prompt extends them
      with a JSON response requirement.
    - llm_client.call_llm() is the inference call, same as all other nodes.
    - Beliefs written here are what wm.get_active_beliefs() returns — the
      collective's committed worldview that all nodes read as ground truth.
"""

import json
import asyncio
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from llm_client import call_llm

if TYPE_CHECKING:
    from world_model import SharedWorldModel, Goal
    from dispatcher import LegionNode


# ── Prompt ────────────────────────────────────────────────────────────────────

CHALLENGE_SYSTEM = (
    "You are a rigorous evaluator in a multi-agent AI system. "
    "Your job is to assess whether a proposed output adequately accomplishes "
    "the stated goal. You accept good work. You reject work that has clear "
    "logical errors, fails to address the goal, or would cause downstream "
    "failures if treated as true.\n\n"
    "IMPORTANT: Evaluate the output against the goal only. Do not apply "
    "criteria that are irrelevant to the goal type. For implementation tasks, "
    "assess correctness and completeness. For planning tasks, assess "
    "feasibility and specificity. Do not raise ethical or stakeholder concerns "
    "unless the goal explicitly involves ethics or stakeholders.\n\n"
    "You must respond with a single JSON object and nothing else. "
    "No preamble, no explanation outside the JSON. Schema:\n"
    '{"verdict": "accept" | "reject", "reason": "<one sentence>", '
    '"confidence": <float 0.0-1.0>}'
)

CHALLENGE_USER = (
    "GOAL: {goal_description}\n\n"
    "PROPOSED OUTPUT:\n{result}\n\n"
    "COLLECTIVE CONTEXT (recent relevant events):\n{context}\n\n"
    "Evaluate the proposed output. Does it adequately accomplish the goal? "
    "Are there logical errors, unsupported claims, or failure modes that "
    "would make this unsafe to treat as a committed belief?\n\n"
    "Respond with JSON only."
)

# Belief ID derived from goal — stable across retries
def _belief_id(goal: "Goal") -> str:
    return f"belief_{goal.id}"


# ── Verdict parsing ───────────────────────────────────────────────────────────

def _parse_verdict(raw: str) -> dict:
    """
    Parse the evaluator's JSON response into a verdict dict.

    On any parse failure, returns a safe default:
        {"verdict": "accept", "confidence": 0.3, "reason": "parse_failed", "malformed": True}

    This default is intentional: a broken evaluator must not permanently
    block a goal. The "malformed" flag ensures the failure is visible
    in the event log and the low confidence marks the belief as unreviewed.
    """
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        data = json.loads(cleaned)

        verdict    = str(data.get("verdict", "accept")).lower()
        confidence = float(data.get("confidence", 0.5))
        reason     = str(data.get("reason", "no reason given"))

        if verdict not in ("accept", "reject"):
            verdict = "accept"
            confidence = 0.3

        confidence = max(0.0, min(1.0, confidence))

        return {
            "verdict":    verdict,
            "confidence": confidence,
            "reason":     reason,
            "malformed":  False,
        }

    except Exception as e:
        return {
            "verdict":    "accept",
            "confidence": 0.3,
            "reason":     f"parse_failed: {e}",
            "malformed":  True,
        }


# ── ConsensusEngine ───────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Manages the challenge/accept protocol and belief commitment.

    Usage (called by Dispatcher after a node completes a goal):
        engine = ConsensusEngine(wm, config, evaluative_nodes)
        committed = await engine.evaluate(goal, result, producing_node_name)
        if committed:
            # belief is now in wm.beliefs, visible to all nodes
        else:
            # goal was returned to pending for retry
    """

    def __init__(
        self,
        world_model:      "SharedWorldModel",
        config,
        evaluative_nodes: list["LegionNode"],
        max_retries:      int = 2,
    ):
        self.wm               = world_model
        self.config           = config
        self.evaluative_nodes = {n.name: n for n in evaluative_nodes}
        self.max_retries      = max_retries

    # ── Evaluator selection ───────────────────────────────────────────────────

    def _select_evaluator(self) -> Optional["LegionNode"]:
        """
        Find the least-used idle evaluative node (lowest tasks_completed).
        Ties broken by dict insertion order (stable, arbitrary).
        Returns None if all evaluative nodes are busy — caller will skip challenge.

        Least-used selection naturally rotates across evaluators when load is
        equal and load-balances when it isn't. No explicit round-robin state needed.
        """
        idle = [
            (self.wm.nodes[n.name].tasks_completed, n)
            for n in self.evaluative_nodes.values()
            if self.wm.nodes.get(n.name) and self.wm.nodes[n.name].status == "idle"
        ]
        if not idle:
            return None
        _, node = min(idle, key=lambda pair: pair[0])
        return node

    # ── Core protocol ─────────────────────────────────────────────────────────

    async def evaluate(
        self,
        goal:     "Goal",
        result:   str,
        producer: str,
    ) -> bool:
        """
        Run the challenge/accept protocol for a completed goal.

        Returns:
            True  → result committed as a belief, goal remains complete.
            False → result rejected, goal returned to pending for retry.
        """
        evaluator = self._select_evaluator()

        if evaluator is None:
            await self._commit(
                goal=goal,
                result=result,
                confidence=0.6,
                source=producer,
                reason="no evaluator available",
                challenged=False,
            )
            return True

        record = self.wm.nodes[evaluator.name]
        record.status          = "busy"
        record.current_goal_id = goal.id
        await self.wm.save()

        try:
            verdict = await self._challenge(goal, result, evaluator)
        finally:
            record.status          = "idle"
            record.current_goal_id = None
            record.tasks_completed += 1
            await self.wm.save()

        if verdict["malformed"]:
            await self.wm.add_event(
                agent="consensus",
                event_type="evaluator_malformed",
                content=f"Evaluator {evaluator.name} returned unparseable verdict. "
                        f"Committing with low confidence. Raw reason: {verdict['reason']}",
                importance=0.7,
                goal_id=goal.id,
            )

        if verdict["verdict"] == "accept":
            await self._commit(
                goal=goal,
                result=result,
                confidence=verdict["confidence"],
                source=f"{producer} / verified by {evaluator.name}",
                reason=verdict["reason"],
                challenged=True,
            )
            return True
        else:
            await self._reject(goal, verdict, evaluator.name)
            return False

    # ── Challenge call ────────────────────────────────────────────────────────

    async def _challenge(
        self,
        goal:      "Goal",
        result:    str,
        evaluator: "LegionNode",
    ) -> dict:
        """
        Call the evaluative node's LLM with the challenge prompt.
        Returns a parsed verdict dict.
        """
        context = self.wm.format_context_for_prompt(
            query=goal.description,
            top_k=4,
            agent_filter=None,
        )

        messages = [
            {"role": "system", "content": CHALLENGE_SYSTEM},
            {
                "role": "user",
                "content": CHALLENGE_USER.format(
                    goal_description=goal.description,
                    result=result,
                    context=context or "(no prior collective context)",
                ),
            },
        ]

        raw_response, _usage = await call_llm(messages, self.config)
        return _parse_verdict(raw_response)

    # ── Commit / Reject ───────────────────────────────────────────────────────

    async def _commit(
        self,
        goal:       "Goal",
        result:     str,
        confidence: float,
        source:     str,
        reason:     str,
        challenged: bool,
    ) -> None:
        """Write the result to wm.beliefs and log the commit event."""
        belief_id = _belief_id(goal)
        await self.wm.add_belief(
            belief_id=belief_id,
            content=result,
            confidence=confidence,
            source=source,
            tags=[goal.id, "consensus"],
            evidence=[reason],
        )

        # If this goal was generated to close a gap, mark that gap resolved
        if goal.closes_gap:
            await self.wm.resolve_gap(goal.closes_gap)
            await self.wm.add_event(
                agent="consensus",
                event_type="gap_resolved",
                content=f"Gap resolved: {goal.closes_gap} (closed by goal: {goal.description})",
                importance=0.8,
                goal_id=goal.id,
                tags=["gap", "resolved"],
            )

        challenged_str = "challenged+accepted" if challenged else "unchallenged"
        await self.wm.add_event(
            agent="consensus",
            event_type="belief_formed",
            content=(
                f"Belief committed [{challenged_str}, confidence={confidence:.2f}]: "
                f"{goal.description}\n{result[:200]}{'...' if len(result) > 200 else ''}"
            ),
            importance=confidence,
            goal_id=goal.id,
            tags=["belief", challenged_str],
        )

    async def _reject(
        self,
        goal:           "Goal",
        verdict:        dict,
        evaluator_name: str,
    ) -> None:
        """
        Log the rejection and return the goal to pending for retry.

        Tracks rejection count via a belief with low confidence.
        If max_retries exceeded, marks goal abandoned and logs escalation.
        """
        retry_belief_id = f"retry_count_{goal.id}"
        existing = self.wm.beliefs.get(retry_belief_id)
        retry_count = int(existing.content) + 1 if existing else 1

        await self.wm.add_belief(
            belief_id=retry_belief_id,
            content=str(retry_count),
            confidence=0.0,
            source="consensus",
            tags=["internal", "retry_counter"],
        )

        await self.wm.add_event(
            agent="consensus",
            event_type="belief_rejected",
            content=(
                f"Evaluator {evaluator_name} rejected output for: {goal.description}\n"
                f"Reason: {verdict['reason']}\n"
                f"Retry {retry_count}/{self.max_retries}"
            ),
            importance=0.8,
            goal_id=goal.id,
            tags=["rejection"],
        )

        if retry_count >= self.max_retries:
            await self.wm.update_goal_status(goal.id, status="abandoned")
            # Clean up the retry counter — it served its purpose and would
            # otherwise accumulate permanently across long runs.
            await self.wm.delete_belief(retry_belief_id)
            await self.wm.add_event(
                agent="consensus",
                event_type="goal_escalated",
                content=(
                    f"Goal exceeded max retries ({self.max_retries}) and requires "
                    f"human review: {goal.description}"
                ),
                importance=1.0,
                goal_id=goal.id,
                tags=["escalation", "human_required"],
            )
        else:
            await self.wm.update_goal_status(goal.id, status="pending", clear_assignment=True)

    # ── Introspection ─────────────────────────────────────────────────────────

    def status_report(self) -> str:
        """Terse evaluator status for run_loop logging."""
        lines = []
        for name, node in self.evaluative_nodes.items():
            record = self.wm.nodes.get(name)
            status = record.status if record else "unregistered"
            lines.append(f"  {name} [{status}]")
        return "Evaluators:\n" + "\n".join(lines) if lines else "Evaluators: none registered"