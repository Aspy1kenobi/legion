# Legion — Claude Code Session Guide

## What this project is

Legion is a proactive multi-agent AI hivemind in Python 3.11/asyncio.
LLM-driven nodes collaborate through a shared world model, consensus engine,
and goal dispatcher to set, decompose, and pursue goals autonomously.
It is the successor to a prior project called JARVIS.

## Repo layout

```
~/legion/
  src/
    legion/          ← Legion-native modules (run from here)
      run_loop.py    ← entry point and node definitions
      goal_stack.py  ← goal lifecycle, llm_decompose()
      dispatcher.py  ← node routing, claim/execute/release
      consensus.py   ← challenge/accept commit protocol
      world_model.py ← shared persistent state
      bootstrap_beliefs.py ← seed beliefs on startup
    llm_client.py    ← call_llm() — only LLM interface, routes Anthropic/Ollama
    config.py        ← Config dataclass, reads from .env
    agent_memory.py  ← per-agent episodic memory (JARVIS legacy)
  data/
    world_model.json ← runtime state, gitignored, recreated on clean run
  .env               ← ANTHROPIC_API_KEY and model config, never committed
```

## Standard run commands

```bash
# Clean run (always do this before a fresh experiment)
cd ~/legion/src/legion
rm -f data/world_model.json && python run_loop.py

# Inspect world model after a run
python -c "
import json, sys
wm = json.load(open('data/world_model.json'))
print('=== GOALS ===')
for gid, g in wm['goals'].items():
    print(f\"{gid}: [{g['status']}] parent={g.get('parent_id','—')} depends={g['depends_on']}\")
    print(f\"  {g['description'][:90]}\")
print()
print('=== LAST 6 EVENTS ===')
for e in wm['events'][-6:]:
    print(f\"[{e['event_type']} | {e['agent']}]\")
    print(f\"  {e['content'][:120]}\")
"

# Inspect planner output specifically
python -c "
import json
wm = json.load(open('data/world_model.json'))
for e in wm['events']:
    if e['event_type'] == 'node_output' and e['agent'] == 'planner':
        print(e['content'])
        print('---')
"
```

## Commit discipline

**Only commit after a clean shutdown line appears in the output:**
```
[Legion HH:MM:SS] Shutdown complete {...}
```

Commit format:
```bash
git add -A && git commit -m "feat/fix: description" && git push
```

## Architecture — core data flow

```
SharedWorldModel → GoalStack → Dispatcher → node.fn() → ConsensusEngine → Belief
```

- **SharedWorldModel** (`world_model.py`): all state lives here — beliefs, goals, events, node registry. asyncio.Lock on all writes. `_save_unlocked()` for internal callers holding the lock; `save()` for external callers.
- **GoalStack** (`goal_stack.py`): interface over wm.goals. `push/complete/fail/decompose/llm_decompose`. Holds NO independent state — all reads from wm.
- **Dispatcher** (`dispatcher.py`): `dispatch_one` returns `True` (success) / `False` (execution error) / `None` (no capable node — skipped). `dispatch_all` runs all eligible goals concurrently via asyncio.gather.
- **ConsensusEngine** (`consensus.py`): challenge/accept protocol. One evaluative node gets one veto round. Malformed evaluator response → accept with confidence=0.3. Goal exceeding max_retries → abandoned + escalation event.
- **RunLoop** (`run_loop.py`): tick loop. `_run_strategist()` must execute before `_should_halt()` or gap goals never get pushed in short sessions.

## Nodes — current pool

| Node | role_type | capabilities | fn |
|------|-----------|-------------|-----|
| planner | procedural | plan, decompose, analyze, design, build, implement, create, define, structure | `_planner_fn` |
| skeptic | evaluative | review, evaluate, assess, check, test, verify, challenge, audit | `_skeptic_fn` |
| engineer | procedural | implement, build, code, write, create, develop, construct, generate | `_engineer_fn` |
| ethicist | evaluative | ethics, values, fairness, harm, safety, oversight, privacy, risk | `_ethicist_fn` |

**Capability list disjointness is critical.** Any vocabulary overlap between nodes causes systematic misrouting. Planner must not contain execution vocabulary that would also match engineer goals.

## LLM decomposition

`GoalStack.llm_decompose(goal_id, config, source)`:
- Triggered by `DECOMPOSE_TRIGGERS` keywords in goal description: `{"decompose", "break down", "break into", "split into", "subdivide"}`
- Calls LLM → parses numbered list (2–5 items) → prefixes each subgoal with `"implement: "` to guarantee `can_handle()` match → calls `decompose()` → returns confirmation string
- Raises `ValueError` on unparseable output (dispatcher catches, marks goal failed, retry next tick)
- `MAX_DECOMPOSE_DEPTH = 2` with `_goal_depth()` parent-chain traversal prevents unbounded recursion

## `can_handle()` keyword matching

`LegionNode.can_handle(goal)` matches capability keywords against `goal.description.lower()`. This is fragile — subgoal descriptions must contain capability keywords or dispatch is skipped. The `"implement: "` prefix in `llm_decompose()` is the current workaround. See `gap_can_handle_keyword` in bootstrap beliefs.

## Belief injection

`_planner_fn` uses `wm.retrieve_context(goal.description, top_k=3)` for belief context — relevance-filtered, not a full dump of all active beliefs. Dumping all beliefs causes anchoring on stale/irrelevant context (prior issue, fixed).

## Bootstrap beliefs

27 atomic beliefs seeded from `bootstrap_beliefs.py` on every startup. `add_belief()` updates in place — re-running does not duplicate. Key beliefs to keep current:
- `module_goal_stack_pending` — update when decomposition capability changes
- `gap_*` beliefs — update when gaps are closed
- `finding_*` beliefs — experimental results, rarely change

## Known patterns and failure modes

**`skipped` vs `failed` in dispatch output:**
- `skipped` = no capable idle node for the goal description (keyword mismatch)
- `failed` = node was found and ran but threw an exception
- If you see persistent `skipped` on subgoals, the descriptions lack capability keywords

**Belief injection anchoring:** if planner output is generic boilerplate rather than specific to the goal, `retrieve_context` is pulling irrelevant beliefs. Check `top_k` and query string.

**Goal proliferation:** depth limit (`MAX_DECOMPOSE_DEPTH = 2`) caps recursion. Width is capped at 5 in `llm_decompose()` prompt + `subgoals[:5]` slice. If goals_total grows unexpectedly, check strategist gap goal descriptions for trigger keyword overlap.

**Engineer hallucination:** engineer node's system prompt must include a CODEBASE CONTEXT block with the actual file structure. Without it, engineer hallucinates PyTorch/transformers imports for ML tasks.

**Nested asyncio lock deadlock:** `_save_unlocked()` exists to prevent this. Internal write methods that already hold `self._lock` call `_save_unlocked()`. External callers use `save()`. Never call `save()` from inside a `async with self._lock` block.

## Session handoff pattern

At the start of a new session, run:
```bash
git log --oneline -5
```
And paste the output to establish where the last session ended.

## Current known gaps (as of last session)

- Parent auto-completion via `_maybe_complete_parent()` — needs verification
  it fires correctly when all siblings complete
- Strategist gap goals — verify strategist still pushes correctly after
  _planner_fn routing change (no strategist run since that commit)
- Engineer and ethicist nodes — defined in prior sessions but need a live
  run to confirm routing and output quality