# Legion Project — Session Kickoff
_Paste this at the start of every new Claude session._

## What This Is
Building "Legion" — a Geth-style hivemind AI collective. Multiple agents sharing
a world model, pursuing self-directed goals autonomously. Not a chatbot. Not a
research tool. A living program.

## Current Stack
- Language: Python 3.12
- Model: Anthropic API (claude-haiku-4-5) — temporary; will migrate to local when hardware allows
- Hardware: M1 MacBook Air, 8GB RAM (real constraint — no local training yet)
- Location: ~/jarvis-lab/

## Project Structure
```
jarvis-lab/
├── src/                        # Main system (all substantive code lives here)
│   ├── llm_bridge.py           # 78 lines  — model abstraction layer
│   ├── llm_client.py           # 109 lines — low-level API client
│   ├── agent_memory.py         # 92 lines  — episodic memory w/ scored retrieval
│   ├── memory.py               # 383 lines — original JARVIS note storage
│   ├── agents.py               # 108 lines — 5 agent functions
│   ├── prompts.py              # 129 lines — agent system prompts
│   ├── debate.py               # 72 lines  — sequential debate loop
│   ├── debate_async.py         # 301 lines — async debate (primary runner)
│   ├── run_experiment.py       # 94 lines  — experiment entry point
│   ├── run_single_agent.py     # 120 lines — single agent baseline
│   ├── run_control.py          # 147 lines — control condition runner
│   ├── scorer.py               # 68 lines  — 3-component quality scorer
│   ├── experiment_logger.py    # 31 lines  — CSV logger
│   ├── main.py                 # 447 lines — JARVIS CLI
│   ├── research_mode.py        # 155 lines — research CLI commands
│   ├── config.py               # 100 lines — config/env loading
│   └── colors.py               # 60 lines  — terminal colors
├── experiments/
│   ├── baseline/               # GPT-2 zero-shot baseline
│   └── debate/                 # Proposer→Critic→Refiner experiments
└── .env.template               # Originally Ollama; now Anthropic

## Completed Research (informs Legion architecture)
- Exp 0: GPT-2 zero-shot → 0% (generates shape, not logic)
- Exp 1: Two-agent debate → 53% (debate works w/ hint injection)
- Exp 2: Planner+Proposer+Skeptic+Refiner+Memory → [PENDING — file built, not yet run]
- JARVIS paper findings:
  - Evaluative agents (Skeptic, Ethicist) self-correct via memory retrieval unprompted
  - Procedural agents (Planner, Engineer) collapse without attribution scaffolding
  - Standard coverage metrics are BROKEN for multi-agent eval — use task pass rate

## Key Architecture Decisions Already Made
- bootstrap_beliefs() seeds wm with atomic beliefs at startup (before first tick)
- _save_unlocked() is the internal save path for write methods that hold self._lock
- Goal scoping matters: decompose analysis goals from design goals before dispatch
- can_handle() keyword match is a live constraint 
— child goal descriptions must
contain node capability keywords or dispatch is skipped silently

## The Legion Vision
Transform from: reactive debate system (you prompt it, it responds, stops)
Into: proactive hivemind (collective sets own goals, decomposes, assigns, pursues)

Missing pieces to build:
1. Goal stack — collective decides what to work on
2. Autonomous task dispatch — agents pull work, don't wait to be called
3. Shared world model — persistent state all nodes read/write
4. Consensus mechanism — how the collective resolves disagreement and commits
5. Model-agnostic abstraction — already partially exists in llm_bridge.py

## Long-term Constraints
- API cost: fine now ($24.53 of $25 remaining), unsustainable when always-on
- Migration path: Anthropic (now) → Ollama + local model (when hardware allows)
- Target local model: ~7B param quantized on M1, or larger on future workstation
- Fine-tuning loop (agents learn from own experience): Phase 3+ effort

## Session Protocol
- Paste this doc at the start of each new session
- Share relevant src/ files as needed (llm_bridge, agents, memory are usually relevant)
- Flag context window when responses get very long or 15+ substantial exchanges
- Keep kickoff doc updated when architecture decisions are made

## Files to Share When Resuming Architecture Work
```bash
dump_context.sh > session_context.txt

## Current Session Status
world_model.py    ✓ built + patched (_save_unlocked fix applied)
goal_stack.py     ✓ built
dispatcher.py     ✓ built
consensus.py      ✓ built
run_loop.py       ✓ built
bootstrap_beliefs ✓ built and confirmed working (run 2)