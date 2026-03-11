
# Legion

A proactive multi-agent hivemind built on the JARVIS codebase.
Agents set their own goals, decompose them, assign work, and pursue it autonomously.

## Structure
- `src/` — inherited JARVIS modules (llm_bridge, llm_client, agent_memory, etc.)
- `src/legion/` — Legion-native modules (world_model, goal_stack, dispatcher, consensus, run_loop)
- `data/` — world model persistence (gitignored)

## Run
```bash
cd src/legion
python run_loop.py
```

## Status
Phase 1 build complete. First live run pending.
