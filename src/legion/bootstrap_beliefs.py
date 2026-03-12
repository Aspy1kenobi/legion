async def bootstrap_beliefs(wm: "SharedWorldModel") -> int:
    """
    Seed the world model with Legion's known architecture, experimental
    findings, constraints, and gaps before the first tick.

    Called once during RunLoop.startup() after wm.load(). Safe to call
    on subsequent runs — add_belief() updates existing beliefs in place,
    so re-running does not duplicate entries.

    Returns the number of beliefs written.
    """
    beliefs = [

        # ── Build status ──────────────────────────────────────────────────────
        ("module_world_model_built",
         "world_model.py is built and operational.",
         1.0),

        ("module_goal_stack_built",
         "goal_stack.py is built and operational.",
         1.0),

        ("module_dispatcher_built",
         "dispatcher.py is built and operational.",
         1.0),

        ("module_consensus_built",
         "consensus.py is built and operational.",
         1.0),

        ("module_run_loop_built",
         "run_loop.py is built and operational.",
         1.0),

        ("module_goal_stack_pending",
         "goal_stack.py has no LLM-driven decomposition. "
         "Callers provide subgoal descriptions manually.",
         1.0),

        # ── Experimental findings ─────────────────────────────────────────────
        ("exp0_result",
         "GPT-2 zero-shot scored 0%. It generates shape, not logic.",
         1.0),

        ("exp1_result",
         "Two-agent debate scored 53% with hint injection.",
         1.0),

        ("exp2_result",
         "Exp 2 full pipeline was built but has not been run.",
         1.0),

        ("finding_evaluative_selfcorrect",
         "Evaluative agents (Skeptic, Ethicist) self-correct via memory "
         "retrieval unprompted.",
         1.0),

        ("finding_procedural_collapse",
         "Procedural agents (Planner, Engineer) collapse without "
         "attribution scaffolding.",
         1.0),

        ("finding_coverage_metrics_broken",
         "Standard coverage metrics are broken for multi-agent eval. "
         "Task pass rate is the correct measure.",
         1.0),

        ("finding_debate_viable",
         "Debate architecture is viable as a commit protocol.",
         1.0),

        # ── Architecture decisions ────────────────────────────────────────────
        ("arch_async_first",
         "debate_async.py is the primary runner pattern. "
         "Async-first is the established approach.",
         1.0),

        ("arch_memory_dual_layer",
         "Memory is dual-layer: memory.py for persistent notes, "
         "agent_memory.py for episodic retrieval.",
         1.0),

        ("arch_role_differentiation",
         "Procedural vs evaluative role distinction changes agent behavior "
         "functionally, not cosmetically.",
         1.0),

        ("arch_identity_constitutive",
         "Role identity must be present from first output. "
         "Adding it later degrades behavior.",
         1.0),

        ("arch_model_agnostic",
         "call_llm() routes to Anthropic or Ollama based on "
         "ANTHROPIC_API_KEY presence in config.",
         1.0),

        ("arch_consensus_challenge_accept",
         "Consensus uses challenge/accept: one node produces output, "
         "Skeptic gets one veto round.",
         1.0),

        ("arch_sequential_decomposition",
         "GoalStack.decompose() chains children sequentially by default. "
         "Parallel decomposition requires explicit depends_on.",
         1.0),

        # ── Hardware and cost constraints ─────────────────────────────────────
        ("constraint_hardware",
         "Primary hardware is M1 MacBook Air 8GB. "
         "No local training is possible on this hardware.",
         1.0),

        ("constraint_api_cost",
         "Anthropic API is the current model provider. "
         "Cost is a real constraint on always-on operation.",
         1.0),

        ("constraint_migration_target",
         "Target migration path is Anthropic now, "
         "Ollama plus local model when hardware allows.",
         1.0),

        # ── Known gaps ────────────────────────────────────────────────────────
        ("gap_llm_decomposition",
         "No node currently calls an LLM to decompose goals. "
         "Planner decomposition is not yet autonomous.",
         1.0),

        ("gap_can_handle_keyword",
         "LegionNode.can_handle() uses keyword matching. "
         "Subgoal descriptions must contain capability keywords "
         "or dispatch is skipped.",
         1.0),

        ("gap_engineer_node_missing",
         "No Engineer node exists. Goals requiring implementation, "
         "coding, or testing have no capable node.",
         1.0),

        ("gap_ethicist_node_missing",
         "No Ethicist node exists. "
         "Consensus evaluator pool has only Skeptic.",
         1.0),
    ]

    print("[bootstrap] starting", flush=True)

    for i, (belief_id, content, confidence) in enumerate(beliefs):
        print(f"[bootstrap] writing {i}: {belief_id}", flush=True)
        await wm.add_belief(
            belief_id=belief_id,
            content=content,
            confidence=confidence,
            source="bootstrap",
            tags=["bootstrap"],
        )

    print(f"[bootstrap] done, wrote {len(beliefs)}", flush=True)
    return len(beliefs)