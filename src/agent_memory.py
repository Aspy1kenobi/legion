"""
Per-agent memory stream for Phase 2 debate experiments.
Implements the Generative Agents retrieval formula:
    score = recency + importance + relevance
All three components normalized to [0, 1] before summing.

Intentionally minimal — no disk I/O, no tags, no export.
Each AgentMemory instance lives for one experiment run only.
"""

import re
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0


class AgentMemory:
    """
    Ephemeral memory stream for a single debate agent.
    Accumulates responses across rounds and retrieves relevant
    entries using recency + importance + relevance scoring.
    """

    def __init__(self, agent_name: str, decay: float = 0.01):
        self.agent_name = agent_name
        self.decay = decay
        self.entries: list[MemoryEntry] = []

    def add(self, text: str, importance: float = 1.0) -> None:
        """Add a new memory entry."""
        self.entries.append(MemoryEntry(text=text, importance=importance))

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        """
        Return top-k entries scored by recency + importance + relevance.
        Returns empty list if memory is empty or top_k is 0.
        """
        if not self.entries or top_k == 0:
            return []

        n = len(self.entries)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        # ── 1. Recency ────────────────────────────────────────────
        now = datetime.now()
        hours_elapsed = np.array([
            (now - e.timestamp).total_seconds() / 3600
            for e in self.entries
        ])
        recency_scores = np.exp(-self.decay * hours_elapsed)

        # ── 2. Importance ─────────────────────────────────────────
        importance_scores = np.array([e.importance for e in self.entries])
        max_imp = importance_scores.max()
        if max_imp > 0:
            importance_scores = importance_scores / max_imp

        # ── 3. Relevance ──────────────────────────────────────────
        raw_relevance = np.array([
            len(set(re.findall(r'\b\w+\b', e.text.lower())) & query_words)
            for e in self.entries
        ])
        max_rel = raw_relevance.max()
        relevance_scores = raw_relevance / max_rel if max_rel > 0 else np.zeros(n)

        # ── 4. Combine and rank ───────────────────────────────────
        combined = recency_scores + importance_scores + relevance_scores
        top_indices = combined.argsort()[::-1][:top_k]

        return [self.entries[i] for i in top_indices]

    def build_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve top-k memories and format them as a context string
        for injection into the agent's prompt.
        Returns empty string if memory is empty.
        """
        memories = self.retrieve(query, top_k)
        if not memories:
            return ""

        lines = [f"[Memory {i+1}] {m.text}" for i, m in enumerate(memories)]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)