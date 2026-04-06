"""
Legion fidelity inspection script.
Run from src/legion/ after any run to check three health signals:

    python inspect_fidelity.py [path/to/world_model.json]

Signals checked:
  1. Planner JSON parse fidelity  — parse_error events from planner (target: 0)
  2. Follow-on goal push          — goals with source="planner" (target: ≥1)
  3. Consensus parse fidelity     — beliefs with confidence=0.3 from skeptic (target: 0)
                                    (0.3 is the malformed-verdict fallback in ConsensusEngine)
"""
import json
import sys
import os

wm_path = sys.argv[1] if len(sys.argv) > 1 else "data/world_model.json"

if not os.path.exists(wm_path):
    print(f"ERROR: world model not found at {wm_path}")
    sys.exit(1)

wm = json.load(open(wm_path))

# ── Signal 1: Planner JSON parse errors ──────────────────────────────────────
parse_errors = [
    e for e in wm["events"]
    if e["agent"] == "planner" and e["event_type"] == "parse_error"
]
sig1_pass = len(parse_errors) == 0
print(f"[{'PASS' if sig1_pass else 'FAIL'}] Signal 1: Planner parse_error events = {len(parse_errors)} (target: 0)")
if parse_errors:
    print(f"  First failure raw output:\n{parse_errors[0]['content'][:500]}")

# ── Signal 2: Follow-on goals pushed by planner ───────────────────────────────
planner_followons = [g for g in wm["goals"].values() if g.get("source") == "planner"]
sig2_pass = len(planner_followons) >= 1
print(f"[{'PASS' if sig2_pass else 'FAIL'}] Signal 2: Goals with source=planner = {len(planner_followons)} (target: ≥1 if any planner tick ran)")
for g in planner_followons[:5]:
    print(f"  - [{g['status']}] {g['description'][:80]}")

# ── Signal 3: Consensus malformed-verdict beliefs ─────────────────────────────
# confidence=0.3 is the ConsensusEngine fallback for unparseable evaluator responses
low_conf = [
    b for b in wm["beliefs"].values()
    if b.get("confidence") == 0.3 and "skeptic" in b.get("source", "")
]
sig3_pass = len(low_conf) == 0
print(f"[{'PASS' if sig3_pass else 'FAIL'}] Signal 3: Malformed-verdict beliefs (confidence=0.3, source~skeptic) = {len(low_conf)} (target: 0)")
if low_conf:
    for b in low_conf:
        print(f"  - {b['id']}: {b['content'][:120]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
overall = sig1_pass and sig2_pass and sig3_pass
print(f"Overall: {'ALL PASS' if overall else 'FAILURES DETECTED'}")
sys.exit(0 if overall else 1)
