#!/bin/bash
# dump_context.sh
# Run at the start of each session: bash dump_context.sh > session_context.txt
# Then paste session_context.txt into Claude.

export COLUMNS=200

echo "=== REPO STRUCTURE ==="
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | sort
echo ""

echo "=== KEY CONFIG ==="
cat .env.template 2>/dev/null || echo "(not found)"
echo ""

for f in $(find . -name "*.py" -not -path "./.git/*" -not -path "*__pycache__*" | sort); do
  echo "=== $f ==="
  cat "$f"
  echo ""
done