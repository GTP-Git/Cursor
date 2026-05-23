#!/bin/bash
# Run all tracked cruise price checks (for cron / launchd).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/check_prices.log"
LOCK_DIR="$LOG_DIR/check_prices.lockdir"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") Skipped — previous check still running." >>"$LOG_FILE"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

{
  echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
  source "$ROOT/.venv/bin/activate"
  python "$ROOT/scraper.py"
} >>"$LOG_FILE" 2>&1
