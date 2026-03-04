#!/bin/zsh
# DeepTrade auto-restart launcher with crash recovery.
# Usage: ./run_bot.sh
# To stop permanently: use Kill Switch on the dashboard, then Ctrl+C here.

cd "$(dirname "$0")"
LOGFILE="deeptrade.log"

close_positions() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Closing all open positions for safety..." | tee -a "$LOGFILE"
    uv run python - <<'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys
sys.path.insert(0, '.')
try:
    from prompt_builder import build_exchange, fetch_account_position_map, close_position
    ex = build_exchange()
    positions = fetch_account_position_map(ex)
    if not positions:
        print("  No open positions found.")
    else:
        for coin, pos in positions.items():
            try:
                close_position(ex, pos['symbol'], pos['contracts'])
                print(f"  ✅ Closed {coin}")
            except Exception as e:
                print(f"  ❌ Failed to close {coin}: {e}")
except Exception as e:
    print(f"  ❌ Exchange connection error: {e}")
PYEOF
}

echo "=============================================="  | tee -a "$LOGFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 DeepTrade launcher started" | tee -a "$LOGFILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dashboard: http://127.0.0.1:8050"  | tee -a "$LOGFILE"
echo "=============================================="  | tee -a "$LOGFILE"

while true; do
    # Remove any stale stop flag from previous run
    rm -f .stop_requested

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dash_app.py..." | tee -a "$LOGFILE"
    uv run python dash_app.py 2>&1 | tee -a "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Bot exited with code $EXIT_CODE" | tee -a "$LOGFILE"

    # Check if Kill Switch requested a clean stop
    if [ -f .stop_requested ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏹  Clean stop requested (Kill Switch used). Positions already closed." | tee -a "$LOGFILE"
        rm -f .stop_requested
        break
    fi

    # Crash detected: close positions then restart
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 💥 Crash detected!" | tee -a "$LOGFILE"
        close_positions
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in 15 seconds... (Ctrl+C to abort)" | tee -a "$LOGFILE"
        sleep 15
    else
        # Clean exit (Ctrl+C by user) — ask before restarting
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Clean exit. Restarting in 15 seconds... (Ctrl+C to stop)" | tee -a "$LOGFILE"
        sleep 15
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DeepTrade launcher stopped." | tee -a "$LOGFILE"
