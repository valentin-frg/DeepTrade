from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SENTINELLE_CACHE_PATH = Path("sentinelle_cache.json")
CYCLE_HISTORY_PATH = Path("cycle_history.json")

SENTINELLE_SYSTEM_PROMPT = """You are the Chief Risk Officer (CRO), the risk audit AI of an elite quantitative hedge fund. Your role is critical: you are the capital's last line of defense.

Every 5 minutes, you receive a structured situation report containing: (1) the current Macro Strategy, (2) the Execution Agent's recent performance history, (3) live market data (price, indicators), (4) the latest news sentiment, and (5) the real-time account state — including open positions, their entry prices, unrealized PnL, and current mark prices. Use section 5 to assess whether any open positions are already in severe drawdown, which is a key signal for a Tactical Hemorrhage alarm.

YOUR MISSION:
Decide whether the current context invalidates the ongoing macroeconomic strategy. If so, you must trigger the alarm to force the Trader to urgently recalculate a new strategy.

INTERVENTION RULES (Only trigger the alarm in these cases):
1. Fundamental Shock (News): A major policy, economic, or geopolitical event has just emerged (e.g., SEC announcement, exchange bankruptcy, influential presidential statement) that directly contradicts the current strategy.
2. Tactical Hemorrhage (Drawdown): The Execution Agent is accumulating abnormal losses (repeated Stop Loss hits), proving that the current bias is disconnected from actual order flow reality.
3. Black Swan / Price Anomaly: Price or volume undergoes an extreme deviation outside normal statistical bounds (Flash Crash or sudden Pump).

WHAT YOU MUST NOT DO:
- Do NOT panic in the face of normal market "noise" (minor fluctuations under 1%).
- You do not place trading orders yourself.

MANDATORY RESPONSE FORMAT:
You must respond ONLY with a valid JSON object, without markdown fences (no ```json), and without any text before or after.
Required structure:
{
  "trigger_recalculation": true | false,
  "confidence_in_alarm": [integer from 0 to 100],
  "reasoning_for_logs": "Short, technical explanation of your decision for our internal logs. If false, explain why the situation is normal.",
  "message_to_macro_strategist": "If trigger_recalculation is true, write an urgent message to the Trader explaining what has changed and what must be factored into the recalculation. If false, set to null."
}"""


def _load_cycle_history() -> List[Dict[str, Any]]:
    """Load the last 10 cycles from disk."""
    if not CYCLE_HISTORY_PATH.exists():
        return []
    try:
        with CYCLE_HISTORY_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data[-10:]
    except Exception:  # noqa: BLE001
        pass
    return []


def _summarise_cycles(cycles: List[Dict[str, Any]]) -> str:
    """Produce a compact, readable summary of the last N cycles for the Sentinelle."""
    if not cycles:
        return "No execution cycle data available."

    lines = []
    for i, c in enumerate(cycles, 1):
        ts = c.get("run_timestamp", "?")
        inv = c.get("invocation_count", "?")
        bal_before = (c.get("balances_before") or {}).get("total_balance", "?")
        bal_after = (c.get("balances_after") or {}).get("total_balance", "?")
        pnl = (c.get("balances_after") or {}).get("pnl_percent", None)
        decisions = c.get("decisions") or {}

        signals = []
        for coin, payload in decisions.items():
            sig = (payload.get("trade_signal_args") or {}).get("signal", "?")
            signals.append(f"{coin}:{sig}")

        pnl_str = f"{pnl:.2f}%" if pnl is not None else "?"
        lines.append(
            f"[Cycle #{inv} | {ts}] Balance: {bal_before} → {bal_after} | Total PnL: {pnl_str} | "
            f"Signals: {', '.join(signals) if signals else 'none'}"
        )
    return "\n".join(lines)


def build_sentinelle_prompt(
    sentiment: str,
    macro_strategy: str,
    market_data: str,
    cycles: Optional[List[Dict[str, Any]]] = None,
    account_prompt: Optional[str] = None,
) -> str:
    """Assemble the user prompt sent to the Sentinelle."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cycle_summary = _summarise_cycles(cycles or [])

    account_section = (
        f"5. LIVE POSITIONS & ACCOUNT STATE (real-time):\n"
        f"{account_prompt.strip()}\n\n"
    ) if account_prompt else ""

    return (
        f"SITUATION REPORT — RISK AUDIT (5-MINUTE CYCLE)\n"
        f"Current Time: {now}\n\n"
        f"1. CURRENT MACRO STRATEGY (To Audit):\n"
        f"{macro_strategy.strip() if macro_strategy else 'No macro strategy available.'}\n\n"
        f"2. EXECUTION AGENT PERFORMANCE (Since strategy start):\n"
        f"{cycle_summary}\n\n"
        f"3. CURRENT MARKET DATA:\n"
        f"{market_data.strip()}\n\n"
        f"4. LIVE NEWS FEED (Urgency / Sentiment):\n"
        f"{sentiment.strip() if sentiment else 'No sentiment data available.'}\n\n"
        f"{account_section}"
        f"ANALYSIS REQUIRED:\n"
        f"Based on this data, is the current macro strategy still safe and consistent with market reality?\n"
        f"Generate your response immediately in strict JSON format."
    )


def run_sentinelle(
    market_data: str,
    sentiment: str,
    macro_strategy: str,
    account_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call the Sentinelle (CRO) AI with the latest market context.
    Returns the parsed JSON decision dict, or an error dict on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"trigger_recalculation": False, "reasoning_for_logs": "GEMINI_API_KEY not set."}

    cycles = _load_cycle_history()
    user_prompt = build_sentinelle_prompt(sentiment, macro_strategy, market_data, cycles, account_prompt)

    try:
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=SENTINELLE_SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        )
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=contents,
            config=config,
        )
        raw = (response.text or "").strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        decision = json.loads(raw)

        # Persist the Sentinelle's decision for the dashboard
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "user_prompt": user_prompt,
        }
        temp_path = SENTINELLE_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        temp_path.replace(SENTINELLE_CACHE_PATH)

        print("\n===== SENTINELLE (CRO) =====")
        triggered = decision.get("trigger_recalculation", False)
        confidence = decision.get("confidence_in_alarm", 0)
        reasoning = decision.get("reasoning_for_logs", "")
        print(f"Trigger recalculation: {triggered} | Confidence: {confidence}/100")
        print(f"Reasoning: {reasoning}")

        return decision

    except json.JSONDecodeError as exc:
        print(f"\n===== SENTINELLE ERROR =====\nJSON parse failed: {exc}")
        return {"trigger_recalculation": False, "reasoning_for_logs": f"JSON parse error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        print(f"\n===== SENTINELLE ERROR =====\n{exc}")
        return {"trigger_recalculation": False, "reasoning_for_logs": f"Error: {exc}"}


def get_cached_sentinelle() -> Optional[Dict[str, Any]]:
    """Return the latest Sentinelle decision from cache, or None."""
    if not SENTINELLE_CACHE_PATH.exists():
        return None
    try:
        with SENTINELLE_CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None
