from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SENTIMENT_CACHE_PATH = Path("sentiment_snapshot.json")
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]


def _symbol_to_coin(symbol: str) -> str:
    return symbol.split("/")[0]


def fetch_and_cache_sentiment(coins: List[str] = COINS) -> None:
    """
    Background job that fetches qualitative market data via Gemini and stores it on disk.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    try:
        client = genai.Client(api_key=api_key)

        coin_names = ", ".join(_symbol_to_coin(symbol) for symbol in coins)
        prompt = (
            f"Using Google Search, analyze the last 4 hours of market-moving information for: {coin_names}. "
            "Include: "
            "1. Overall crypto sentiment (Bullish/Bearish/Neutral). "
            "2. Top 3 recent headlines most likely to move prices in the next few hours. "
            "3. Any macro triggers: USD index (DXY), equity futures (S&P 500/Nasdaq), Fed or CPI news. "
            "4. Any crypto-specific catalyst: Bitcoin ETF flows, regulatory news, major liquidations or whale activity. "
            "Be concise (max 250 words). Prioritize recency — focus on the last 4 hours."
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=contents,
            config=config,
        )
        sentiment_text = (response.text or "").strip()
        if not sentiment_text:
            sentiment_text = "No sentiment content returned."

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content": sentiment_text,
            "status": "fresh",
        }

        temp_path = SENTIMENT_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        temp_path.replace(SENTIMENT_CACHE_PATH)

        print("\n===== SENTIMENT =====")
        print("Gemini cache updated successfully.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n===== ERROR =====\nSentiment fetch failed: {exc}")


def get_cached_sentiment() -> str:
    """
    Returns formatted qualitative intelligence if we have a recent snapshot.
    """
    if not SENTIMENT_CACHE_PATH.exists():
        return ""

    try:
        with SENTIMENT_CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        timestamp = data.get("timestamp")
        content = data.get("content", "")
        if not timestamp or not content:
            return ""

        ts = datetime.fromisoformat(timestamp)
        age_minutes = (datetime.now(timezone.utc) - ts).total_seconds() / 60

        label = "Latest News"
        if age_minutes > 60:
            label = "Old News (Context Only)"

        return (
            f"\n### QUALITATIVE INTELLIGENCE ({label} - {int(age_minutes)}m ago)\n"
            f"{content}\n"
        )
    except Exception:  # noqa: BLE001
        return ""
