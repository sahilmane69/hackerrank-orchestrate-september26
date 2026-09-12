"""Offline LLM evidence extraction: 16 images + all messages.

Run from project root:
    GEMINI_API_KEY=... .venv/bin/python code/extraction/run_extraction.py

Writes:
    code/extraction/image_amounts.json    - amount per blank-amount event (via linked image)
    code/extraction/message_facts.json    - structured amendment facts per user
    code/extraction/usage_log.json        - call/token accounting for usage_report.md

The deterministic pipeline (code/run_full.py) only reads the JSON artifacts;
it never calls the API. Safe to re-run: results are written after every call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from extraction_config import (  # noqa: E402
    DATASET_DIR,
    EXTRACTION_DIR,
    IMAGE_PROMPT,
    MEDIA_DIR,
    MESSAGE_SCHEMA,
    MODEL_NAME,
    message_prompt,
    write_json,
)

IMAGES_CSV = DATASET_DIR / "images.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
EVENTS_CSV = DATASET_DIR / "financial_events.csv"
REQUESTS_CSV = DATASET_DIR / "requests.csv"
PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"

IMAGE_AMOUNTS_PATH = EXTRACTION_DIR / "image_amounts.json"
MESSAGE_FACTS_PATH = EXTRACTION_DIR / "message_facts.json"
USAGE_LOG_PATH = EXTRACTION_DIR / "usage_log.json"

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5


def load_model() -> Any:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set in the environment.", file=sys.stderr)
        raise SystemExit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def parse_json_block(text: str) -> Any:
    """Extract JSON from a model reply that may wrap it in prose or fences."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min(
            (i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0),
            default=-1,
        )
        if start >= 0:
            return json.loads(cleaned[start : cleaned.rfind("}") + 1])
        raise


def call_with_retry(model: Any, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.time()
            response = model.generate_content(**kwargs)
            elapsed = time.time() - start
            usage = getattr(response, "usage_metadata", None)
            usage_dict = {}
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                    "candidates_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
                }
            return response.text, usage_dict, elapsed
        except Exception as exc:  # network, quota, parsing of response object
            last_error = exc
            print(f"  attempt {attempt} failed: {exc}; retrying in {RETRY_WAIT_SECONDS}s")
            time.sleep(RETRY_WAIT_SECONDS)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def extract_images(model: Any, usage: dict[str, Any]) -> None:
    images_df = pd.read_csv(IMAGES_CSV)
    events_df = pd.read_csv(EVENTS_CSV)
    blank_event_ids = set(
        events_df[events_df["amount"].isna()]["event_id"].astype(str)
    )

    results: dict[str, dict[str, Any]] = {}
    if IMAGE_AMOUNTS_PATH.exists():
        results = json.loads(IMAGE_AMOUNTS_PATH.read_text(encoding="utf-8"))

    generation_config = {"response_mime_type": "application/json", "temperature": 0}

    for row in images_df.itertuples(index=False):
        image_id = str(row.image_id)
        linked_event = str(row.related_event_id) if pd.notna(row.related_event_id) else None
        if image_id in results:
            continue

        png_path = MEDIA_DIR / f"{image_id}.png"
        if not png_path.exists():
            results[image_id] = {
                "doc_type": "other",
                "amount": None,
                "currency": None,
                "date": None,
                "recurring": None,
                "linked_event_id": linked_event,
                "event_linkage_note": "image file missing",
            }
            write_json(IMAGE_AMOUNTS_PATH, results)
            continue

        from PIL import Image

        image = Image.open(png_path)
        prompt = (
            f"{IMAGE_PROMPT}\n\nThe image is linked to financial event id: "
            f"{linked_event or 'unknown'}."
        )
        print(f"[image] {image_id} (event {linked_event})...")
        text, usage_dict, elapsed = call_with_retry(
            model,
            content=[prompt, image],
            generation_config=generation_config,
        )
        record_usage(usage, "image", image_id, usage_dict, elapsed)
        try:
            parsed = parse_json_block(text)
            parsed["image_id"] = image_id
            parsed["linked_event_id"] = linked_event
            results[image_id] = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  !! could not parse reply for {image_id}: {exc}")
            results[image_id] = {
                "doc_type": "other",
                "amount": None,
                "currency": None,
                "date": None,
                "recurring": None,
                "linked_event_id": linked_event,
                "event_linkage_note": f"parse failure: {text[:200]}",
            }
        write_json(IMAGE_AMOUNTS_PATH, results)

    print(f"Image extraction complete: {len(results)} records -> {IMAGE_AMOUNTS_PATH.name}")


def build_message_context(
    messages_df: pd.DataFrame,
    events_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    requests_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Group messages per user with the event/request facts they may amend."""
    per_user: dict[str, dict[str, Any]] = {}
    for user_id, group in messages_df.groupby("user_id"):
        user_events = events_df[events_df["user_id"] == user_id]
        event_rows = [
            {
                "event_id": str(e.event_id),
                "event_type": str(e.event_type),
                "category": str(e.category),
                "direction": str(e.direction),
                "amount": None if pd.isna(e.amount) else float(e.amount),
                "currency": None if pd.isna(e.currency) else str(e.currency),
                "event_date": None if pd.isna(e.event_date) else str(e.event_date),
                "settlement_date": (
                    None if pd.isna(e.settlement_date) else str(e.settlement_date)
                ),
                "status": str(e.status),
            }
            for e in user_events.itertuples(index=False)
        ]
        user_requests = requests_df[requests_df["user_id"] == user_id]
        profile = profiles_df[profiles_df["user_id"] == user_id]
        per_user[str(user_id)] = {
            "messages": [
                {
                    "message_id": str(m.message_id),
                    "request_id": None if pd.isna(m.request_id) else str(m.request_id),
                    "related_event_id": (
                        None if pd.isna(m.related_event_id) else str(m.related_event_id)
                    ),
                    "sent_at": str(m.sent_at),
                    "source_type": str(m.source_type),
                    "text": str(m.message_text),
                }
                for m in group.itertuples(index=False)
            ],
            "events": event_rows,
            "requests": [
                {
                    "request_id": str(r.request_id),
                    "request_date": str(r.request_date),
                    "requested_amount": float(r.requested_amount),
                    "desired_completion_date": str(r.desired_completion_date),
                }
                for r in user_requests.itertuples(index=False)
            ],
            "home_currency": (
                str(profile.iloc[0]["home_currency"]) if not profile.empty else None
            ),
        }
    return per_user


def extract_messages(model: Any, usage: dict[str, Any]) -> None:
    messages_df = pd.read_csv(MESSAGES_CSV)
    events_df = pd.read_csv(EVENTS_CSV)
    profiles_df = pd.read_csv(PROFILES_CSV)
    requests_df = pd.read_csv(REQUESTS_CSV)

    context = build_message_context(messages_df, events_df, profiles_df, requests_df)

    results: dict[str, Any] = {}
    if MESSAGE_FACTS_PATH.exists():
        results = json.loads(MESSAGE_FACTS_PATH.read_text(encoding="utf-8"))

    generation_config = {"response_mime_type": "application/json", "temperature": 0}

    for user_id, bundle in context.items():
        if user_id in results:
            continue
        prompt = message_prompt(json.dumps(bundle, ensure_ascii=False))
        print(f"[messages] {user_id}: {len(bundle['messages'])} messages, "
              f"{len(bundle['events'])} events in context...")
        text, usage_dict, elapsed = call_with_retry(
            model, content=prompt, generation_config=generation_config
        )
        record_usage(usage, "message", user_id, usage_dict, elapsed)
        try:
            parsed = parse_json_block(text)
            results[user_id] = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  !! could not parse reply for {user_id}: {exc}")
            results[user_id] = [
                {
                    "kind": "none",
                    "amount": None,
                    "currency": None,
                    "effective_date": None,
                    "target_event_id": None,
                    "recurring": None,
                    "note": f"parse failure: {text[:200]}",
                }
            ]
        write_json(MESSAGE_FACTS_PATH, results)

    print(f"Message extraction complete: {len(results)} users -> {MESSAGE_FACTS_PATH.name}")


def record_usage(
    usage: dict[str, Any],
    phase: str,
    subject: str,
    usage_dict: dict[str, int],
    elapsed: float,
) -> None:
    usage["calls"].append(
        {
            "phase": phase,
            "subject": subject,
            "model": MODEL_NAME,
            **usage_dict,
            "elapsed_seconds": round(elapsed, 2),
        }
    )
    usage["totals"] = {
        "calls": len(usage["calls"]),
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in usage["calls"]),
        "candidates_tokens": sum(c.get("candidates_tokens", 0) for c in usage["calls"]),
        "total_tokens": sum(c.get("total_tokens", 0) for c in usage["calls"]),
    }
    write_json(USAGE_LOG_PATH, usage)


def main() -> None:
    EXTRACTION_DIR.mkdir(exist_ok=True)
    model = load_model()

    usage_path = USAGE_LOG_PATH
    usage = {"model": MODEL_NAME, "calls": [], "totals": {}}
    if usage_path.exists():
        existing = json.loads(usage_path.read_text(encoding="utf-8"))
        if existing.get("model") == MODEL_NAME:
            usage = existing
            usage.setdefault("calls", [])
            usage["totals"] = {
                "calls": len(usage["calls"]),
                "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in usage["calls"]),
                "candidates_tokens": sum(c.get("candidates_tokens", 0) for c in usage["calls"]),
                "total_tokens": sum(c.get("total_tokens", 0) for c in usage["calls"]),
            }

    extract_images(model, usage)
    extract_messages(model, usage)
    print(f"Usage totals: {usage['totals']}")


if __name__ == "__main__":
    main()
