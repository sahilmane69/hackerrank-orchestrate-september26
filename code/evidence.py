"""Apply extracted evidence (images + messages) to per-user event data.

Reads the JSON artifacts written by extraction/run_extraction.py:
    code/extraction/image_amounts.json
    code/extraction/message_facts.json

Rules implemented (mirroring the problem statement):
- Blank-amount events take their amount from the linked image. Never zero.
- Salary changes from employer messages replace the projected salary-stream
  amount from the effective date (returned as a salary_overrides mapping).
- Explicit event cancellations/delays/settlements/amendments targeting a
  specific event id adjust that row before forecasting.
- Everything stays deterministic: this module never calls an API.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = CODE_DIR / "extraction"
IMAGE_AMOUNTS_PATH = EXTRACTION_DIR / "image_amounts.json"
MESSAGE_FACTS_PATH = EXTRACTION_DIR / "message_facts.json"


def _parse_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def load_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    image_amounts: dict[str, Any] = {}
    message_facts: dict[str, Any] = {}
    if IMAGE_AMOUNTS_PATH.exists():
        image_amounts = json.loads(IMAGE_AMOUNTS_PATH.read_text(encoding="utf-8"))
    if MESSAGE_FACTS_PATH.exists():
        message_facts = json.loads(MESSAGE_FACTS_PATH.read_text(encoding="utf-8"))
    return image_amounts, message_facts


def image_amount_by_event(image_amounts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map linked_event_id -> extracted image fact."""
    by_event: dict[str, dict[str, Any]] = {}
    for image_id, fact in image_amounts.items():
        linked = fact.get("linked_event_id")
        if linked:
            by_event[str(linked)] = {**fact, "image_id": image_id}
    return by_event


def apply_image_amounts(
    events: pd.DataFrame,
    image_amounts: dict[str, Any],
    skipped: list[str],
) -> pd.DataFrame:
    """Fill blank amounts from linked images. Returns a copy when changes apply."""
    if events.empty or not image_amounts:
        return events

    by_event = image_amount_by_event(image_amounts)
    filled = events.copy()
    changed = False
    for idx, row in filled.iterrows():
        if not pd.isna(row.get("amount")):
            continue
        fact = by_event.get(str(row["event_id"]))
        if fact is None or fact.get("amount") is None:
            skipped.append(
                f"{row['event_id']}: blank amount and no usable image amount"
            )
            continue
        amount = float(fact["amount"])
        currency = fact.get("currency")
        event_date = _parse_date(row.get("event_date")) or date.min
        effective = _parse_date(fact.get("date"))
        # If the image states the amount only takes effect later, do not count
        # the old-style value today; the row keeps its (late) effective date via
        # settlement handling in the forecast. Here we just fill the amount.
        filled.at[idx, "amount"] = amount
        if currency and isinstance(currency, str) and currency.strip():
            filled.at[idx, "currency"] = currency.strip()
        if effective is not None and event_date != date.min and effective > event_date:
            # Push the cash date forward to the effective date when later.
            filled.at[idx, "event_date"] = effective.isoformat()
            if pd.isna(filled.at[idx, "settlement_date"]):
                filled.at[idx, "settlement_date"] = effective.isoformat()
        changed = True
        skipped.append(
            f"{row['event_id']}: amount {amount} filled from {fact.get('image_id')}"
        )
    return filled if changed else events


def salary_overrides_for_user(
    user_id: str,
    message_facts: dict[str, Any],
) -> dict[str, Any]:
    """Extract the newest salary-change fact for a user, if any."""
    facts = message_facts.get(user_id)
    if not facts:
        return {}
    if isinstance(facts, dict):
        facts = [facts]
    overrides: dict[str, Any] = {}
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("kind") != "salary_change":
            continue
        amount = fact.get("amount")
        if amount is None:
            continue
        effective = _parse_date(fact.get("effective_date"))
        current = overrides.get("amount")
        # Prefer the fact with the latest effective date; ties -> later in list.
        if current is None or (effective is not None and effective >= (overrides.get("effective") or date.min)):
            overrides = {
                "amount": float(amount),
                "currency": fact.get("currency"),
                "effective": effective,
                "recurring": fact.get("recurring"),
                "note": fact.get("note", ""),
            }
    return overrides


def event_overrides_for_user(
    user_id: str,
    message_facts: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map target_event_id -> amendment fact (cancel/delay/amend/settle)."""
    facts = message_facts.get(user_id)
    if not facts:
        return {}
    if isinstance(facts, dict):
        facts = [facts]
    overrides: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        target = fact.get("target_event_id")
        kind = fact.get("kind")
        if not target or kind not in {
            "event_cancellation",
            "event_amendment",
            "event_delay",
            "settlement_confirmation",
        }:
            continue
        overrides[str(target)] = {
            "kind": kind,
            "amount": fact.get("amount"),
            "currency": fact.get("currency"),
            "effective_date": _parse_date(fact.get("effective_date")),
            "note": fact.get("note", ""),
        }
    return overrides


def apply_event_overrides(
    events: pd.DataFrame,
    overrides: dict[str, dict[str, Any]],
    skipped: list[str],
) -> pd.DataFrame:
    """Apply explicit cancel/delay/amend/settle facts to specific event rows."""
    if events.empty or not overrides:
        return events

    adjusted = events.copy()
    for idx, row in adjusted.iterrows():
        event_id = str(row["event_id"])
        override = overrides.get(event_id)
        if override is None:
            continue
        kind = override["kind"]
        if kind == "event_cancellation":
            adjusted.at[idx, "status"] = "cancelled"
            skipped.append(f"{event_id}: cancelled by message evidence")
        elif kind == "event_delay":
            new_date = override.get("effective_date")
            if new_date is not None:
                adjusted.at[idx, "settlement_date"] = new_date.isoformat()
                skipped.append(f"{event_id}: delayed to {new_date.isoformat()}")
        elif kind == "event_amendment":
            if override.get("amount") is not None:
                adjusted.at[idx, "amount"] = float(override["amount"])
                if override.get("currency"):
                    adjusted.at[idx, "currency"] = str(override["currency"])
            new_date = override.get("effective_date")
            if new_date is not None:
                adjusted.at[idx, "settlement_date"] = new_date.isoformat()
            skipped.append(f"{event_id}: amended by message evidence")
        elif kind == "settlement_confirmation":
            adjusted.at[idx, "status"] = "settled"
            if override.get("effective_date") is not None:
                adjusted.at[idx, "settlement_date"] = override["effective_date"].isoformat()
            skipped.append(f"{event_id}: settlement confirmed by message evidence")
    return adjusted


def prepare_user_events(
    user_id: str,
    events: pd.DataFrame,
    image_amounts: dict[str, Any],
    message_facts: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """One-call preparation: fill amounts, apply overrides, return salary override."""
    skipped: list[str] = []
    working = events
    working = apply_image_amounts(working, image_amounts, skipped)
    working = apply_event_overrides(
        working, event_overrides_for_user(user_id, message_facts), skipped
    )
    salary = salary_overrides_for_user(user_id, message_facts)
    return working, salary, skipped
