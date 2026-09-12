"""
Buy or Wait? — full-dataset deterministic pipeline.

Reads the dataset and the LLM extraction artifacts (never calls the API),
decides every request in dataset/requests.csv, and writes dataset/output.csv
plus code/evaluation/usage_report.md.

Run from the project root:
    .venv/bin/python code/run_full.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evidence import load_artifacts, prepare_user_events  # noqa: E402
from forecast import parse_date, to_number  # noqa: E402
from models import (  # noqa: E402
    CandidatePlan,
    format_payment_plan,
    format_spending_changes,
)
from plan_generator import generate_candidates, rank_key  # noqa: E402

PROJECT_ROOT = CODE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_PATH = DATASET_DIR / "output.csv"
USAGE_LOG_PATH = CODE_DIR / "extraction" / "usage_log.json"
USAGE_REPORT_PATH = CODE_DIR / "evaluation" / "usage_report.md"

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

REQUEST_COUNTS = 0

VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc
    if df.empty:
        raise ValueError(f"CSV file has no data rows: {path}")
    return df


def format_money(amount: float) -> str:
    """Plain decimal for CSV columns: no thousands separators, 2dp max."""
    rounded = round(float(amount) + 1e-9, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}"


def format_currency(amount: float, currency: str) -> str:
    """Comma-grouped currency string for human-readable explanations."""
    rounded = round(float(amount) + 1e-9, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        text = f"{int(round(rounded)):,}"
    else:
        text = f"{rounded:,.2f}"
    return f"{currency} {text}"


def human_date(day: date) -> str:
    return f"{day.day} {day.strftime('%B %Y')}"


def category_of(event_id: str, events: pd.DataFrame) -> str:
    match = events[events["event_id"].astype(str) == str(event_id)]
    if match.empty:
        return ""
    return str(match.iloc[0]["category"])


def category_words(category: str) -> str:
    return category.replace("_", " ").strip()


def minimum_balance_text(profile: pd.Series) -> str:
    currency = str(profile["home_currency"])
    minimum = to_number(profile["minimum_balance_to_keep"], "minimum_balance_to_keep")
    return format_currency(minimum, currency)


def status_for_method(method: str) -> str:
    if method == "full_payment":
        return "affordable_now"
    if method in {"partial_payment", "installments"}:
        return "affordable_with_plan"
    if method == "wait":
        return "affordable_later"
    return "not_affordable"


def explanation_for(
    plan: CandidatePlan,
    request: pd.Series,
    profile: pd.Series,
    events: pd.DataFrame,
    safe_amount: float,
) -> str:
    currency = str(profile["home_currency"])
    requested = to_number(request["requested_amount"], "requested_amount")
    min_text = minimum_balance_text(profile)

    if plan.method == "full_payment":
        lead = f"Pay {format_currency(requested, currency)} today."
        if plan.spending_changes:
            first = plan.spending_changes[0]
            words = category_words(category_of(first.event_id, events))
            if first.action == "stop":
                lead = f"Stop {words}, then pay {format_currency(requested, currency)} today."
            elif first.new_amount is not None:
                lead = (
                    f"Reduce {words} to {format_currency(first.new_amount, currency)}, "
                    f"then pay {format_currency(requested, currency)} today."
                )
        return f"{lead} This leaves at least {min_text} available over the next 90 days."

    if plan.method == "installments":
        count = len(plan.payments)
        each = plan.payments[0].amount if plan.payments else 0.0
        start = plan.payments[0].pay_date if plan.payments else parse_date(request["request_date"])
        return (
            f"Use {count} installments of {format_currency(each, currency)}, "
            f"starting {human_date(start)}. "
            f"This leaves at least {min_text} available over the next 90 days."
        )

    if plan.method == "partial_payment":
        last_payment = plan.payments[-1]
        return (
            f"Pay {format_currency(safe_amount, currency)} today and the remaining "
            f"{format_currency(requested - safe_amount, currency)} on "
            f"{human_date(last_payment.pay_date)}. "
            f"This completes the full request and keeps the {min_text} minimum."
        )

    if plan.method == "wait":
        pay_day = plan.payments[0].pay_date if plan.payments else parse_date(request["request_date"])
        return (
            f"Pay {format_currency(requested, currency)} in full on {human_date(pay_day)}. "
            f"Paying earlier would take the balance below the {min_text} minimum."
        )

    return f"Re-evaluate this request later; no safe option is available now."


def not_recommended_explanation(
    request: pd.Series,
    profile: pd.Series,
    earliest: date | None,
) -> str:
    currency = str(profile["home_currency"])
    deadline = parse_date(request["desired_completion_date"])
    deadline_text = human_date(deadline) if deadline else "the requested date"
    if earliest is not None and deadline is not None and earliest > deadline:
        return (
            f"Do not make this payment by {deadline_text}. The earliest safe full payment "
            f"is {human_date(earliest)}, after the deadline, and no option keeps the "
            f"{minimum_balance_text(profile)} minimum."
        )
    return (
        f"Do not make this payment by {deadline_text}. None of the available options keeps "
        f"the {minimum_balance_text(profile)} minimum protected."
    )


def decide_request(
    request: pd.Series,
    profile: pd.Series,
    events: pd.DataFrame,
    payment_options: pd.DataFrame,
    exchange_rates: pd.DataFrame | None,
    salary_override: dict[str, Any] | None,
) -> dict[str, str]:
    candidates, safe_amount, earliest = generate_candidates(
        request,
        profile,
        events,
        payment_options,
        exchange_rates=exchange_rates,
        salary_override=salary_override,
    )

    requested = to_number(request["requested_amount"], "requested_amount")
    safe_amount = max(0.0, min(safe_amount, requested))

    eligible = [candidate for candidate in candidates if candidate.eligible]
    chosen = min(eligible, key=rank_key) if eligible else None

    if chosen is None:
        return {
            "request_id": str(request["request_id"]),
            "amount_safe_to_pay": format_money(safe_amount),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": (
                earliest.isoformat() if earliest is not None else ""
            ),
            "spending_changes_needed": "none",
            "decision_explanation": not_recommended_explanation(
                request, profile, earliest
            ),
        }

    if chosen.method == "full_payment":
        earliest_text = str(request["request_date"])[:10]
    else:
        earliest_text = earliest.isoformat() if earliest is not None else ""

    return {
        "request_id": str(request["request_id"]),
        "amount_safe_to_pay": format_money(safe_amount),
        "affordability_status": status_for_method(chosen.method),
        "recommended_payment_method": chosen.method,
        "payment_plan": format_payment_plan(chosen.payments),
        "earliest_date_for_full_payment": earliest_text,
        "spending_changes_needed": format_spending_changes(chosen.spending_changes),
        "decision_explanation": explanation_for(
            chosen, request, profile, events, safe_amount
        ),
    }


def fallback_row(request_id: str, reason: str) -> dict[str, str]:
    print(f"  !! fallback for {request_id}: {reason}", file=sys.stderr)
    return {
        "request_id": request_id,
        "amount_safe_to_pay": "0",
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": (
            "Do not make this payment: the financial data for this request could not be "
            "evaluated safely, so no payment is recommended."
        ),
    }


def run_pipeline() -> pd.DataFrame:
    requests_df = load_csv(DATASET_DIR / "requests.csv")
    profiles_df = load_csv(DATASET_DIR / "financial_profiles.csv")
    events_df = load_csv(DATASET_DIR / "financial_events.csv")
    options_df = load_csv(DATASET_DIR / "request_payment_options.csv")
    rates_path = DATASET_DIR / "exchange_rates.csv"
    exchange_rates = load_csv(rates_path) if rates_path.exists() else None

    image_amounts, message_facts = load_artifacts()
    print(
        f"Loaded {len(requests_df)} requests, {len(profiles_df)} profiles, "
        f"{len(events_df)} events, {len(options_df)} payment options, "
        f"{len(image_amounts)} image facts, {len(message_facts)} message-fact users"
    )

    options_by_request = {
        str(request_id): group
        for request_id, group in options_df.groupby("request_id")
    }

    rows: list[dict[str, str]] = []
    fallbacks = 0
    for index, request in requests_df.iterrows():
        request_id = str(request["request_id"])
        user_id = str(request["user_id"])
        try:
            profile_rows = profiles_df[profiles_df["user_id"] == user_id]
            if profile_rows.empty:
                raise ValueError(f"no profile for {user_id}")
            profile = profile_rows.iloc[0]

            user_events = events_df[events_df["user_id"] == user_id]
            working_events, salary_override, _notes = prepare_user_events(
                user_id, user_events, image_amounts, message_facts
            )

            row = decide_request(
                request,
                profile,
                working_events,
                options_by_request.get(request_id, options_df.iloc[0:0]),
                exchange_rates,
                salary_override if salary_override else None,
            )
        except Exception as exc:  # keep producing one row per request_id
            fallbacks += 1
            row = fallback_row(request_id, f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"  processed {index + 1}/{len(requests_df)} requests")

    if fallbacks:
        print(f"WARNING: {fallbacks} requests used the fallback row")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def validate_output(df: pd.DataFrame, requests_df: pd.DataFrame) -> None:
    assert list(df.columns) == OUTPUT_COLUMNS, "output columns must match the contract"
    assert len(df) == len(requests_df), "one row per request required"
    assert df["request_id"].tolist() == requests_df["request_id"].astype(str).tolist(), (
        "request_id order must match requests.csv"
    )
    bad_status = set(df["affordability_status"]) - VALID_STATUSES
    bad_method = set(df["recommended_payment_method"]) - VALID_METHODS
    assert not bad_status, f"invalid affordability_status values: {bad_status}"
    assert not bad_method, f"invalid recommended_payment_method values: {bad_method}"
    for _, row in df.iterrows():
        assert row["payment_plan"] == "none" or ":" in row["payment_plan"], row["request_id"]
        assert row["spending_changes_needed"], row["request_id"]
        assert row["decision_explanation"].strip(), row["request_id"]
    print("Output validation passed.")


def write_output(df: pd.DataFrame) -> None:
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow(row.to_dict())
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


def write_usage_report() -> None:
    """Summarize the final extraction run's token usage (AGENTS.md §6.5)."""
    if not USAGE_LOG_PATH.exists():
        print("No usage log found; skipping usage report.")
        return
    usage = json.loads(USAGE_LOG_PATH.read_text(encoding="utf-8"))
    calls: list[dict[str, Any]] = usage.get("calls", [])
    totals = usage.get("totals", {})
    model = usage.get("model", "unknown")

    phases: dict[str, dict[str, int]] = {}
    for call in calls:
        bucket = phases.setdefault(
            call.get("phase", "unknown"),
            {"calls": 0, "prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        for key in ("prompt_tokens", "candidates_tokens", "total_tokens"):
            bucket[key] += int(call.get(key, 0) or 0)

    n_calls = int(totals.get("calls", len(calls)) or 0)
    total_tokens = int(totals.get("total_tokens", 0) or 0)
    avg = (total_tokens / n_calls) if n_calls else 0

    # Public list prices for gemini-3.6-flash (USD per 1M tokens); estimate only.
    price_in = 0.10
    price_out = 0.40
    prompt_tokens = int(totals.get("prompt_tokens", 0) or 0)
    output_tokens = int(totals.get("candidates_tokens", 0) or 0)
    cost_total = prompt_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
    cost_per_request = cost_total / REQUEST_COUNTS if REQUEST_COUNTS else 0.0

    lines = [
        "# Token Usage Report",
        "",
        f"- Model: `{model}` (Google Gemini API)",
        f"- Total model calls: {n_calls}",
        f"- Input (prompt) tokens: {prompt_tokens:,}",
        f"- Output (candidate) tokens: {output_tokens:,}",
        f"- Total tokens: {total_tokens:,}",
        f"- Average tokens per request: {avg:,.1f}",
        "",
        "## By phase",
        "",
        "| Phase | Calls | Prompt tokens | Output tokens | Total tokens |",
        "|---|---|---|---|---|",
    ]
    for phase, bucket in sorted(phases.items()):
        lines.append(
            f"| {phase} | {bucket['calls']} | {bucket['prompt_tokens']:,} | "
            f"{bucket['candidates_tokens']:,} | {bucket['total_tokens']:,} |"
        )
    lines += [
        "",
        "## Cost estimate",
        "",
        f"- Assumed pricing: USD {price_in:.2f} per 1M input tokens, "
        f"USD {price_out:.2f} per 1M output tokens (public list price).",
        f"- Estimated total cost: USD {cost_total:.4f}",
        f"- Estimated cost per request (over {REQUEST_COUNTS} evaluation requests): "
        f"USD {cost_per_request:.6f}",
        "",
        "Notes:",
        "",
        "- Model calls happen only once, in the offline extraction step "
        "(`code/extraction/run_extraction.py`). The scoring pipeline "
        "(`code/run_full.py`) is deterministic and makes no API calls.",
        "- Prices are estimates for reporting only; actual billing may differ.",
        "- No API keys or credentials are included in this report.",
        "",
    ]
    USAGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {USAGE_REPORT_PATH}")


def main() -> None:
    global REQUEST_COUNTS
    output = run_pipeline()
    requests_df = load_csv(DATASET_DIR / "requests.csv")
    REQUEST_COUNTS = len(requests_df)
    validate_output(output, requests_df)
    write_output(output)
    write_usage_report()

    print("\nDecision summary:")
    print(output["affordability_status"].value_counts().to_string())
    print(output["recommended_payment_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
