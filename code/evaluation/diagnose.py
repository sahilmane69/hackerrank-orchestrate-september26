"""Per-field mismatch diagnostics comparing pipeline decisions with sample ground truth.

Run from the project root:
    .venv/bin/python code/evaluation/diagnose.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

_parent = Path(__file__).resolve().parent
if (_parent / "code").exists():
    CODE_DIR = _parent / "code"
else:
    CODE_DIR = _parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import DATASET_DIR
from data import load_dataset_files, parse_date
from evidence import build_evidence_lookups
from plans import generate_candidates, make_decision, rank_key

SAMPLE_PATH = DATASET_DIR / "sample_requests.csv"


def main() -> None:
    sample_df = pd.read_csv(SAMPLE_PATH)
    _, profiles_df, events_df, opts_df, rates_df = load_dataset_files(DATASET_DIR)
    blank_amounts, confirmed_incomes, cancelled_events, amended_events = build_evidence_lookups()

    fields = ["status", "method", "plan", "earliest", "spending"]
    cat_counts: dict[str, int] = {f: 0 for f in fields}

    for _, expected in sample_df.iterrows():
        req_id = str(expected["request_id"])
        user_id = str(expected["user_id"])

        profile = profiles_df[profiles_df["user_id"] == user_id].iloc[0]
        events = events_df[events_df["user_id"] == user_id]
        opts = opts_df[opts_df["request_id"] == req_id] if not opts_df.empty else pd.DataFrame()
        incomes = confirmed_incomes.get(user_id, [])

        pred = make_decision(
            expected,
            profile,
            events,
            opts,
            exchange_rates=rates_df,
            blank_amounts=blank_amounts,
            confirmed_incomes=incomes,
            cancelled_events=cancelled_events,
            amended_events=amended_events,
        )

        exp_status = str(expected["affordability_status"]).strip()
        exp_method = str(expected["recommended_payment_method"]).strip()
        exp_plan = str(expected["payment_plan"]).strip()
        raw_earliest = expected.get("earliest_date_for_full_payment")
        exp_earliest = str(raw_earliest).strip() if pd.notna(raw_earliest) else ""
        exp_spending = str(expected["spending_changes_needed"]).strip()

        matches = {
            "status": pred.affordability_status == exp_status,
            "method": pred.recommended_payment_method == exp_method,
            "plan": pred.payment_plan == exp_plan,
            "earliest": pred.earliest_date_for_full_payment == exp_earliest,
            "spending": pred.spending_changes_needed == exp_spending,
        }
        if all(matches.values()):
            continue

        bad_fields = [f for f, ok in matches.items() if not ok]
        for f in bad_fields:
            cat_counts[f] += 1

        print("=" * 100)
        print(f"MISMATCH {req_id} ({user_id}) bad={bad_fields}")
        print(f"  requested={expected['requested_amount']} on {expected['request_date']} "
              f"deadline={expected['desired_completion_date']} allows_partial={expected.get('allows_partial_payment')}")
        print(f"  EXPECTED : status={exp_status} method={exp_method} plan={exp_plan} "
              f"earliest={exp_earliest} spending={exp_spending}")
        print(f"  PREDICTED: status={pred.affordability_status} method={pred.recommended_payment_method} "
              f"plan={pred.payment_plan} earliest={pred.earliest_date_for_full_payment} "
              f"spending={pred.spending_changes_needed}")
        print(f"  safe_today={pred.amount_safe_to_pay}")
        print(f"  EXPLANATION (expected): {expected.get('decision_explanation', '')}")
        print(f"  EXPLANATION (pred): {pred.decision_explanation}")

        # Candidate table: every candidate with rank key and eligibility
        for plan in sorted(pred.candidates, key=rank_key)[:12]:
            pays = ",".join(f"{p.pay_date}:{p.amount}" for p in plan.payments) or "-"
            changes = ",".join(
                f"{c.action}:{c.event_id}" + (f":{c.new_amount}" if c.new_amount else "")
                for c in plan.spending_changes
            ) or "-"
            print(
                f"    cand method={plan.method:16s} safe={plan.is_safe!s:5s} dl={plan.meets_deadline!s:5s} "
                f"elig={plan.eligible!s:5s} total={plan.total_paid:>9} changes={changes:20s} pays=[{pays}] "
                f"opt={plan.payment_option_id} notes={plan.notes}"
            )

    print("=" * 100)
    print(f"Mismatch field counts over {len(sample_df)} samples: {cat_counts}")


if __name__ == "__main__":
    main()
