"""Data cleaning helpers for the banking churn dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "balance",
    "num_products",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
    "geography",
    "gender",
    "churned",
]

NUMERIC_COLUMNS = [
    "age",
    "tenure_months",
    "balance",
    "num_products",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
    "churned",
]


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, coerce dtypes, and apply range guards.

    Returns a cleaned copy with a deterministic row order.
    """
    df = df.copy()

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    df = df[EXPECTED_COLUMNS]

    # Drop duplicate customers and rows missing the target.
    df = df.drop_duplicates(subset="customer_id")
    df = df.dropna(subset=["churned"])

    # Coerce numerics (defensive: a hand-edited CSV could carry strings).
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Median imputation for any residual numeric gaps.
    for col in ["age", "tenure_months", "balance", "estimated_salary"]:
        df[col] = df[col].fillna(df[col].median())

    # Range guards: clamp to plausible bounds rather than dropping rows.
    df["age"] = df["age"].clip(lower=18, upper=100).astype(int)
    df["tenure_months"] = df["tenure_months"].clip(lower=0, upper=120).astype(int)
    df["balance"] = df["balance"].clip(lower=0).round(2)
    df["num_products"] = df["num_products"].clip(lower=1, upper=4).astype(int)
    df["estimated_salary"] = df["estimated_salary"].clip(lower=0).round(2)
    for col in ["has_credit_card", "is_active_member", "churned"]:
        df[col] = df[col].fillna(0).astype(int)

    df["geography"] = df["geography"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip()

    # Derived tenure band for segment analysis (also used in SQL).
    bins = [-1, 12, 36, 60, 120]
    labels = ["0-12 months", "13-36 months", "37-60 months", "61+ months"]
    df["tenure_band"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)

    return df.reset_index(drop=True)


def basic_quality_report(df: pd.DataFrame) -> dict:
    """Return row count, churn rate, missing-value and duplicate stats."""
    return {
        "rows": int(len(df)),
        "churn_rate": float(df["churned"].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_customer_ids": int(df.duplicated(subset="customer_id").sum()),
    }
