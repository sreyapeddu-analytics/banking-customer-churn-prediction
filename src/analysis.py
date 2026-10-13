"""EDA helpers for the banking churn dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate, retention rate, and customer counts per value of `column`."""
    grouped = (
        df.groupby(column, observed=True)
        .agg(customers=("customer_id", "count"), churned_count=("churned", "sum"))
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churned_count"] / grouped["customers"]
    grouped["retention_rate"] = 1.0 - grouped["churn_rate"]
    grouped["pct_of_customers"] = grouped["customers"] / grouped["customers"].sum()
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame, by_churn: bool = True) -> pd.DataFrame:
    """Mean/median/std of numeric features, split by churned vs retained."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != "customer_id"]
    if by_churn:
        return df.groupby("churned")[numeric].agg(["mean", "median", "std"]).round(2)
    return df[numeric].agg(["mean", "median", "std"]).round(2)


def at_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """High-risk segment: inactive members with short tenure and one product.

    Mirrors the segments churn drivers point at; useful for sizing a
    retention campaign.
    """
    mask = (
        (df["is_active_member"] == 0)
        & (df["tenure_months"] <= 24)
        & (df["num_products"] == 1)
    )
    return df.loc[mask].copy()
