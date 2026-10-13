"""Generate a synthetic banking customer dataset for churn prediction.

SYNTHETIC DATA ONLY. All rows are generated with numpy using a fixed seed
(random_state=42). No real customer, employer, or proprietary data is used.

Churn is engineered so that:
  - low account activity (is_active_member = 0)  -> higher churn probability
  - short tenure                                  -> higher churn probability
  - fewer products held                          -> higher churn probability
  - low balance                                   -> slightly higher churn probability
Overall churn rate is targeted to land between ~15% and ~20%.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_CUSTOMERS = 10_000

GEOGRAPHIES = ["USA", "UK", "Canada"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    customer_id = np.arange(100_000, 100_000 + N_CUSTOMERS)
    age = rng.integers(18, 80, size=N_CUSTOMERS)
    tenure_months = rng.integers(0, 121, size=N_CUSTOMERS)  # 0-10 years
    balance = np.round(rng.gamma(shape=2.0, scale=25_000.0, size=N_CUSTOMERS), 2)
    balance = np.minimum(balance, 300_000.0)  # realistic upper bound
    num_products = rng.choice([1, 2, 3, 4], size=N_CUSTOMERS, p=[0.45, 0.35, 0.15, 0.05])
    has_credit_card = rng.choice([0, 1], size=N_CUSTOMERS, p=[0.30, 0.70])
    is_active_member = rng.choice([0, 1], size=N_CUSTOMERS, p=[0.45, 0.55])
    estimated_salary = np.round(rng.normal(loc=100_000.0, scale=35_000.0, size=N_CUSTOMERS), 2)
    estimated_salary = np.clip(estimated_salary, 15_000.0, 250_000.0)
    geography = rng.choice(GEOGRAPHIES, size=N_CUSTOMERS, p=[0.50, 0.30, 0.20])
    gender = rng.choice(["Male", "Female"], size=N_CUSTOMERS, p=[0.52, 0.48])

    # --- Churn probability engineered from business rules ---
    # Base log-odds calibrated to produce an overall churn rate of ~15-20%.
    log_odds = (
        -1.40
        + 1.20 * (1 - is_active_member)            # inactive members churn much more
        - 0.012 * tenure_months                     # longer tenure protects
        - 0.35 * (num_products - 1)                 # more products protect
        - 0.60 * (balance > 100_000).astype(int)    # high-balance protects
        + 0.25 * (age < 30).astype(int)             # younger customers churn slightly more
        + rng.normal(0.0, 0.25, size=N_CUSTOMERS)   # unobserved noise
    )
    churn_prob = sigmoid(log_odds)
    churned = (rng.random(N_CUSTOMERS) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "tenure_months": tenure_months,
            "balance": balance,
            "num_products": num_products,
            "has_credit_card": has_credit_card,
            "is_active_member": is_active_member,
            "estimated_salary": estimated_salary,
            "geography": geography,
            "gender": gender,
            "churned": churned,
        }
    )

    out_path = Path(__file__).resolve().parent.parent / "data" / "customers.csv"
    df.to_csv(out_path, index=False)

    churn_rate = df["churned"].mean()
    print(f"Wrote {len(df):,} synthetic customers to {out_path}")
    print(f"Overall churn rate: {churn_rate:.2%}")
    print(f"Size: {out_path.stat().st_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
