"""Generates synthetic credit-scoring dataset into data/raw/train.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RANDOM_STATE = 42
N_SAMPLES = 5_000

rng = np.random.default_rng(RANDOM_STATE)

age = rng.integers(18, 70, N_SAMPLES)
income = rng.normal(60_000, 20_000, N_SAMPLES).clip(10_000, 200_000)
loan_amount = rng.normal(15_000, 8_000, N_SAMPLES).clip(1_000, 80_000)
credit_history = rng.integers(0, 15, N_SAMPLES)
num_accounts = rng.integers(1, 10, N_SAMPLES)
employment_type = rng.choice(["employed", "self_employed", "unemployed"], N_SAMPLES, p=[0.6, 0.25, 0.15])
education = rng.choice(["secondary", "bachelor", "master", "phd"], N_SAMPLES, p=[0.35, 0.40, 0.20, 0.05])

log_odds = (
    -2.0
    + 0.02 * (age - 18)
    + 0.000015 * income
    - 0.00005 * loan_amount
    + 0.15 * credit_history
    + 0.1 * num_accounts
    + np.where(employment_type == "employed", 0.5, np.where(employment_type == "self_employed", 0.1, -0.8))
    + np.where(education == "phd", 0.6, np.where(education == "master", 0.3, 0.0))
    + rng.normal(0, 0.5, N_SAMPLES)
)
prob = 1 / (1 + np.exp(-log_odds))
target = (rng.uniform(size=N_SAMPLES) < prob).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income.round(2),
    "loan_amount": loan_amount.round(2),
    "credit_history_years": credit_history,
    "num_accounts": num_accounts,
    "employment_type": employment_type,
    "education": education,
    "target": target,
})

out_path = Path("data/raw/train.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows → {out_path}  |  default rate: {target.mean():.2%}")
