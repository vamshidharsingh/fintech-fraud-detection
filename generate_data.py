import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_records = 20000
fraud_rate = 0.04
n_fraud = int(n_records * fraud_rate)
n_normal = n_records - n_fraud

amount_normal = np.random.lognormal(mean=3, sigma=1, size=n_normal)
amount_fraud = np.random.lognormal(mean=5, sigma=1.2, size=n_fraud)

time_normal = np.random.exponential(scale=12, size=n_normal)
time_fraud = np.random.exponential(scale=1, size=n_fraud)

age_normal = np.random.normal(loc=500, scale=100, size=n_normal)
age_fraud = np.random.normal(loc=100, scale=50, size=n_fraud)

loc_normal = np.random.beta(a=2, b=8, size=n_normal)
loc_fraud = np.random.beta(a=8, b=2, size=n_fraud)

df_normal = pd.DataFrame({
    'amount': amount_normal,
    'time_since_last_tx': time_normal,
    'account_age_days': age_normal,
    'location_risk_score': loc_normal,
    'is_fraud': 0
})

df_fraud = pd.DataFrame({
    'amount': amount_fraud,
    'time_since_last_tx': time_fraud,
    'account_age_days': age_fraud,
    'location_risk_score': loc_fraud,
    'is_fraud': 1
})

df = pd.concat([df_normal, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

df['amount'] = df['amount'].clip(lower=0.1)
df['account_age_days'] = df['account_age_days'].clip(lower=0)

os.makedirs('data', exist_ok=True)
df.to_csv('data/transactions.csv', index=False)
print("Generated data/transactions.csv successfully.")
