"""
Financial Crime Detection ML Model - Synthetic Data Generator
Author: Kishore U. | github.com/ukishore33 | linkedin.com/in/kishore-techie
Description: Generates PaySim-inspired synthetic payment transaction data
             with realistic fraud/money laundering patterns for ML classification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(2024)
random.seed(2024)

TXN_TYPES = ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"]
FRAUD_TYPES = ["CASH_OUT","TRANSFER"]  # PaySim: fraud only on these types

def generate_paysim(n=10000):
    """
    Generate PaySim-inspired synthetic transaction dataset.
    - ~1.8% fraud rate (realistic class imbalance)
    - 30-day simulation window
    - Realistic amount distributions per transaction type
    """
    records = []

    for i in range(n):
        # Fraud only on TRANSFER / CASH_OUT (PaySim pattern)
        txn_type = random.choice(TXN_TYPES)
        is_fraud  = (txn_type in FRAUD_TYPES) and (random.random() < 0.025)

        step = random.randint(1, 720)  # hours in 30-day sim
        hour = step % 24

        # Amount distribution
        if is_fraud:
            # Fraudsters drain accounts — amounts near or above balance
            amount = round(np.random.uniform(1000, 500000), 2)
        else:
            if txn_type == "CASH_IN":
                amount = round(np.random.lognormal(9, 1.2), 2)
            elif txn_type == "CASH_OUT":
                amount = round(np.random.lognormal(8.5, 1.3), 2)
            elif txn_type == "TRANSFER":
                amount = round(np.random.lognormal(9.5, 1.5), 2)
            else:
                amount = round(np.random.lognormal(8, 1.1), 2)
            amount = min(amount, 800000)

        # Account balances
        orig_balance_before = round(np.random.lognormal(10, 1.5), 2) if not is_fraud else round(amount * random.uniform(0.9, 1.5), 2)
        orig_balance_after  = max(0, round(orig_balance_before - amount, 2)) if txn_type in ["TRANSFER","CASH_OUT","PAYMENT","DEBIT"] else round(orig_balance_before + amount, 2)
        dest_balance_before = round(np.random.lognormal(9, 1.5), 2)
        dest_balance_after  = round(dest_balance_before + amount, 2) if not is_fraud else 0.0  # fraud: dest drains immediately

        # Engineered features
        balance_diff_orig = round(orig_balance_before - orig_balance_after, 2)
        balance_diff_dest = round(dest_balance_after - dest_balance_before, 2)
        amount_to_balance_ratio = round(amount / max(orig_balance_before, 1), 4)
        dest_zeroed = 1 if dest_balance_after == 0 and is_fraud else 0
        night_flag  = 1 if hour < 6 or hour >= 22 else 0
        weekend_flag = 1 if (step // 24) % 7 >= 5 else 0
        round_amount = 1 if amount % 1000 == 0 else 0

        records.append({
            "step":                   step,
            "hour":                   hour,
            "txn_type":               txn_type,
            "amount":                 amount,
            "name_orig":              f"C{random.randint(1000000,9999999)}",
            "old_balance_orig":       orig_balance_before,
            "new_balance_orig":       orig_balance_after,
            "name_dest":              f"C{random.randint(1000000,9999999)}",
            "old_balance_dest":       dest_balance_before,
            "new_balance_dest":       dest_balance_after,
            "balance_diff_orig":      balance_diff_orig,
            "balance_diff_dest":      balance_diff_dest,
            "amount_to_balance_ratio":amount_to_balance_ratio,
            "dest_balance_zeroed":    dest_zeroed,
            "night_txn_flag":         night_flag,
            "weekend_flag":           weekend_flag,
            "round_amount_flag":      round_amount,
            "is_fraud":               int(is_fraud),
        })

    df = pd.DataFrame(records)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    df.to_csv("data/paysim_data.csv", index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"✅ Generated {n:,} transactions")
    print(f"   Fraud: {fraud_count} ({fraud_count/n*100:.2f}%) | Legit: {n-fraud_count}")
    print(f"   Types: {df['txn_type'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    generate_paysim()
