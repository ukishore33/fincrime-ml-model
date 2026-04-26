# 🤖 Financial Crime Detection ML Model

> **Production-grade financial fraud detection pipeline** — XGBoost + Random Forest trained on a PaySim-inspired synthetic dataset with SMOTE oversampling for class imbalance. Achieves ROC-AUC 1.00, PR-AUC 1.00, and zero false negatives on a 1.12% fraud-rate dataset.

---

## 📌 Project Overview

This project builds an end-to-end ML pipeline for detecting financial crime in payment transactions — directly mirroring the models used by fraud analytics teams at banks, payment platforms, and fintech companies. The dataset is modelled on the PaySim simulation framework (IEEE-CIS-style), with realistic class imbalance, transaction balance mechanics, and fraud typologies.

**Relevance:** Directly applicable to Data Scientist II, Data Analyst (ML), and Financial Crime Analytics roles at UST Global, Wissen Technology, Oracle, and BFSI analytics teams.

---

## 🏗️ Project Structure

```
fincrime-ml-model/
│
├── generate_data.py          # PaySim-inspired synthetic data generator (10,000 txns)
├── train_model.py            # XGBoost + Random Forest + SMOTE training pipeline
├── fincrime_dashboard.html   # Interactive ML dashboard (Chart.js)
│
├── paysim_data.csv           # Raw synthetic payment transactions
├── paysim_scored.csv         # Enriched with ML fraud scores and tiers
├── model_results.json        # Full metrics, feature importances, ROC data
│
└── README.md
```

---

## 📊 Dataset — PaySim-Inspired Synthetic Data

| Property | Value |
|---|---|
| Total Transactions | 10,000 |
| Simulation Window | 30 days (720 hours) |
| Fraud Cases | 112 (1.12% — realistic class imbalance) |
| Fraud Transaction Types | CASH_OUT, TRANSFER only (PaySim pattern) |
| Total Fraud Amount | ₹2.87 Crore |
| Avg Fraud Amount | ₹2.56 Lakh |
| Avg Legit Amount | ₹15,448 |

---

## ⚖️ Class Imbalance — Why SMOTE Matters

At 1.12% fraud rate, a naïve model predicts all-legit and achieves **98.88% accuracy** while catching **zero fraud**. This is the classic imbalanced-class trap in financial crime ML.

**Solution:** SMOTE (Synthetic Minority Oversampling Technique) from `imblearn` — generates synthetic fraud samples via k-nearest-neighbours interpolation:

| Stage | Fraud Samples | Legit Samples | Ratio |
|---|---|---|---|
| Before SMOTE | 90 (train) | 7,910 | 1 : 88 |
| After SMOTE | 7,910 | 7,910 | 1 : 1 |

This forces the model to learn genuine fraud decision boundaries rather than exploiting the majority-class shortcut.

---

## 🔧 Feature Engineering

| Feature | Description | Importance |
|---|---|---|
| `dest_balance_zeroed` | Destination balance drained to zero post-txn | **45.25%** |
| `balance_diff_dest` | Destination account balance change | **30.40%** |
| `error_balance_dest` | Accounting error: old_dest + amount ≠ new_dest | **24.30%** |
| `log_amount` | Log-transformed transaction amount | 0.20% |
| `old_balance_orig` | Originator account balance before txn | 0.20% |
| `surp_orig_zeroed` | Originator account drained to zero | Derived |
| `balance_drain_pct` | Amount / originator balance | Derived |
| `is_transfer_cashout` | TRANSFER or CASH_OUT binary flag | Derived |
| `night_txn_flag` | Transaction between 22:00–06:00 | Derived |
| `error_balance_orig` | Accounting discrepancy on originator side | Derived |

> The top 3 features are **balance mechanics** — fraudsters drain destination accounts to zero immediately, leaving detectable accounting inconsistencies. This mirrors real-world fraud ML findings from published PaySim research.

---

## 🤖 ML Models

**Primary:** XGBoost Classifier (300 estimators, depth 6, lr 0.05, SMOTE-balanced)  
**Baseline:** Random Forest (200 estimators, balanced class weights, 5-fold CV)

### Performance

| Metric | XGBoost | Random Forest |
|---|---|---|
| Accuracy | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1 Score | 1.00 | 1.00 |
| ROC-AUC | 1.00 | 1.00 |
| **PR-AUC** | **1.00** | **1.00** |
| 5-Fold CV AUC | — | 1.00 ± 0.00 |

> PR-AUC (Precision-Recall AUC) is the gold standard metric for fraud detection on imbalanced datasets — unlike ROC-AUC, it is sensitive to the minority class performance and does not inflate under class imbalance.

### Confusion Matrix (Test Set — 2,000 transactions)

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 1,978 ✓ | 0 |
| **Actual Fraud** | 0 | 22 ✓ |

---

## 📊 Dashboard Features

- **8 KPI cards** — Txns, fraud cases, fraud amount, ROC-AUC, PR-AUC, F1, SMOTE ratio, false negatives
- **Fraud vs Legit by Type** — stacked bar (PAYMENT/TRANSFER/CASH_OUT/DEBIT/CASH_IN)
- **Fraud distribution by hour** — time-series line chart
- **Alert tier distribution** — doughnut
- **Feature importance** — colour-coded bars with importance percentages
- **SMOTE explainer panel** — before/after oversampling with rationale
- **Model comparison** — XGBoost vs Random Forest metrics + confusion matrix
- **Fraud alert queue** — 112 cases filterable by type, tier, and flag

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python · Pandas · NumPy (PaySim-inspired) |
| Class Imbalance | imblearn · SMOTE |
| ML Models | XGBoost · scikit-learn (Random Forest) |
| Feature Engineering | Balance error detection, log transforms, derived flags |
| Cross-Validation | StratifiedKFold (5-fold) |
| Evaluation | ROC-AUC · PR-AUC · F1 · Confusion Matrix |
| Visualisation | Chart.js · HTML/CSS · Fira Code font |

---

## ▶️ How to Run

```bash
git clone https://github.com/ukishore33/fincrime-ml-model.git
cd fincrime-ml-model
pip install pandas numpy scikit-learn xgboost imbalanced-learn
python generate_data.py
python train_model.py
open fincrime_dashboard.html
```

---

## 👤 Author

**Kishore U.**  
AML/KYC Compliance Analyst | Data Analytics  
📱 6303308133 | Bengaluru, Karnataka | Immediate Joiner  
🔗 [LinkedIn](https://www.linkedin.com/in/kishore-techie/) · [GitHub](https://github.com/ukishore33)

---

## 📜 Disclaimer

All data is 100% synthetic — generated programmatically with no real financial or personal data. PaySim-inspired dataset structure based on published academic research. Built purely for portfolio demonstration.
