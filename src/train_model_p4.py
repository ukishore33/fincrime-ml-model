"""
Financial Crime Detection ML Model - Training Pipeline
Author: Kishore U. | github.com/ukishore33 | linkedin.com/in/kishore-techie
Description: XGBoost + Random Forest with SMOTE oversampling for class imbalance.
             Full evaluation: ROC-AUC, PR-AUC, confusion matrix, feature importances.
             Mirrors production ML pipeline for fraud/AML detection on payment data.
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, average_precision_score
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, using Gradient Boosting")

def train():
    df = pd.read_csv("data/paysim_data.csv")

    # ── FEATURE ENGINEERING ────────────────────────────────────────────────
    le = LabelEncoder()
    df["txn_type_enc"] = le.fit_transform(df["txn_type"])

    df["log_amount"]          = np.log1p(df["amount"])
    df["log_old_bal_orig"]    = np.log1p(df["old_balance_orig"])
    df["error_balance_orig"]  = df["old_balance_orig"] - df["amount"] - df["new_balance_orig"]
    df["error_balance_dest"]  = df["old_balance_dest"] + df["amount"] - df["new_balance_dest"]
    df["surp_orig_zeroed"]    = ((df["new_balance_orig"] == 0) & (df["old_balance_orig"] > 0)).astype(int)
    df["high_amount_flag"]    = (df["amount"] > 200000).astype(int)
    df["balance_drain_pct"]   = df["amount"] / (df["old_balance_orig"] + 1)
    df["is_transfer_cashout"] = df["txn_type"].isin(["TRANSFER","CASH_OUT"]).astype(int)

    FEATURES = [
        "txn_type_enc", "log_amount", "log_old_bal_orig",
        "old_balance_orig", "new_balance_orig",
        "old_balance_dest", "new_balance_dest",
        "balance_diff_orig", "balance_diff_dest",
        "amount_to_balance_ratio", "dest_balance_zeroed",
        "night_txn_flag", "weekend_flag", "round_amount_flag",
        "error_balance_orig", "error_balance_dest",
        "surp_orig_zeroed", "high_amount_flag",
        "balance_drain_pct", "is_transfer_cashout",
    ]

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── SMOTE for class imbalance ──────────────────────────────────────────
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"✅ SMOTE: {y_train.sum()} fraud → {y_train_res.sum()} fraud (balanced)")

    def get_metrics(model_name, y_true, y_pred, y_proba):
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "model":       model_name,
            "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
            "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score":    round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc":     round(float(roc_auc_score(y_true, y_proba)), 4),
            "pr_auc":      round(float(average_precision_score(y_true, y_proba)), 4),
            "confusion_matrix": cm,
        }

    results = {}

    # ── XGBoost / GradientBoosting ─────────────────────────────────────────
    if HAS_XGB:
        xg = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        xg.fit(X_train_res, y_train_res)
        y_pred_xg  = xg.predict(X_test)
        y_proba_xg = xg.predict_proba(X_test)[:, 1]
        results["xgboost"] = get_metrics("XGBoost", y_test, y_pred_xg, y_proba_xg)

        # Feature importance from XGBoost
        feat_imp = sorted(
            zip(FEATURES, xg.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        primary_model = xg
        primary_proba = y_proba_xg
    else:
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
        gb.fit(X_train_res, y_train_res)
        y_pred_xg  = gb.predict(X_test)
        y_proba_xg = gb.predict_proba(X_test)[:, 1]
        results["gradient_boosting"] = get_metrics("Gradient Boosting", y_test, y_pred_xg, y_proba_xg)
        feat_imp = sorted(zip(FEATURES, gb.feature_importances_), key=lambda x: x[1], reverse=True)
        primary_model = gb
        primary_proba = y_proba_xg

    # ── Random Forest (comparison) ─────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    y_pred_rf  = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = get_metrics("Random Forest", y_test, y_pred_rf, y_proba_rf)

    # ── Cross-validation ──────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    results["random_forest"]["cv_auc_mean"] = round(float(cv_scores.mean()), 4)
    results["random_forest"]["cv_auc_std"]  = round(float(cv_scores.std()), 4)

    # ── Feature importances ───────────────────────────────────────────────
    feature_importance = [
        {"feature": f, "importance": round(float(v), 4)}
        for f, v in feat_imp
    ]

    # ── Score full dataset ────────────────────────────────────────────────
    model_key = "xgboost" if HAS_XGB else "gradient_boosting"
    df["fraud_score"] = (primary_model.predict_proba(X[FEATURES])[:, 1] * 100).round(1)
    df["fraud_tier"]  = df["fraud_score"].apply(lambda s: "High" if s>=60 else ("Medium" if s>=30 else "Low"))
    df["ml_alert"]    = (df["fraud_score"] >= 60).astype(int)
    df.to_csv("data/paysim_scored.csv", index=False)

    # ── ROC curve data ────────────────────────────────────────────────────
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, primary_proba)
    roc_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                for f, t in zip(fpr[::10], tpr[::10])]  # sample for JSON

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "total_transactions": int(len(df)),
        "total_fraud":        int(df["is_fraud"].sum()),
        "fraud_rate_pct":     round(float(df["is_fraud"].mean() * 100), 2),
        "ml_alerts":          int(df["ml_alert"].sum()),
        "fraud_by_type":      df.groupby("txn_type")["is_fraud"].sum().to_dict(),
        "fraud_amt_total":    round(float(df[df["is_fraud"]==1]["amount"].sum()), 2),
        "avg_fraud_amount":   round(float(df[df["is_fraud"]==1]["amount"].mean()), 2),
        "avg_legit_amount":   round(float(df[df["is_fraud"]==0]["amount"].mean()), 2),
    }

    output = {
        "metrics":            results,
        "feature_importance": feature_importance,
        "roc_data":           roc_data,
        "summary":            summary,
        "primary_model":      model_key,
    }

    with open("model_results.json", "w") as f:
        json.dump(output, f, indent=2)

    primary = results[model_key]
    rf_r    = results["random_forest"]
    print(f"\n✅ Training complete!")
    print(f"   {'XGBoost' if HAS_XGB else 'GradBoost'} → AUC: {primary['roc_auc']} | PR-AUC: {primary['pr_auc']} | F1: {primary['f1_score']} | Recall: {primary['recall']}")
    print(f"   Random Forest   → AUC: {rf_r['roc_auc']} | PR-AUC: {rf_r['pr_auc']} | F1: {rf_r['f1_score']} | CV-AUC: {rf_r.get('cv_auc_mean','—')}")
    print(f"   Fraud detected: {summary['total_fraud']} ({summary['fraud_rate_pct']}%) | Avg fraud amount: ₹{summary['avg_fraud_amount']:,.0f}")
    return output

if __name__ == "__main__":
    train()
