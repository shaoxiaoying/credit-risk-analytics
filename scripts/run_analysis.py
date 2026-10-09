"""Reproducible cleaning, SQL summaries, descriptive plots, and baseline model."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/credit_default_source.csv"
PROCESSED = ROOT / "data/processed"
FIGURES = ROOT / "reports/figures"

# Canonical lower-case names used by the app and SQL file.
ALIASES = {
    "id": "client_id", "limit_bal": "limit_bal", "sex": "sex",
    "education": "education", "marriage": "marriage", "age": "age",
    "pay_0": "pay_0", "pay_2": "pay_2", "pay_3": "pay_3",
    "pay_4": "pay_4", "pay_5": "pay_5", "pay_6": "pay_6",
    "bill_amt1": "bill_amt1", "bill_amt2": "bill_amt2", "bill_amt3": "bill_amt3",
    "bill_amt4": "bill_amt4", "bill_amt5": "bill_amt5", "bill_amt6": "bill_amt6",
    "pay_amt1": "pay_amt1", "pay_amt2": "pay_amt2", "pay_amt3": "pay_amt3",
    "pay_amt4": "pay_amt4", "pay_amt5": "pay_amt5", "pay_amt6": "pay_amt6",
    "default.payment.next.month": "default_payment_next_month",
    "default_payment_next_month": "default_payment_next_month",
}
TARGET = "default_payment_next_month"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # UCI sources have appeared with both descriptive and X1...X23 names.
    if "X1" in df.columns:
        numbered = {f"X{i}": name for i, name in enumerate([
            "limit_bal", "sex", "education", "marriage", "age", "pay_0", "pay_2", "pay_3",
            "pay_4", "pay_5", "pay_6", "bill_amt1", "bill_amt2", "bill_amt3",
            "bill_amt4", "bill_amt5", "bill_amt6", "pay_amt1", "pay_amt2", "pay_amt3",
            "pay_amt4", "pay_amt5", "pay_amt6", "default_payment_next_month"], start=1)}
        df = df.rename(columns=numbered)
    rename = {c: ALIASES[c.strip().lower()] for c in df.columns if c.strip().lower() in ALIASES}
    df = df.rename(columns=rename)
    # Some source copies call the outcome Y.
    if TARGET not in df and "Y" in df:
        df = df.rename(columns={"Y": TARGET})
    return df


def make_plots(df: pd.DataFrame, y_test, pred, prob) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    status = df.groupby("pay_0", observed=True)[TARGET].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(status.pay_0.astype(str), status["mean"] * 100, color="#2864a5")
    ax.set(title="Observed default rate by most recent repayment status",
           xlabel="PAY_0 source status code", ylabel="Observed default rate (%)")
    ax.bar_label(bars, labels=[f"{v:.1f}%\nn={n:,}" for v, n in zip(status["mean"]*100, status["count"])], fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "default_by_repayment_status.png", dpi=180); plt.close(fig)

    bands = pd.cut(df["limit_bal"], bins=[-np.inf, 50000, 100000, 200000, 500000, np.inf],
                   labels=["<50k", "50–99k", "100–199k", "200–499k", "500k+"])
    grouped = df.assign(limit_band=bands).groupby("limit_band", observed=False)[TARGET].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.limit_band.astype(str), grouped["mean"]*100, color="#188977")
    ax.set(title="Observed default rate by credit-limit band", xlabel="Credit limit (NT$)", ylabel="Observed default rate (%)")
    ax.bar_label(bars, labels=[f"{v:.1f}%\nn={n:,}" for v, n in zip(grouped["mean"]*100, grouped["count"])], fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "default_by_credit_limit.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax, color="#a24871")
    ax.set_title("Baseline logistic regression: holdout ROC curve")
    fig.tight_layout(); fig.savefig(FIGURES / "holdout_roc_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Holdout confusion matrix (threshold = 0.50)")
    fig.tight_layout(); fig.savefig(FIGURES / "holdout_confusion_matrix.png", dpi=180); plt.close(fig)


def model_data(df: pd.DataFrame):
    features = [c for c in df.columns if c not in (TARGET, "client_id")]
    return df[features], df[TARGET].astype(int)


def build_model(columns) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())])
    prep = ColumnTransformer([("numeric", numeric, list(columns))])
    # Conservative regularization keeps rare, strongly separating categories from
    # producing unstable coefficients on this imbalanced historical sample.
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000, C=0.1, solver="liblinear", class_weight=None, random_state=42))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if not RAW.exists():
        if args.skip_download:
            raise SystemExit(f"Missing {RAW}; run python scripts/download_data.py first.")
        from download_data import main as download_main
        download_main()
    PROCESSED.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    df = normalize_columns(pd.read_csv(RAW))
    required = {"client_id", "limit_bal", "sex", "education", "marriage", "age",
                "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}. Source columns: {list(df.columns)}")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only non-missing 0/1 values.")
    df = df.drop_duplicates().copy()
    # Avoid row-number artifacts in the published clean table.
    df = df.drop(columns=[c for c in df if c.lower().startswith("unnamed:")], errors="ignore")
    df.to_csv(PROCESSED / "credit_clients_clean.csv", index=False)

    con = duckdb.connect()
    clean_path = (PROCESSED / "credit_clients_clean.csv").as_posix().replace("'", "''")
    con.execute(f"CREATE VIEW clients AS SELECT * FROM read_csv_auto('{clean_path}')")
    overall = con.execute(f"SELECT COUNT(*) n, AVG({TARGET}) rate FROM clients").fetchone()
    status = con.execute(f"SELECT pay_0 status, COUNT(*) customers, AVG({TARGET}) default_rate FROM clients GROUP BY 1 ORDER BY 1").df()
    status.to_csv(PROCESSED / "default_by_repayment_status.csv", index=False)
    limit = con.execute(f"""SELECT CASE WHEN limit_bal < 50000 THEN '< NT$50k' WHEN limit_bal < 100000 THEN 'NT$50k–99k'
        WHEN limit_bal < 200000 THEN 'NT$100k–199k' WHEN limit_bal < 500000 THEN 'NT$200k–499k' ELSE 'NT$500k+' END band,
        COUNT(*) customers, AVG({TARGET}) default_rate FROM clients GROUP BY 1 ORDER BY MIN(limit_bal)""").df()
    limit.to_csv(PROCESSED / "default_by_credit_limit.csv", index=False)

    X, y = model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = build_model(X.columns); model.fit(X_train, y_train)
    pred = model.predict(X_test); prob = model.predict_proba(X_test)[:, 1]
    metrics = {"rows": int(len(df)), "observed_default_rate": float(overall[1]), "train_rows": int(len(X_train)),
               "test_rows": int(len(X_test)), "threshold": 0.5,
               "accuracy": float(accuracy_score(y_test, pred)), "roc_auc": float(roc_auc_score(y_test, prob)),
               "precision_default": float(precision_score(y_test, pred, zero_division=0)),
               "recall_default": float(recall_score(y_test, pred, zero_division=0)),
               "f1_default": float(f1_score(y_test, pred, zero_division=0))}
    (PROCESSED / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    make_plots(df, y_test, pred, prob)
    summary = f"""# Analysis summary

## Sample

- Customers: **{metrics['rows']:,}**
- Observed next-month default rate: **{metrics['observed_default_rate']:.1%}**
- Duplicate rows removed: **{len(pd.read_csv(RAW)) - len(df):,}**

## Baseline model

Stratified 80/20 holdout; logistic regression; 0.50 classification threshold.

| Metric | Holdout |
|---|---:|
| ROC AUC | {metrics['roc_auc']:.3f} |
| Accuracy | {metrics['accuracy']:.3f} |
| Default precision | {metrics['precision_default']:.3f} |
| Default recall | {metrics['recall_default']:.3f} |
| Default F1 | {metrics['f1_default']:.3f} |

This model is an educational benchmark only. See the README for limitations and responsible-use notes.
"""
    (ROOT / "reports/analysis_summary.md").write_text(summary)
    print(f"Analysis complete: {metrics['rows']:,} rows; observed default rate {metrics['observed_default_rate']:.1%}; holdout ROC AUC {metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
