"""
generate_metrics_matrix.py
-----------------------------------------------------------
Reads the Llama checkpoint CSV and prints a full metrics
matrix including:
  - Output Validity Rate
  - Conditional Accuracy  (valid predictions only)
  - End-to-End Accuracy   (INVALID counted as wrong)
  - Macro-F1              (valid predictions only)
  - Per-class F1          (Positive / Negative / Mixed_feelings)

Usage:
    python generate_metrics_matrix.py

    By default reads the checkpoint file path set in CSV_PATH below.
    Change CSV_PATH to point to your actual file.
-----------------------------------------------------------
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

# ── CONFIG — change this path to your actual checkpoint file ──────────────────
CSV_PATH = r"..\results\run_llama_final_475_20260428_230540.csv"

# Model columns and display names
MODELS = {
    "llama33_70b_pred":  "Llama 3.3 70B",
    "llama31_8b_pred":   "Llama 3.1 8B",
    "llama4_scout_pred": "Llama 4 Scout",
}

VALID_LABELS = ["Positive", "Negative", "Mixed_feelings"]
LABEL_ORDER  = ["Positive", "Negative", "Mixed_feelings"]

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
N  = len(df)
print(f"\nLoaded {N} samples from: {CSV_PATH}")
print(f"True label distribution:")
print(df["true_label"].value_counts().to_string())

# ── Main metrics table ────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"{'Model':<20} {'N':>5} {'Validity':>10} {'Cond.Acc':>10} {'E2E.Acc':>10} {'Macro-F1':>10}")
print("-" * 78)

results_summary = []

for col, name in MODELS.items():
    preds = df[col]

    # Validity
    valid_mask   = preds.isin(VALID_LABELS)
    n_valid      = valid_mask.sum()
    validity     = n_valid / N

    # Conditional accuracy (valid only)
    valid_df     = df[valid_mask]
    cond_acc     = (valid_df[col] == valid_df["true_label"]).mean() if n_valid > 0 else 0.0

    # End-to-end accuracy (INVALID = wrong)
    e2e_acc      = (preds == df["true_label"]).mean()

    # Macro-F1 on valid predictions only
    if n_valid > 0:
        macro_f1 = f1_score(
            valid_df["true_label"], valid_df[col],
            labels=LABEL_ORDER, average="macro", zero_division=0
        )
    else:
        macro_f1 = 0.0

    print(f"{name:<20} {n_valid:>5}/{N} {validity:>9.1%} {cond_acc:>10.1%} {e2e_acc:>10.1%} {macro_f1:>10.4f}")
    results_summary.append({
        "Model": name, "N_valid": n_valid, "N_total": N,
        "Validity": round(validity, 4),
        "Cond_Acc": round(cond_acc, 4),
        "E2E_Acc":  round(e2e_acc, 4),
        "Macro_F1": round(macro_f1, 4),
    })

print("=" * 78)

# ── Per-class F1 breakdown ────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PER-CLASS F1 SCORES (on valid predictions only)")
print("=" * 78)
print(f"{'Model':<20} {'Positive':>12} {'Negative':>12} {'Mixed_feelings':>16}")
print("-" * 78)

for col, name in MODELS.items():
    valid_mask = df[col].isin(VALID_LABELS)
    valid_df   = df[valid_mask]
    if len(valid_df) == 0:
        print(f"{name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>16}")
        continue
    f1s = f1_score(
        valid_df["true_label"], valid_df[col],
        labels=LABEL_ORDER, average=None, zero_division=0
    )
    # f1s order matches LABEL_ORDER: Positive, Negative, Mixed_feelings
    print(f"{name:<20} {f1s[0]:>12.4f} {f1s[1]:>12.4f} {f1s[2]:>16.4f}")

print("=" * 78)

# ── Full classification report per model ─────────────────────────────────────
for col, name in MODELS.items():
    valid_mask = df[col].isin(VALID_LABELS)
    valid_df   = df[valid_mask]
    print(f"\n{'─'*50}")
    print(f"Classification Report — {name}  ({valid_mask.sum()}/{N} valid)")
    print(f"{'─'*50}")
    if len(valid_df) == 0:
        print("  No valid predictions.")
        continue
    print(classification_report(
        valid_df["true_label"], valid_df[col],
        labels=LABEL_ORDER, zero_division=0
    ))

# ── Confusion matrices ────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("CONFUSION MATRICES (rows = true, cols = predicted)")
print(f"Label order: {LABEL_ORDER}")
print("=" * 78)

for col, name in MODELS.items():
    valid_mask = df[col].isin(VALID_LABELS)
    valid_df   = df[valid_mask]
    print(f"\n{name}:")
    if len(valid_df) == 0:
        print("  No valid predictions.")
        continue
    cm = confusion_matrix(
        valid_df["true_label"], valid_df[col], labels=LABEL_ORDER
    )
    header = f"{'':>16}" + "".join(f"{l:>16}" for l in LABEL_ORDER)
    print(header)
    for i, row_label in enumerate(LABEL_ORDER):
        row_str = f"{row_label:>16}" + "".join(f"{cm[i][j]:>16}" for j in range(len(LABEL_ORDER)))
        print(row_str)

# ── Save summary CSV ──────────────────────────────────────────────────────────
import os
out_dir     = os.path.dirname(os.path.abspath(CSV_PATH))
summary_csv = os.path.join(out_dir, "metrics_summary.csv")
pd.DataFrame(results_summary).to_csv(summary_csv, index=False)
print(f"\nSummary saved → {summary_csv}")
print("\nDone.")
