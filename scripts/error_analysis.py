"""
Error analysis on Mistral Large (500-sample, 5-shot).
Categorises the 182 misclassified samples.
Produces: results/error_analysis_<timestamp>.csv + .txt
"""
import os, sys, glob, re
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from load_data import VALID_LABELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def find_latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return files[-1] if files else None


# ── Error category heuristics ─────────────────────────────────────────────────
# Priority: detect sarcasm/irony markers → else check transition type

SARCASM_KEYWORDS = [
    r'\bfine\b', r'\bgreat\b.*\bbut\b', r'\b(entha|enth)\b',   # Malayalam sarcasm phrases
    r'sarcastically', r'\boh sure\b', r'\bhaha\b.*\bbore\b',
    r'\bwow\b.*\bwaste\b', r'\bsuperb\b.*\bwaste\b',
    r'ivide onnum', r'evideyenkilum',
]

MIXED_KEYWORDS = [
    r'\bbut\b', r'\bpakshe\b', r'\bennalum\b', r'\bathava\b',
    r'\bhowever\b', r'\bthough\b', r'\bstill\b', r'\baverage\b',
    r'\bokay\b', r'\bokke\b',
]


def categorise_error(row):
    true_l  = row['true_label']
    pred_l  = row['mistral_pred']
    text    = str(row['text']).lower()

    if true_l == pred_l:
        return None  # not an error

    # Sarcasm/irony: correct label is Positive but predicted Negative
    if true_l == 'Positive' and pred_l == 'Negative':
        for pat in SARCASM_KEYWORDS:
            if re.search(pat, text):
                return 'Sarcasm/Irony'

    # Mixed feelings confusion: anything involving Mixed_feelings
    if true_l == 'Mixed_feelings' or pred_l == 'Mixed_feelings':
        return 'Mixed_feelings Confusion'

    # Clear-polarity flip
    if (true_l == 'Positive' and pred_l == 'Negative') or \
       (true_l == 'Negative' and pred_l == 'Positive'):
        return 'Clear-Polarity Flip'

    return 'Other'


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 72)
    log("ERROR ANALYSIS — Mistral Large (500-sample, 5-shot)")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)

    f500 = find_latest("exp4_5_500sample_*.csv")
    if f500 is None:
        f500 = find_latest("sample_results.csv")
    if f500 is None:
        print("ERROR: 500-sample results not found. Run exp4_5_500sample.py first.")
        return

    df = pd.read_csv(f500)
    log(f"Source: {os.path.basename(f500)}  ({len(df)} rows)")

    # Only valid Mistral predictions
    valid_df = df[df['mistral_pred'].isin(VALID_LABELS)].copy()
    log(f"Valid Mistral predictions: {len(valid_df)}")

    valid_df['is_error'] = valid_df['mistral_pred'] != valid_df['true_label']
    errors   = valid_df[valid_df['is_error']].copy()
    log(f"Errors: {len(errors)}  (error rate: {len(errors)/len(valid_df):.1%})")

    # Categorise
    errors['error_category'] = errors.apply(categorise_error, axis=1)

    # ── Distribution by true label ─────────────────────────────────────────────
    log("\n" + "─" * 72)
    log("ERROR RATE BY TRUE LABEL")
    log(f"\n{'True Label':<20} {'Total':>8} {'Errors':>8} {'Error Rate':>12}")
    log("─" * 55)
    for label in VALID_LABELS:
        ldf = valid_df[valid_df['true_label'] == label]
        errs = ldf['is_error'].sum()
        rate = errs / len(ldf) if len(ldf) > 0 else 0
        log(f"{label:<20} {len(ldf):>8} {errs:>8} {rate:>11.1%}")

    # ── Category distribution ─────────────────────────────────────────────────
    log("\n" + "─" * 72)
    log("ERROR CATEGORY DISTRIBUTION")
    cat_counts = errors['error_category'].value_counts()
    log(f"\n{'Category':<30} {'Count':>8} {'%':>8}")
    log("─" * 50)
    for cat, cnt in cat_counts.items():
        log(f"{cat:<30} {cnt:>8} {cnt/len(errors):>7.1%}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    log("\n" + "─" * 72)
    log("CONFUSION MATRIX (Mistral Large, valid preds only)")
    cm_data = []
    header = f"{'True → Pred':<25}" + "".join(f"{l:>16}" for l in VALID_LABELS)
    log(header)
    log("─" * (25 + 16 * len(VALID_LABELS)))
    for true_l in VALID_LABELS:
        row_str = f"{true_l:<25}"
        for pred_l in VALID_LABELS:
            n = ((valid_df['true_label'] == true_l) & (valid_df['mistral_pred'] == pred_l)).sum()
            row_str += f"{n:>16}"
            cm_data.append({'true_label': true_l, 'pred_label': pred_l, 'count': n})
        log(row_str)

    # ── Linguistic correlates ─────────────────────────────────────────────────
    if 'malayalam_density' in valid_df.columns:
        log("\n" + "─" * 72)
        log("LINGUISTIC CORRELATES OF ERRORS")
        err_density = valid_df[valid_df['is_error']]['malayalam_density'].mean()
        ok_density  = valid_df[~valid_df['is_error']]['malayalam_density'].mean()
        log(f"  Avg Malayalam density — errors:  {err_density:.3f}")
        log(f"  Avg Malayalam density — correct: {ok_density:.3f}")
        log(f"  Delta: {err_density - ok_density:+.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    err_csv   = os.path.join(RESULTS_DIR, f"error_analysis_{ts}.csv")
    cm_csv    = os.path.join(RESULTS_DIR, f"confusion_matrix_{ts}.csv")
    txt_path  = os.path.join(RESULTS_DIR, f"error_analysis_{ts}.txt")

    # Full error records
    save_cols = ['sample_id' if 'sample_id' in errors.columns else errors.index.name,
                 'text', 'true_label', 'mistral_pred', 'error_category']
    save_cols = [c for c in save_cols if c and c in errors.columns]
    errors[save_cols].to_csv(err_csv, index=False)
    pd.DataFrame(cm_data).to_csv(cm_csv, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {err_csv}")
    print(f"Saved: {cm_csv}")
    print(f"Saved: {txt_path}")

if __name__ == "__main__":
    main()
