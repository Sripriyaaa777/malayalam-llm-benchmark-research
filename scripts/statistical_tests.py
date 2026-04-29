"""
Statistical significance tests for all experiments.
Run AFTER all experiment scripts have completed.
Produces: results/statistical_tests_<timestamp>.txt + .csv
"""
import os, sys, glob
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from load_data import VALID_LABELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def find_latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return files[-1] if files else None


def load_500():
    f = find_latest("exp4_5_500sample_*.csv")
    if f is None:
        f = find_latest("sample_results.csv")
    if f is None:
        raise FileNotFoundError("500-sample results not found.")
    return pd.read_csv(f), f


def load_roman():
    f = find_latest("exp6_romanization_*.csv")
    if f is None:
        raise FileNotFoundError("Romanization results not found. Run exp6_romanization.py first.")
    return pd.read_csv(f), f


def mcnemar_test(a_valid, b_valid):
    """McNemar's test on paired validity vectors (boolean Series)."""
    from scipy.stats import chi2
    b_b = ((a_valid == True)  & (b_valid == False)).sum()  # A valid, B invalid
    c_c = ((a_valid == False) & (b_valid == True)).sum()   # A invalid, B valid
    if b_b + c_c == 0:
        return float('nan'), float('nan')
    chi2_stat = (abs(b_b - c_c) - 1) ** 2 / (b_b + c_c)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2_stat, df=1)
    return chi2_stat, p


def cohens_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def bootstrap_ci(series_bool, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for a proportion."""
    arr = series_bool.values.astype(float)
    boot = np.array([arr[np.random.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)])
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi


def proportion_z_test(p, n, p0=0.5):
    """One-sample Z-test for proportion vs p0."""
    se = np.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val


def chi2_three_way(counts_valid, counts_total):
    """Chi-square test for three models' valid/invalid counts."""
    from scipy.stats import chi2_contingency
    table = np.array([counts_valid, [t - v for t, v in zip(counts_total, counts_valid)]])
    chi2, p, dof, _ = chi2_contingency(table)
    return chi2, p, dof


def main():
    lines = []
    records = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 72)
    log("STATISTICAL SIGNIFICANCE TESTS")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────────────
    df500, f500 = load_500()
    log(f"\nSource (500-sample): {os.path.basename(f500)}")

    mistral_valid = df500['mistral_pred'].isin(VALID_LABELS)
    llama_valid   = df500['llama_pred'].isin(VALID_LABELS)
    gemma_col = 'gemma_pred' if 'gemma_pred' in df500.columns else None
    gemma_valid = df500[gemma_col].isin(VALID_LABELS) if gemma_col else pd.Series([False] * len(df500))

    N = len(df500)
    p_mis = mistral_valid.mean()
    p_lla = llama_valid.mean()
    p_gem = gemma_valid.mean()

    # ── Test 1: McNemar's (Mistral vs Llama) ──────────────────────────────────
    log("\n" + "─" * 72)
    log("TEST 1 — McNemar's Test: Mistral vs Llama (paired validity)")
    chi2_stat, p_val = mcnemar_test(mistral_valid, llama_valid)
    log(f"  χ² = {chi2_stat:.4f},  p = {p_val:.2e}")
    log(f"  {'SIGNIFICANT (p < 0.001)' if p_val < 0.001 else 'NOT significant'}")
    records.append({"test": "McNemar Mistral vs Llama", "statistic": chi2_stat, "p_value": p_val})

    # ── Test 2: Three-way chi-square ──────────────────────────────────────────
    log("\n" + "─" * 72)
    log("TEST 2 — Chi-square: All three models (validity rates)")
    chi2_3, p_3, dof_3 = chi2_three_way(
        [mistral_valid.sum(), llama_valid.sum(), gemma_valid.sum()],
        [N, N, N]
    )
    log(f"  χ² = {chi2_3:.4f},  df = {dof_3},  p = {p_3:.2e}")
    log(f"  {'SIGNIFICANT (p < 0.001)' if p_3 < 0.001 else 'NOT significant'}")
    records.append({"test": "Chi-square three-way", "statistic": chi2_3, "p_value": p_3})

    # ── Test 3: Z-tests vs 0.5 baseline ──────────────────────────────────────
    log("\n" + "─" * 72)
    log("TEST 3 — One-sample Z-tests vs 50% baseline")
    for name, p_hat in [("Mistral", p_mis), ("Llama", p_lla), ("Gemma", p_gem)]:
        z, pz = proportion_z_test(p_hat, N)
        sig = "ABOVE 50%" if (z > 0 and pz < 0.01) else "BELOW 50%" if (z < 0 and pz < 0.01) else "NOT SIG"
        log(f"  {name}: p̂={p_hat:.3f}  Z={z:.2f}  p={pz:.4f}  → {sig}")
        records.append({"test": f"Z-test {name} vs 0.5", "statistic": z, "p_value": pz})

    # ── Test 4: Cohen's h (effect sizes) ──────────────────────────────────────
    log("\n" + "─" * 72)
    log("TEST 4 — Cohen's h (effect size for proportion differences)")
    pairs = [("Mistral", p_mis, "Llama", p_lla),
             ("Mistral", p_mis, "Gemma", p_gem),
             ("Llama",   p_lla, "Gemma", p_gem)]
    for n1, p1, n2, p2 in pairs:
        h = abs(cohens_h(p1, p2))
        mag = "very large (>2.0)" if h > 2 else "large (>0.8)" if h > 0.8 else "medium (>0.5)" if h > 0.5 else "small"
        log(f"  {n1} vs {n2}: h = {h:.3f}  → {mag}")
        records.append({"test": f"Cohen's h {n1} vs {n2}", "statistic": h, "p_value": None})

    # ── Test 5: Bootstrap CIs ──────────────────────────────────────────────────
    log("\n" + "─" * 72)
    log("TEST 5 — 95% Bootstrap Confidence Intervals (10 000 resamples)")
    np.random.seed(42)
    for name, valid_mask in [("Mistral", mistral_valid), ("Llama", llama_valid), ("Gemma", gemma_valid)]:
        lo, hi = bootstrap_ci(valid_mask)
        log(f"  {name}: {valid_mask.mean():.3f}  [{lo:.3f}, {hi:.3f}]")
        records.append({"test": f"Bootstrap CI {name}", "statistic": valid_mask.mean(), "p_value": f"[{lo:.3f},{hi:.3f}]"})

    # ── Test 6: Romanization paired test (McNemar) ────────────────────────────
    try:
        dfR, fR = load_roman()
        log("\n" + "─" * 72)
        log("TEST 6 — McNemar's Test: Llama native vs romanized (Exp 6)")
        log(f"  Source: {os.path.basename(fR)}")
        nat_v = dfR['llama_native_pred'].isin(VALID_LABELS)
        rom_v = dfR['llama_roman_pred'].isin(VALID_LABELS)
        chi2_r, p_r = mcnemar_test(nat_v, rom_v)
        delta = rom_v.mean() - nat_v.mean()
        log(f"  Native validity:    {nat_v.mean():.1%}  ({nat_v.sum()}/{len(dfR)})")
        log(f"  Romanized validity: {rom_v.mean():.1%}  ({rom_v.sum()}/{len(dfR)})")
        log(f"  Delta:              {delta:+.1%}")
        log(f"  χ² = {chi2_r:.4f},  p = {p_r:.2e}")
        log(f"  {'SIGNIFICANT (p < 0.001)' if p_r < 0.001 else 'NOT significant'}")
        records.append({"test": "McNemar Llama native vs roman", "statistic": chi2_r, "p_value": p_r})
    except FileNotFoundError as e:
        log(f"\n  [SKIP] {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(RESULTS_DIR, f"statistical_tests_{ts}.txt")
    csv_path = os.path.join(RESULTS_DIR, f"statistical_tests_{ts}.csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
