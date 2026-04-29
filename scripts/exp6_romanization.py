"""
Experiment 6 — Romanization Control (paired design).
Transliterates Malayalam Unicode → Latin using aksharamukha, then re-runs
Llama 3.3 70B on the same 500 samples under identical 5-shot conditions.

Requires: pip install aksharamukha
Produces:  results/exp6_romanization_<timestamp>.csv
"""
import os, sys, time
from datetime import datetime
import pandas as pd
import glob

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_malayalam_sentiment_data, VALID_LABELS, compute_malayalam_density
from api_clients import _get_groq_client, predict_llama, clean_prediction

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Romanization ─────────────────────────────────────────────────────────────

def romanize_text(text):
    """Transliterate Malayalam Unicode chars to Latin (ISO 15919 via Aksharamukha)."""
    try:
        from aksharamukha import transliterate
        return transliterate.process("Malayalam", "ISO", text)
    except ImportError:
        raise ImportError(
            "aksharamukha not installed. Run: pip install aksharamukha"
        )
    except Exception:
        return text  # fallback: keep original if transliteration fails

# 5-shot examples also romanized for consistency
ROMAN_FIVE_SHOT = """Here are five examples of Malayalam-English code-mixed sentiment classification:

Example 1:
Text: "ee padam kidu aanu! Climax scene vere level! Totally paisa vasool. Must watch!"
Sentiment: Positive

Example 2:
Text: "Bore adichu mari. First half okay aarunnu but second half valya waste. Time and money poyi."
Sentiment: Negative

Example 3:
Text: "Songs kollaam, bgm nannaayi. But story weak aanu. Average padam ennu paranjaal."
Sentiment: Mixed_feelings

Example 4:
Text: "Adipoli performance! Hero mass aanu. Interval scene kollaam. Theatre il energy vere level!"
Sentiment: Positive

Example 5:
Text: "Trailer kandappo excited aayi but padam disappointment aayi. Expected onnum illatha feel."
Sentiment: Negative
"""

def make_roman_prompt(text):
    system = (
        "You are a sentiment classifier for Malayalam-English code-mixed text (Manglish). "
        "The Malayalam words have been romanized to Latin script. "
        "Classify as exactly one of: Positive, Negative, Mixed_feelings. "
        "Output ONLY the label word and nothing else.\n\n" + ROMAN_FIVE_SHOT
    )
    user = f'Text: "{text}"\nSentiment:'
    return system, user


def call_groq_roman(client, text, retries=3):
    system, user = make_roman_prompt(text)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=20, temperature=0,
            )
            return clean_prediction(resp.choices[0].message.content.strip())
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"\n  [Rate limit] waiting {wait}s…", end="", flush=True)
                time.sleep(wait)
            else:
                time.sleep(2)
    return "INVALID"


def progress(i, n, start):
    pct = i / n
    bar = '█' * int(40 * pct) + '░' * (40 - int(40 * pct))
    elapsed = (datetime.now() - start).total_seconds()
    eta = (elapsed / i * (n - i)) / 60 if i > 0 else 0
    print(f'\r[{bar}] {i}/{n} ({pct*100:.1f}%) ETA {eta:.1f}min', end='', flush=True)


def main():
    print("=" * 70)
    print("EXPERIMENT 6 — Romanization Control (Llama 3.3 70B, 500 samples)")
    print("=" * 70)

    # ── Load original 500-sample results to get the SAME samples ──────────────
    # Look for exp4_5 results first; fall back to legacy sample_results.csv
    candidates = sorted(glob.glob(os.path.join(RESULTS_DIR, "exp4_5_500sample_*.csv")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(RESULTS_DIR, "sample_results.csv")))
    if not candidates:
        print("ERROR: Run exp4_5_500sample.py first (Experiment 4/5 results not found).")
        sys.exit(1)

    src = candidates[-1]
    print(f"Loading original predictions from: {src}")
    orig_df = pd.read_csv(src)

    # We need: sample_id, text, true_label, llama_pred (original native script)
    # Determine llama column name
    llama_col = 'llama_pred' if 'llama_pred' in orig_df.columns else None
    if llama_col is None:
        print("ERROR: 'llama_pred' column not found in source file.")
        sys.exit(1)

    print(f"Loaded {len(orig_df)} samples.")

    # ── Romanize all texts ─────────────────────────────────────────────────────
    print("\nRomanizing texts (aksharamukha)…")
    orig_df['text_romanized'] = orig_df['text'].apply(romanize_text)
    orig_df['roman_density'] = orig_df['text_romanized'].apply(compute_malayalam_density)
    print(f"  Avg residual Malayalam density after romanization: "
          f"{orig_df['roman_density'].mean():.3f}")

    # ── Re-run Llama on romanized text ────────────────────────────────────────
    groq = _get_groq_client()
    results = []
    start = datetime.now()
    print(f"\nStarted: {start.strftime('%H:%M:%S')}\n")

    for i, (_, row) in enumerate(orig_df.iterrows(), 1):
        roman_pred = call_groq_roman(groq, row['text_romanized'])
        results.append({
            'sample_id': row['sample_id'],
            'text_original': row['text'],
            'text_romanized': row['text_romanized'],
            'true_label': row['true_label'],
            'malayalam_density': row.get('malayalam_density', compute_malayalam_density(row['text'])),
            'llama_native_pred': row[llama_col],
            'llama_roman_pred': roman_pred,
        })
        progress(i, len(orig_df), start)

    print("\n\nDone!")
    rdf = pd.DataFrame(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"exp6_romanization_{ts}.csv")
    rdf.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    native_valid = rdf['llama_native_pred'].isin(VALID_LABELS)
    roman_valid  = rdf['llama_roman_pred'].isin(VALID_LABELS)
    n = len(rdf)

    print("\n" + "=" * 70)
    print("ROMANIZATION EXPERIMENT — RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'Native':>10} {'Romanized':>12} {'Delta':>8}")
    print("-" * 70)
    print(f"{'Output Validity Rate':<35} {native_valid.mean():>9.1%} {roman_valid.mean():>11.1%} {roman_valid.mean()-native_valid.mean():>+7.1%}")

    # Conditional accuracy
    nat_cdf = rdf[native_valid]
    rom_cdf = rdf[roman_valid]
    nat_cond = (nat_cdf['llama_native_pred'] == nat_cdf['true_label']).mean() if len(nat_cdf) > 0 else 0
    rom_cond = (rom_cdf['llama_roman_pred']  == rom_cdf['true_label']).mean()  if len(rom_cdf) > 0 else 0
    print(f"{'Conditional Accuracy (valid only)':<35} {nat_cond:>9.1%} {rom_cond:>11.1%} {rom_cond-nat_cond:>+7.1%}")

    # E2E
    nat_e2e = (rdf['llama_native_pred'] == rdf['true_label']).mean()
    rom_e2e = (rdf['llama_roman_pred']  == rdf['true_label']).mean()
    print(f"{'End-to-End Accuracy':<35} {nat_e2e:>9.1%} {rom_e2e:>11.1%} {rom_e2e-nat_e2e:>+7.1%}")

    print(f"\nValidity: {native_valid.sum()}/{n} → {roman_valid.sum()}/{n}  (+{roman_valid.sum()-native_valid.sum()} pp)")

    duration = (datetime.now() - start).total_seconds() / 60
    print(f"\nTotal time: {duration:.1f} min")

if __name__ == "__main__":
    main()
