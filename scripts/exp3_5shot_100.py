"""
Experiment 3 — 5-shot improved prompting on 100 samples.
Produces: results/exp3_5shot_100_<timestamp>.csv
"""
import os, sys, time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_malayalam_sentiment_data, VALID_LABELS
from api_clients import (
    _get_groq_client, _get_mistral_client,
    predict_llama, predict_mistral, clean_prediction
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def progress(i, n, start):
    pct = i / n
    bar = '█' * int(40 * pct) + '░' * (40 - int(40 * pct))
    elapsed = (datetime.now() - start).total_seconds()
    eta = (elapsed / i * (n - i)) / 60 if i > 0 else 0
    print(f'\r[{bar}] {i}/{n} ({pct*100:.1f}%) ETA {eta:.1f}min', end='', flush=True)


def main():
    print("=" * 70)
    print("EXPERIMENT 3 — 5-shot Improved (100 samples)")
    print("=" * 70)

    df = load_malayalam_sentiment_data()

    samples = pd.concat([
        df[df['label'] == 'Positive'].sample(50, random_state=42),
        df[df['label'] == 'Negative'].sample(30, random_state=42),
        df[df['label'] == 'Mixed_feelings'].sample(20, random_state=42),
    ]).reset_index(drop=True)
    print(f"\nSample set: {len(samples)} rows  (50 Pos / 30 Neg / 20 Mix)")

    groq = _get_groq_client()
    mistral_client, sdk_ver = _get_mistral_client()

    results = []
    start = datetime.now()
    print(f"Started: {start.strftime('%H:%M:%S')}\n")

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        text = row['text']
        results.append({
            'sample_id': i,
            'text': text,
            'true_label': row['label'],
            'malayalam_density': row['malayalam_density'],
            'llama_pred': predict_llama(groq, text, shot="5shot"),
            'mistral_pred': predict_mistral(mistral_client, sdk_ver, text, shot="5shot"),
        })
        progress(i, len(samples), start)

    print("\n\nDone!")
    rdf = pd.DataFrame(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp3_5shot_100_{ts}.csv")
    rdf.to_csv(out, index=False)
    print(f"Saved → {out}")

    for model, col in [("Llama", "llama_pred"), ("Mistral", "mistral_pred")]:
        valid = rdf[col].isin(VALID_LABELS)
        cond_acc = (rdf[valid][col] == rdf[valid]['true_label']).mean() if valid.sum() > 0 else 0
        e2e = (rdf[col] == rdf['true_label']).mean()
        print(f"  {model}: valid={valid.sum()}/100  cond_acc={cond_acc:.1%}  e2e={e2e:.1%}")

    duration = (datetime.now() - start).total_seconds() / 60
    print(f"\nTotal time: {duration:.1f} min")

if __name__ == "__main__":
    main()
