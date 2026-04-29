"""
Experiments 4 & 5 — Large-scale 500-sample evaluation (all 6 models, 5-shot).

Models:
  Groq:    Llama 3.3 70B | Llama 4 Scout | Gemma 2 9B | Qwen 3 32B
  Mistral: Mistral Large
  Google:  Gemini 2.5 Flash

Saves progress every 50 samples so a crash loses at most 50 rows.
Produces: results/exp4_5_500sample_<timestamp>.csv
"""
import os, sys, time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_malayalam_sentiment_data, VALID_LABELS
from api_clients import (
    _get_groq_client, _get_mistral_client, _get_gemini_client,
    predict_llama33, predict_llama4, predict_gemma, predict_qwen3,
    predict_mistral, predict_gemini,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def progress(i, n, start):
    pct = i/n
    bar = '█'*int(40*pct) + '░'*(40-int(40*pct))
    elapsed = (datetime.now()-start).total_seconds()
    eta = (elapsed/i*(n-i))/60 if i > 0 else 0
    print(f'\r[{bar}] {i}/{n} ({pct*100:.1f}%) ETA {eta:.1f}min', end='', flush=True)

def main():
    print("="*70)
    print("EXPERIMENTS 4 & 5 — 500 samples, all 6 models, 5-shot")
    print("="*70)
    print("\nModels: Llama 3.3 70B | Llama 4 Scout | Gemma 2 9B | Qwen 3 32B | Mistral Large | Gemini 2.5 Flash")
    print("Estimated time: ~60–80 minutes\n")

    df = load_malayalam_sentiment_data()
    samples = pd.concat([
        df[df['label']=='Positive'].sample(250, random_state=42),
        df[df['label']=='Negative'].sample(150, random_state=42),
        df[df['label']=='Mixed_feelings'].sample(100, random_state=42),
    ]).reset_index(drop=True)
    print(f"Sample set: {len(samples)} rows  (250 Pos / 150 Neg / 100 Mix)\n")

    # Initialise clients
    groq   = _get_groq_client()
    mis_c, mis_sdk = _get_mistral_client()
    gem    = _get_gemini_client()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = os.path.join(RESULTS_DIR, f"exp4_5_progress_{ts}.csv")
    out_path      = os.path.join(RESULTS_DIR, f"exp4_5_500sample_{ts}.csv")

    results = []
    start = datetime.now()
    print(f"Started: {start.strftime('%H:%M:%S')}\n")

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        text = row['text']
        results.append({
            'sample_id':          i,
            'text':               text,
            'true_label':         row['label'],
            'malayalam_density':  row['malayalam_density'],
            # Groq models
            'llama33_pred':       predict_llama33(groq,  text),
            'llama4_pred':        predict_llama4 (groq,  text),
            'gemma_pred':         predict_gemma  (groq,  text),
            'qwen3_pred':         predict_qwen3  (groq,  text),
            # Mistral
            'mistral_pred':       predict_mistral(mis_c, mis_sdk, text),
            # Gemini
            'gemini_pred':        predict_gemini (gem,   text),
        })
        progress(i, len(samples), start)

        if i % 50 == 0:
            pd.DataFrame(results).to_csv(progress_path, index=False)

    print("\n\nDone! Saving…")
    rdf = pd.DataFrame(results)
    rdf.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # Quick summary
    print("\n--- Quick summary (5-shot, 500 samples) ---")
    cols = {
        'Llama 3.3 70B':   'llama33_pred',
        'Llama 4 Scout':   'llama4_pred',
        'Gemma 2 9B':      'gemma_pred',
        'Qwen 3 32B':      'qwen3_pred',
        'Mistral Large':   'mistral_pred',
        'Gemini 2.5 Flash':'gemini_pred',
    }
    print(f"  {'Model':<20} {'Valid':>8} {'Validity':>10} {'Cond.Acc':>10} {'E2E.Acc':>10}")
    print("  " + "-"*62)
    for name, col in cols.items():
        valid = rdf[col].isin(VALID_LABELS)
        vdf   = rdf[valid]
        val   = valid.mean()
        cond  = (vdf[col]==vdf['true_label']).mean() if len(vdf)>0 else 0
        e2e   = (rdf[col]==rdf['true_label']).mean()
        print(f"  {name:<20} {valid.sum():>4}/500 {val:>9.1%} {cond:>10.1%} {e2e:>10.1%}")

    duration = (datetime.now()-start).total_seconds()/60
    print(f"\nTotal time: {duration:.1f} min")

if __name__ == "__main__":
    main()
