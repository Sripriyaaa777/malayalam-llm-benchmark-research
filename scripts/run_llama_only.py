"""
run_llama_only.py — 3 active Llama models on Groq, 500 samples, 5-shot.

As of April 2026, only these Llama models are active on Groq:
  - llama-3.3-70b-versatile              (Llama 3.3, 70B)
  - llama-3.1-8b-instant                 (Llama 3.1, 8B)
  - meta-llama/llama-4-scout-17b-16e-instruct  (Llama 4, MoE 17B active / 109B total)

All older models (llama3-8b-8192, llama3-70b-8192, gemma2-9b-it, etc.)
are deprecated and return 400 errors.

Paper angle: Llama generation comparison (3.1 vs 3.3 vs 4) + size comparison.

Estimated time: ~60–70 min
Produces: results/run_llama_<timestamp>.csv
"""
import os, sys, re, time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

MODEL_A = "llama-3.3-70b-versatile"                     # Llama 3.3 — 70B
MODEL_B = "llama-3.1-8b-instant"                        # Llama 3.1 — 8B
MODEL_C = "meta-llama/llama-4-scout-17b-16e-instruct"   # Llama 4   — 17B active MoE

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a sentiment classifier for Malayalam-English code-mixed text (Manglish). "
    "Classify the sentiment as exactly one of: Positive, Negative, Mixed_feelings. "
    "Output ONLY the label word and nothing else.\n\n"
    "Example 1:\nText: \"ഈ പടം കിടു ആണ്! Climax scene വേറെ level! Totally paisa vasool.\"\nSentiment: Positive\n\n"
    "Example 2:\nText: \"Bore adichu mari. Second half വല്യ waste. Time and money പോയി.\"\nSentiment: Negative\n\n"
    "Example 3:\nText: \"Songs കൊള്ളാം, bgm നന്നായി. But story weak aanu.\"\nSentiment: Mixed_feelings\n\n"
    "Example 4:\nText: \"Adipoli performance! Hero mass aanu. Theatre il energy vere level!\"\nSentiment: Positive\n\n"
    "Example 5:\nText: \"Trailer kandappo excited aayi but padam disappointment aayi.\"\nSentiment: Negative"
)

def clean(raw):
    if not raw:
        return "INVALID"
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = raw.strip('\'"').strip()
    rl  = raw.lower()
    if rl == "positive":                                              return "Positive"
    if rl == "negative":                                              return "Negative"
    if rl in ("mixed_feelings", "mixed feelings", "mixed-feelings"): return "Mixed_feelings"
    for lbl in VALID_LABELS:
        if lbl in raw:        return lbl
        if lbl.lower() in rl: return lbl
    if "mixed" in rl and "feel" in rl: return "Mixed_feelings"
    if rl.startswith("positive"):      return "Positive"
    if rl.startswith("negative"):      return "Negative"
    if rl.startswith("mixed"):         return "Mixed_feelings"
    return "INVALID"

def call_groq(client, model_id, text, retries=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f'Text: "{text}"\nSentiment:'}
    ]
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=20,
                temperature=0,
            )
            return clean(r.choices[0].message.content.strip())
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err or "too many" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"\n  [429 on {model_id}] waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
            elif any(x in err.lower() for x in [
                "decommission", "not found", "deprecated",
                "experiencing high demand", "experimental"
            ]):
                print(f"\n  [unavailable: {model_id}] {err[:60]}", end="", flush=True)
                return "INVALID"
            else:
                print(f"\n  [error on {model_id}] {err[:80]}", end="", flush=True)
                time.sleep(5)
    return "INVALID"

def pbar(i, n, start):
    pct     = i / n
    filled  = int(40 * pct)
    bar     = '#' * filled + '.' * (40 - filled)
    elapsed = (datetime.now() - start).total_seconds()
    eta_min = (elapsed / i * (n - i)) / 60 if i > 0 else 0
    print(f'\r[{bar}] {i}/{n} ({pct*100:.1f}%) ETA {eta_min:.1f}min', end='', flush=True)

def main():
    print("=" * 65)
    print("LLAMA-ONLY RUN -- 500 samples, 5-shot")
    print(f"  {MODEL_A}  (Llama 3.3 - 70B)")
    print(f"  {MODEL_B}  (Llama 3.1 - 8B)")
    print(f"  {MODEL_C}")
    print("  (Llama 4 Scout - 17B active MoE)")
    print("Estimated time: ~60-70 min")
    print("=" * 65)

    from load_data import load_malayalam_sentiment_data
    df = load_malayalam_sentiment_data()
    samples = pd.concat([
        df[df['label'] == 'Positive'].sample(250, random_state=42),
        df[df['label'] == 'Negative'].sample(150, random_state=42),
        df[df['label'] == 'Mixed_feelings'].sample(100, random_state=42),
    ]).reset_index(drop=True)
    print(f"\nSamples: {len(samples)}  (250 Pos / 150 Neg / 100 Mix)\n")

    print("Initialising Groq client...")
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    print("  OK Groq ready\n")

    print("Verifying all 3 models...")
    test_text = "ഈ പടം നല്ലതാണ്"
    all_ok = True
    for model_id in [MODEL_A, MODEL_B, MODEL_C]:
        result = call_groq(client, model_id, test_text)
        ok = result != "INVALID"
        print(f"  {'OK' if ok else 'FAILED'} {model_id} -> {result}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("\n  One or more models failed. Aborting.")
        return
    print()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(RESULTS_DIR, f"run_llama_ckpt_{ts}.csv")
    out_path  = os.path.join(RESULTS_DIR, f"run_llama_{ts}.csv")

    results = []
    start   = datetime.now()
    print(f"Started: {start.strftime('%H:%M:%S')}")
    print("(1.5s between calls, 3s after each sample)\n")

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        text = str(row['text'])

        pred_a = call_groq(client, MODEL_A, text)
        time.sleep(1.5)
        pred_b = call_groq(client, MODEL_B, text)
        time.sleep(1.5)
        pred_c = call_groq(client, MODEL_C, text)

        results.append({
            'sample_id':         i,
            'text':              text,
            'true_label':        row['label'],
            'malayalam_density': row['malayalam_density'],
            'llama33_70b_pred':  pred_a,
            'llama31_8b_pred':   pred_b,
            'llama4_scout_pred': pred_c,
        })

        pbar(i, len(samples), start)

        if i % 25 == 0:
            pd.DataFrame(results).to_csv(ckpt_path, index=False)
            print(f"  [checkpoint saved at {i}]", end="", flush=True)

        time.sleep(3)

    print("\n\nDone! Saving final results...")
    rdf = pd.DataFrame(results)
    rdf.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    print("\n" + "=" * 65)
    print(f"{'Model':<24} {'Valid':>8} {'Validity':>10} {'Cond.Acc':>10} {'E2E.Acc':>10}")
    print("-" * 65)
    for name, col in [
        ("Llama 3.3 70B",  "llama33_70b_pred"),
        ("Llama 3.1 8B",   "llama31_8b_pred"),
        ("Llama 4 Scout",  "llama4_scout_pred"),
    ]:
        valid = rdf[col].isin(VALID_LABELS)
        vdf   = rdf[valid]
        cond  = (vdf[col] == vdf['true_label']).mean() if len(vdf) > 0 else 0
        e2e   = (rdf[col] == rdf['true_label']).mean()
        print(f"{name:<24} {valid.sum():>4}/500 {valid.mean():>9.1%} {cond:>10.1%} {e2e:>10.1%}")

    mins = (datetime.now() - start).total_seconds() / 60
    print(f"\nTotal time: {mins:.1f} min")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Final file: {out_path}")
    print(f"\nNext step: python generate_metrics_matrix.py")

if __name__ == "__main__":
    main()