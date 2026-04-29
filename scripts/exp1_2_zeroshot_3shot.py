"""
Experiments 1 & 2 — Zero-shot (0-shot) and 3-shot on 100 samples.
Produces: results/exp1_2_zeroshot_3shot_<timestamp>.csv
"""
import os, sys, time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_malayalam_sentiment_data, VALID_LABELS
from api_clients import (
    _get_groq_client, _get_mistral_client,
    predict_llama, predict_mistral, predict_gemma, clean_prediction
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

THREE_SHOT_EXAMPLES = """Here are three examples:

Example 1:
Text: "Super movie! Ikka pwoli aayittundu. Must watch!"
Sentiment: Positive

Example 2:
Text: "Bore aayipoyi. Waste of time and money. Padam nannaayilla."
Sentiment: Negative

Example 3:
Text: "Songs kollam but story weak aanu. Climax okke aayi."
Sentiment: Mixed_feelings
"""

def make_zero_prompt(text):
    system = (
        "You are a sentiment classifier for Malayalam-English code-mixed text. "
        "Classify as exactly one of: Positive, Negative, Mixed_feelings. "
        "Output ONLY the label word."
    )
    user = f'Text: "{text}"\nSentiment:'
    return system, user

def make_3shot_prompt(text):
    system = (
        "You are a sentiment classifier for Malayalam-English code-mixed text. "
        "Classify as exactly one of: Positive, Negative, Mixed_feelings. "
        "Output ONLY the label word.\n\n" + THREE_SHOT_EXAMPLES
    )
    user = f'Text: "{text}"\nSentiment:'
    return system, user

def call_groq_custom(client, model_id, system, user, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=20, temperature=0,
            )
            return clean_prediction(resp.choices[0].message.content.strip())
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(60 * (attempt + 1))
            else:
                time.sleep(2)
    return "INVALID"

def call_mistral_custom(client, sdk_ver, system, user, retries=3):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(retries):
        try:
            if sdk_ver == "new":
                resp = client.chat.complete(model="mistral-large-latest", messages=messages, max_tokens=20, temperature=0)
                return clean_prediction(resp.choices[0].message.content.strip())
            else:
                from mistralai.models.chat_completion import ChatMessage
                cms = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
                resp = client.chat(model="mistral-large-latest", messages=cms, max_tokens=20, temperature=0)
                return clean_prediction(resp.choices[0].message.content.strip())
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(60 * (attempt + 1))
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
    print("EXPERIMENTS 1 & 2 — Zero-shot vs 3-shot (100 samples)")
    print("=" * 70)

    df = load_malayalam_sentiment_data()

    # 100-sample balanced set (same random_state as original)
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
    print(f"\nStarted: {start.strftime('%H:%M:%S')}")
    print("Running 0-shot …\n")

    for i, (_, row) in enumerate(samples.iterrows(), 1):
        text = row['text']
        true_label = row['label']

        sys0, usr0 = make_zero_prompt(text)
        sys3, usr3 = make_3shot_prompt(text)

        results.append({
            'sample_id': i,
            'text': text,
            'true_label': true_label,
            'malayalam_density': row['malayalam_density'],
            'llama_0shot': call_groq_custom(groq, "llama-3.3-70b-versatile", sys0, usr0),
            'mistral_0shot': call_mistral_custom(mistral_client, sdk_ver, sys0, usr0),
            'llama_3shot': call_groq_custom(groq, "llama-3.3-70b-versatile", sys3, usr3),
            'mistral_3shot': call_mistral_custom(mistral_client, sdk_ver, sys3, usr3),
        })
        progress(i, len(samples), start)

    print("\n\nDone!")
    rdf = pd.DataFrame(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULTS_DIR, f"exp1_2_zeroshot_3shot_{ts}.csv")
    rdf.to_csv(out, index=False)
    print(f"Saved → {out}")

    # Quick summary
    for col in ['llama_0shot', 'mistral_0shot', 'llama_3shot', 'mistral_3shot']:
        valid = rdf[col].isin(VALID_LABELS)
        e2e = (rdf[col] == rdf['true_label']).mean()
        print(f"  {col}: valid={valid.sum()}/100  e2e_acc={e2e:.1%}")

    duration = (datetime.now() - start).total_seconds() / 60
    print(f"\nTotal time: {duration:.1f} min")

if __name__ == "__main__":
    main()
