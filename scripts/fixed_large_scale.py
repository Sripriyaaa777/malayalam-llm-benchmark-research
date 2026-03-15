"""
Fixed Large-Scale Experiment: 500 samples
Properly handles prediction validation
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data
from datetime import datetime

from groq import Groq
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

groq_client = Groq(api_key=groq_key)
mistral_client = MistralClient(api_key=mistral_key)

FIVE_SHOT_EXAMPLES = """Here are some examples of Malayalam-English code-mixed sentiment analysis:

Example 1 (Positive):
Text: "ഈ പടം കിടു ആണ്! Climax scene വേറെ level! Totally paisa vasool. Must watch!"
Sentiment: Positive

Example 2 (Negative):
Text: "Bore adichu mari. First half okay aarunnu but second half വല്യ waste. Time and money പോയി."
Sentiment: Negative

Example 3 (Mixed_feelings):
Text: "Songs കൊള്ളാം, bgm നന്നായി. But story weak aanu. Average padam എന്ന് പറയാം."
Sentiment: Mixed_feelings

Example 4 (Positive):
Text: "Adipoli performance! Hero mass aanu. Interval scene kollaam. Theatre il energy vere level!"
Sentiment: Positive

Example 5 (Negative):
Text: "Trailer kandappo excited aayi but padam disappointment aayi. Expected ഒന്നും illatha feel."
Sentiment: Negative
"""

VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

def create_5shot_prompt(text):
    prompt = f"""You are analyzing sentiment in Malayalam-English code-mixed text (Manglish).

{FIVE_SHOT_EXAMPLES}

Now analyze this text:
Text: "{text}"

Classify the sentiment as:
- Positive
- Negative
- Mixed_feelings

Respond with ONLY ONE WORD - the label. Nothing else."""
    return prompt

def clean_prediction(raw_pred):
    """Extract valid label from model response"""
    if not raw_pred or raw_pred == "ERROR":
        return "INVALID"
    
    # Check if response contains a valid label
    for label in VALID_LABELS:
        if label in raw_pred:
            return label
    
    # Check for lowercase or variations
    raw_lower = raw_pred.lower()
    if "positive" in raw_lower:
        return "Positive"
    elif "negative" in raw_lower:
        return "Negative"
    elif "mixed" in raw_lower:
        return "Mixed_feelings"
    
    return "INVALID"

def test_llama(text):
    try:
        prompt = create_5shot_prompt(text)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        result = completion.choices[0].message.content.strip()
        return clean_prediction(result)
    except Exception as e:
        return "INVALID"

def test_mistral(text):
    try:
        prompt = create_5shot_prompt(text)
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=20,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return clean_prediction(result)
    except Exception as e:
        return "INVALID"

def calculate_metrics(df):
    """Calculate comprehensive metrics"""
    labels = VALID_LABELS
    
    metrics = {'overall': {}, 'per_label': {}}
    
    # Overall accuracy
    df['llama_correct'] = df['llama_pred'] == df['true_label']
    df['mistral_correct'] = df['mistral_pred'] == df['true_label']
    
    metrics['overall']['llama_acc'] = df['llama_correct'].mean() * 100
    metrics['overall']['mistral_acc'] = df['mistral_correct'].mean() * 100
    metrics['overall']['llama_count'] = df['llama_correct'].sum()
    metrics['overall']['mistral_count'] = df['mistral_correct'].sum()
    metrics['overall']['total'] = len(df)
    
    # Per-label metrics with Precision, Recall, F1
    for label in labels:
        label_df = df[df['true_label'] == label]
        if len(label_df) > 0:
            # True positives
            llama_tp = len(df[(df['true_label'] == label) & (df['llama_pred'] == label)])
            mistral_tp = len(df[(df['true_label'] == label) & (df['mistral_pred'] == label)])
            
            # False positives
            llama_fp = len(df[(df['true_label'] != label) & (df['llama_pred'] == label)])
            mistral_fp = len(df[(df['true_label'] != label) & (df['mistral_pred'] == label)])
            
            # False negatives
            llama_fn = len(df[(df['true_label'] == label) & (df['llama_pred'] != label)])
            mistral_fn = len(df[(df['true_label'] == label) & (df['mistral_pred'] != label)])
            
            # Calculate metrics
            llama_precision = llama_tp / (llama_tp + llama_fp) if (llama_tp + llama_fp) > 0 else 0
            mistral_precision = mistral_tp / (mistral_tp + mistral_fp) if (mistral_tp + mistral_fp) > 0 else 0
            
            llama_recall = llama_tp / (llama_tp + llama_fn) if (llama_tp + llama_fn) > 0 else 0
            mistral_recall = mistral_tp / (mistral_tp + mistral_fn) if (mistral_tp + mistral_fn) > 0 else 0
            
            llama_f1 = 2 * (llama_precision * llama_recall) / (llama_precision + llama_recall) if (llama_precision + llama_recall) > 0 else 0
            mistral_f1 = 2 * (mistral_precision * mistral_recall) / (mistral_precision + mistral_recall) if (mistral_precision + mistral_recall) > 0 else 0
            
            metrics['per_label'][label] = {
                'count': len(label_df),
                'llama_acc': (label_df['llama_correct'].sum() / len(label_df)) * 100,
                'mistral_acc': (label_df['mistral_correct'].sum() / len(label_df)) * 100,
                'llama_correct': label_df['llama_correct'].sum(),
                'mistral_correct': label_df['mistral_correct'].sum(),
                'llama_precision': llama_precision * 100,
                'llama_recall': llama_recall * 100,
                'llama_f1': llama_f1 * 100,
                'mistral_precision': mistral_precision * 100,
                'mistral_recall': mistral_recall * 100,
                'mistral_f1': mistral_f1 * 100
            }
    
    return metrics

def print_progress_bar(current, total, start_time):
    """Print progress bar with ETA"""
    progress = current / total
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    eta_seconds = (elapsed / current * (total - current)) if current > 0 else 0
    eta_minutes = eta_seconds / 60
    
    print(f'\r[{bar}] {current}/{total} ({progress*100:.1f}%) - ETA: {eta_minutes:.1f} min', end='', flush=True)

def main():
    print("=" * 80)
    print("LARGE-SCALE EXPERIMENT: 500 SAMPLES (FIXED)")
    print("5-Shot Prompting with Llama 3.3 70B & Mistral Large")
    print("=" * 80)
    
    df = load_malayalam_sentiment_data()
    
    print("\n✓ Selecting 500 balanced samples...")
    
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(250, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(150, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(100, random_state=42).to_dict('records'))
    
    print(f"   - 250 Positive")
    print(f"   - 150 Negative")  
    print(f"   - 100 Mixed_feelings")
    
    results = []
    start_time = datetime.now()
    
    print(f"\n⏰ Started at: {start_time.strftime('%H:%M:%S')}")
    print(f"⏰ Estimated time: 15-20 minutes\n")
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        llama_pred = test_llama(text)
        mistral_pred = test_mistral(text)
        
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred
        })
        
        print_progress_bar(idx, 500, start_time)
        
        # Save progress every 50 samples
        if idx % 50 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("../results/large_scale_progress.csv", index=False)
    
    print("\n\n✓ Experiment complete!")
    
    results_df = pd.DataFrame(results)
    
    # Filter to valid predictions only
    clean_df = results_df[
        (results_df['llama_pred'].isin(VALID_LABELS)) &
        (results_df['mistral_pred'].isin(VALID_LABELS))
    ].copy()
    
    invalid_count = len(results_df) - len(clean_df)
    print(f"✓ Valid predictions: {len(clean_df)}/500")
    if invalid_count > 0:
        print(f"⚠️ Filtered out {invalid_count} invalid predictions")
    
    # Calculate metrics
    metrics = calculate_metrics(clean_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/large_scale_500_fixed_{timestamp}.csv"
    clean_df.to_csv(output_path, index=False)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # RESULTS
    print("\n" + "=" * 80)
    print("FINAL RESULTS - 500 SAMPLES")
    print("=" * 80)
    
    overall = metrics['overall']
    print(f"\n📊 OVERALL ACCURACY ({overall['total']} samples):")
    print(f"  Llama 3.3 70B:  {overall['llama_acc']:.2f}% ({overall['llama_count']}/{overall['total']})")
    print(f"  Mistral Large:  {overall['mistral_acc']:.2f}% ({overall['mistral_count']}/{overall['total']})")
    
    print("\n" + "=" * 80)
    print("PER-LABEL PERFORMANCE")
    print("=" * 80)
    
    for label in VALID_LABELS:
        if label in metrics['per_label']:
            m = metrics['per_label'][label]
            print(f"\n{label} ({m['count']} samples):")
            print(f"  Llama:   Acc={m['llama_acc']:.1f}%  P={m['llama_precision']:.1f}%  R={m['llama_recall']:.1f}%  F1={m['llama_f1']:.1f}%")
            print(f"  Mistral: Acc={m['mistral_acc']:.1f}%  P={m['mistral_precision']:.1f}%  R={m['mistral_recall']:.1f}%  F1={m['mistral_f1']:.1f}%")
    
    # Macro F1
    llama_f1_avg = sum(metrics['per_label'][l]['llama_f1'] for l in metrics['per_label']) / len(metrics['per_label'])
    mistral_f1_avg = sum(metrics['per_label'][l]['mistral_f1'] for l in metrics['per_label']) / len(metrics['per_label'])
    
    print("\n" + "=" * 80)
    print("MACRO-AVERAGED F1-SCORES")
    print("=" * 80)
    print(f"  Llama:   {llama_f1_avg:.2f}%")
    print(f"  Mistral: {mistral_f1_avg:.2f}%")
    
    print(f"\n⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Saved to: {output_path}")
    print("\n" + "=" * 80)
    print("🎉 EXPERIMENT COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()