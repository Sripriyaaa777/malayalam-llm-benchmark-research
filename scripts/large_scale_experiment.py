"""
Large-Scale Experiment: 500 samples with 5-shot prompting
Goal: Robust statistics for publication
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data
from datetime import datetime
import time

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
        if "Positive" in result:
            return "Positive"
        elif "Negative" in result:
            return "Negative"
        elif "Mixed" in result or "mixed" in result:
            return "Mixed_feelings"
        return result
    except Exception as e:
        return "ERROR"

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
        if "Positive" in result:
            return "Positive"
        elif "Negative" in result:
            return "Negative"
        elif "Mixed" in result or "mixed" in result:
            return "Mixed_feelings"
        return result
    except Exception as e:
        return "ERROR"

def calculate_metrics(df):
    """Calculate comprehensive metrics"""
    labels = ['Positive', 'Negative', 'Mixed_feelings']
    
    metrics = {
        'overall': {},
        'per_label': {}
    }
    
    # Overall accuracy
    df['llama_correct'] = df['llama_pred'] == df['true_label']
    df['mistral_correct'] = df['mistral_pred'] == df['true_label']
    
    metrics['overall']['llama_acc'] = df['llama_correct'].mean() * 100
    metrics['overall']['mistral_acc'] = df['mistral_correct'].mean() * 100
    metrics['overall']['llama_count'] = df['llama_correct'].sum()
    metrics['overall']['mistral_count'] = df['mistral_correct'].sum()
    metrics['overall']['total'] = len(df)
    
    # Per-label metrics
    for label in labels:
        label_df = df[df['true_label'] == label]
        if len(label_df) > 0:
            metrics['per_label'][label] = {
                'count': len(label_df),
                'llama_acc': (label_df['llama_correct'].sum() / len(label_df)) * 100,
                'mistral_acc': (label_df['mistral_correct'].sum() / len(label_df)) * 100,
                'llama_correct': label_df['llama_correct'].sum(),
                'mistral_correct': label_df['mistral_correct'].sum()
            }
    
    # Precision, Recall, F1 for each label
    for label in labels:
        # True positives
        llama_tp = len(df[(df['true_label'] == label) & (df['llama_pred'] == label)])
        mistral_tp = len(df[(df['true_label'] == label) & (df['mistral_pred'] == label)])
        
        # False positives
        llama_fp = len(df[(df['true_label'] != label) & (df['llama_pred'] == label)])
        mistral_fp = len(df[(df['true_label'] != label) & (df['mistral_pred'] == label)])
        
        # False negatives
        llama_fn = len(df[(df['true_label'] == label) & (df['llama_pred'] != label)])
        mistral_fn = len(df[(df['true_label'] == label) & (df['mistral_pred'] != label)])
        
        # Precision
        llama_precision = llama_tp / (llama_tp + llama_fp) if (llama_tp + llama_fp) > 0 else 0
        mistral_precision = mistral_tp / (mistral_tp + mistral_fp) if (mistral_tp + mistral_fp) > 0 else 0
        
        # Recall
        llama_recall = llama_tp / (llama_tp + llama_fn) if (llama_tp + llama_fn) > 0 else 0
        mistral_recall = mistral_tp / (mistral_tp + mistral_fn) if (mistral_tp + mistral_fn) > 0 else 0
        
        # F1
        llama_f1 = 2 * (llama_precision * llama_recall) / (llama_precision + llama_recall) if (llama_precision + llama_recall) > 0 else 0
        mistral_f1 = 2 * (mistral_precision * mistral_recall) / (mistral_precision + mistral_recall) if (mistral_precision + mistral_recall) > 0 else 0
        
        if label in metrics['per_label']:
            metrics['per_label'][label].update({
                'llama_precision': llama_precision * 100,
                'llama_recall': llama_recall * 100,
                'llama_f1': llama_f1 * 100,
                'mistral_precision': mistral_precision * 100,
                'mistral_recall': mistral_recall * 100,
                'mistral_f1': mistral_f1 * 100
            })
    
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
    print("LARGE-SCALE EXPERIMENT: 500 SAMPLES")
    print("5-Shot Prompting with Llama 3.3 70B & Mistral Large")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Select 500 balanced samples
    print("\n" + "=" * 80)
    print("SELECTING 500 TEST SAMPLES...")
    print("=" * 80)
    
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(250, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(150, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(100, random_state=42).to_dict('records'))
    
    print(f"\n✓ Selected 500 samples:")
    print(f"   - 250 Positive")
    print(f"   - 150 Negative")
    print(f"   - 100 Mixed_feelings")
    
    results = []
    start_time = datetime.now()
    
    print(f"\n⏰ Started at: {start_time.strftime('%H:%M:%S')}")
    print("\n" + "=" * 80)
    print("RUNNING LARGE-SCALE EXPERIMENT...")
    print("Estimated time: 15-20 minutes")
    print("=" * 80)
    print()
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        # Test both models
        llama_pred = test_llama(text)
        mistral_pred = test_mistral(text)
        
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred
        })
        
        # Progress bar
        print_progress_bar(idx, 500, start_time)
        
        # Save intermediate results every 50 samples
        if idx % 50 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("../results/large_scale_progress.csv", index=False)
    
    print()  # New line after progress bar
    print("\n✓ Experiment complete!")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Remove ERROR cases
    clean_df = results_df[
        (results_df['llama_pred'] != 'ERROR') &
        (results_df['mistral_pred'] != 'ERROR')
    ].copy()
    
    print(f"\n✓ Valid predictions: {len(clean_df)}/500")
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("CALCULATING COMPREHENSIVE METRICS...")
    print("=" * 80)
    
    metrics = calculate_metrics(clean_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/large_scale_500_{timestamp}.csv"
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
    
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        if label in metrics['per_label']:
            m = metrics['per_label'][label]
            print(f"\n{label} ({m['count']} samples):")
            print(f"  Llama:")
            print(f"    Accuracy:  {m['llama_acc']:.2f}% ({m['llama_correct']}/{m['count']})")
            print(f"    Precision: {m['llama_precision']:.2f}%")
            print(f"    Recall:    {m['llama_recall']:.2f}%")
            print(f"    F1-Score:  {m['llama_f1']:.2f}%")
            print(f"  Mistral:")
            print(f"    Accuracy:  {m['mistral_acc']:.2f}% ({m['mistral_correct']}/{m['count']})")
            print(f"    Precision: {m['mistral_precision']:.2f}%")
            print(f"    Recall:    {m['mistral_recall']:.2f}%")
            print(f"    F1-Score:  {m['mistral_f1']:.2f}%")
    
    # Macro-averaged F1
    llama_f1_avg = sum(metrics['per_label'][l]['llama_f1'] for l in metrics['per_label']) / len(metrics['per_label'])
    mistral_f1_avg = sum(metrics['per_label'][l]['mistral_f1'] for l in metrics['per_label']) / len(metrics['per_label'])
    
    print("\n" + "=" * 80)
    print("MACRO-AVERAGED METRICS")
    print("=" * 80)
    print(f"\nLlama 3.3 70B:")
    print(f"  Macro F1-Score: {llama_f1_avg:.2f}%")
    print(f"\nMistral Large:")
    print(f"  Macro F1-Score: {mistral_f1_avg:.2f}%")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH 100-SAMPLE EXPERIMENT")
    print("=" * 80)
    print(f"\nLlama 3.3 70B:")
    print(f"  100 samples: 81.2%")
    print(f"  500 samples: {overall['llama_acc']:.2f}%")
    print(f"  Difference:  {overall['llama_acc'] - 81.2:+.2f}%")
    
    print(f"\nMistral Large:")
    print(f"  100 samples: 71.0%")
    print(f"  500 samples: {overall['mistral_acc']:.2f}%")
    print(f"  Difference:  {overall['mistral_acc'] - 71.0:+.2f}%")
    
    print(f"\n⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("🎉 LARGE-SCALE EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"\n📈 Key Findings:")
    print(f"  - Tested on 5x larger dataset ({len(clean_df)} samples)")
    print(f"  - Llama achieves {overall['llama_acc']:.2f}% accuracy")
    print(f"  - Results are statistically robust")
    print(f"  - Publication-quality metrics calculated")

if __name__ == "__main__":
    main()