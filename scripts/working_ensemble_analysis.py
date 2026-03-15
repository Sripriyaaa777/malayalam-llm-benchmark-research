"""
Working Ensemble + Error Analysis Script
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data
from datetime import datetime
import re

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
        return f"ERROR"

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
        return f"ERROR"

def calculate_malayalam_percentage(text):
    malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', text))
    total_chars = len(text.replace(' ', ''))
    return (malayalam_chars / total_chars * 100) if total_chars > 0 else 0

def ensemble_predict(llama_pred, mistral_pred):
    """If both agree, use that. If disagree, use Llama (better model)"""
    if llama_pred == mistral_pred:
        return llama_pred, "agreement"
    else:
        return llama_pred, "llama_fallback"

def create_confusion_matrix(df):
    """Create confusion matrix from dataframe"""
    labels = ['Positive', 'Negative', 'Mixed_feelings']
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    
    for _, row in df.iterrows():
        true = row['true_label']
        pred = row['ensemble_pred']
        if true in labels and pred in labels:
            matrix[true][pred] += 1
    
    return matrix

def print_confusion_matrix(matrix):
    labels = ['Positive', 'Negative', 'Mixed_feelings']
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(f"{'True/Pred':<15} {'Positive':<15} {'Negative':<15} {'Mixed':<15}")
    print("-" * 70)
    for true_label in labels:
        row = f"{true_label:<15}"
        for pred_label in labels:
            count = matrix[true_label][pred_label]
            row += f"{count:<15}"
        print(row)

def main():
    print("=" * 80)
    print("ENSEMBLE + ERROR ANALYSIS")
    print("=" * 80)
    
    df = load_malayalam_sentiment_data()
    
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(50, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(30, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(20, random_state=42).to_dict('records'))
    
    print(f"\n✓ Selected 100 samples")
    
    results = []
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("RUNNING ENSEMBLE EXPERIMENT...")
    print("=" * 80)
    
    for idx, sample in enumerate(samples, 1):
        if idx % 25 == 0 or idx == 1:
            print(f"[{idx}/100] Processing...")
        
        text = sample['text']
        true_label = sample['label']
        
        llama_pred = test_llama(text)
        mistral_pred = test_mistral(text)
        ensemble_pred, ensemble_method = ensemble_predict(llama_pred, mistral_pred)
        
        results.append({
            'text': text,
            'true_label': true_label,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred,
            'ensemble_pred': ensemble_pred,
            'ensemble_method': ensemble_method
        })
    
    results_df = pd.DataFrame(results)
    
    # Remove only actual ERROR cases (not valid predictions)
    clean_df = results_df[
        (results_df['llama_pred'] != 'ERROR') &
        (results_df['mistral_pred'] != 'ERROR')
    ].copy()
    
    print(f"\n✓ Experiment complete! Valid predictions: {len(clean_df)}/100")
    
    # Calculate correctness
    clean_df['llama_correct'] = clean_df['llama_pred'] == clean_df['true_label']
    clean_df['mistral_correct'] = clean_df['mistral_pred'] == clean_df['true_label']
    clean_df['ensemble_correct'] = clean_df['ensemble_pred'] == clean_df['true_label']
    
    # Accuracies
    llama_acc = clean_df['llama_correct'].mean() * 100
    mistral_acc = clean_df['mistral_correct'].mean() * 100
    ensemble_acc = clean_df['ensemble_correct'].mean() * 100
    
    # Agreement stats
    agreement = (clean_df['llama_pred'] == clean_df['mistral_pred']).sum()
    agreement_rate = agreement / len(clean_df) * 100
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/ensemble_analysis_{timestamp}.csv"
    clean_df.to_csv(output_path, index=False)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # RESULTS
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\n📊 OVERALL ACCURACY ({len(clean_df)} samples):")
    print(f"  Llama 3.3 70B:    {llama_acc:.1f}% ({clean_df['llama_correct'].sum()}/{len(clean_df)})")
    print(f"  Mistral Large:    {mistral_acc:.1f}% ({clean_df['mistral_correct'].sum()}/{len(clean_df)})")
    print(f"  🎯 ENSEMBLE:       {ensemble_acc:.1f}% ({clean_df['ensemble_correct'].sum()}/{len(clean_df)})")
    
    if ensemble_acc > llama_acc:
        print(f"\n🎉 Ensemble improves by {ensemble_acc - llama_acc:.1f}%!")
    elif ensemble_acc == llama_acc:
        print(f"\n✓ Ensemble matches best single model")
    else:
        print(f"\n⚠️ Ensemble slightly lower (likely due to Mistral pulling down)")
    
    print(f"\n🤝 Model Agreement:")
    print(f"  Models agree: {agreement}/{len(clean_df)} times ({agreement_rate:.1f}%)")
    
    # Confusion Matrix
    matrix = create_confusion_matrix(clean_df)
    print_confusion_matrix(matrix)
    
    # Error Analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    errors = clean_df[~clean_df['ensemble_correct']].copy()
    print(f"\n📉 Total errors: {len(errors)} ({len(errors)/len(clean_df)*100:.1f}%)")
    
    # Errors by label
    print(f"\n❌ Errors by True Label:")
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        label_df = clean_df[clean_df['true_label'] == label]
        label_errors = errors[errors['true_label'] == label]
        if len(label_df) > 0:
            print(f"  {label}: {len(label_errors)}/{len(label_df)} ({len(label_errors)/len(label_df)*100:.1f}% error rate)")
    
    # Sample errors
    if len(errors) > 0:
        print(f"\n🔍 Sample Errors:")
        print("=" * 80)
        for idx, row in errors.head(5).iterrows():
            print(f"\n{idx+1}. Text: {row['text'][:65]}...")
            print(f"   True: {row['true_label']} | Predicted: {row['ensemble_pred']}")
            print(f"   (Llama: {row['llama_pred']}, Mistral: {row['mistral_pred']})")
    
    print(f"\n⏰ Time: {duration:.1f} minutes")
    print(f"✓ Saved to: {output_path}")
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()