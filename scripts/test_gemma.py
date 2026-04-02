"""
Test Gemma 2 on Malayalam-English sentiment analysis
Quick test to see if it handles Malayalam script like Mistral or fails like Llama
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data
from datetime import datetime

from groq import Groq

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

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
    
    for label in VALID_LABELS:
        if label in raw_pred:
            return label
    
    raw_lower = raw_pred.lower()
    if "positive" in raw_lower:
        return "Positive"
    elif "negative" in raw_lower:
        return "Negative"
    elif "mixed" in raw_lower:
        return "Mixed_feelings"
    
    return "INVALID"

def test_gemma(text):
    try:
        prompt = create_5shot_prompt(text)
        completion = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        result = completion.choices[0].message.content.strip()
        return clean_prediction(result)
    except Exception as e:
        return "INVALID"

def main():
    print("=" * 80)
    print("GEMMA 2 TEST - Malayalam Script Handling")
    print("Testing if Gemma (Google) handles Malayalam like Mistral or fails like Llama")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Use SAME 500 samples as before (for fair comparison)
    print("\n✓ Selecting same 500 samples used in previous experiments...")
    
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
    print(f"⏰ Estimated time: 10-15 minutes\n")
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        gemma_pred = test_gemma(text)
        
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'gemma_pred': gemma_pred
        })
        
        # Progress indicator
        if idx % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"[{idx}/500] Processed... ({elapsed:.1f} min elapsed)")
    
    print(f"\n✓ Experiment complete!")
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/gemma_500_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    # Analyze
    valid_df = results_df[results_df['gemma_pred'].isin(VALID_LABELS)].copy()
    invalid_count = len(results_df) - len(valid_df)
    
    print("\n" + "=" * 80)
    print("RESULTS - GEMMA 2 (9B)")
    print("=" * 80)
    
    print(f"\n📊 Script Handling:")
    print(f"   Valid predictions: {len(valid_df)}/500 ({len(valid_df)/500*100:.1f}%)")
    print(f"   Invalid (failed): {invalid_count}/500 ({invalid_count/500*100:.1f}%)")
    
    if len(valid_df) > 0:
        valid_df['correct'] = valid_df['gemma_pred'] == valid_df['true_label']
        accuracy = valid_df['correct'].mean() * 100
        
        print(f"\n🎯 Accuracy: {accuracy:.2f}% ({valid_df['correct'].sum()}/{len(valid_df)})")
        
        # Per-label
        print(f"\n📊 Per-Label Accuracy:")
        for label in VALID_LABELS:
            label_df = valid_df[valid_df['true_label'] == label]
            if len(label_df) > 0:
                label_acc = (label_df['correct'].sum() / len(label_df)) * 100
                print(f"   {label}: {label_acc:.1f}% ({label_df['correct'].sum()}/{len(label_df)})")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH OTHER MODELS")
    print("=" * 80)
    print(f"\nScript Handling (% valid predictions):")
    print(f"   Mistral Large: 99.6% (498/500) ✅ EXCELLENT")
    if len(valid_df) >= 450:
        print(f"   Gemma 2:       {len(valid_df)/500*100:.1f}% ({len(valid_df)}/500) ✅ EXCELLENT")
    elif len(valid_df) >= 300:
        print(f"   Gemma 2:       {len(valid_df)/500*100:.1f}% ({len(valid_df)}/500) ⚠️ MODERATE")
    else:
        print(f"   Gemma 2:       {len(valid_df)/500*100:.1f}% ({len(valid_df)}/500) ❌ POOR")
    print(f"   Llama 3.3:     44.0% (220/500) ❌ POOR")
    
    if len(valid_df) > 0:
        print(f"\nAccuracy (on valid samples):")
        print(f"   Gemma 2:       {accuracy:.2f}%")
        print(f"   Mistral Large: 63.45%")
        print(f"   Llama 3.3:     81.2% (but only on 220 samples)")
    
    print(f"\n⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("🔑 KEY FINDING:")
    print("=" * 80)
    
    if len(valid_df) >= 450:
        print("✅ Gemma 2 handles Malayalam script WELL (like Mistral)")
        print("   → Confirms: Google models are robust for non-Latin scripts")
    elif len(valid_df) >= 300:
        print("⚠️ Gemma 2 has MODERATE script handling (between Llama and Mistral)")
        print("   → Interesting middle ground finding!")
    else:
        print("❌ Gemma 2 FAILS on Malayalam script (like Llama)")
        print("   → Confirms: Not all models handle non-Latin scripts well")

if __name__ == "__main__":
    main()