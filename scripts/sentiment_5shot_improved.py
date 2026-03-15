"""
5-Shot Prompting with Improved Examples
Goal: Reach 75%+ accuracy on Malayalam-English sentiment analysis
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data
from datetime import datetime

# Import LLM libraries
from groq import Groq
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables
load_dotenv()

# Configure APIs
groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

groq_client = Groq(api_key=groq_key)
mistral_client = MistralClient(api_key=mistral_key)

# IMPROVED 5-shot examples - More diverse, Malayalam-heavy, realistic
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
    """Create 5-shot prompt with improved examples"""
    prompt = f"""You are analyzing sentiment in Malayalam-English code-mixed text (Manglish).

{FIVE_SHOT_EXAMPLES}

Now analyze this text:
Text: "{text}"

Classify the sentiment as:
- Positive (if overall positive/enthusiastic/happy tone)
- Negative (if overall negative/disappointed/angry tone)
- Mixed_feelings (if both positive AND negative elements present)

Respond with ONLY ONE WORD - the label. Nothing else."""
    return prompt

def test_llama(text):
    """Test Llama with 5-shot"""
    try:
        prompt = create_5shot_prompt(text)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        result = completion.choices[0].message.content.strip()
        # Clean up response - sometimes models add extra text
        if "Positive" in result:
            return "Positive"
        elif "Negative" in result:
            return "Negative"
        elif "Mixed" in result or "mixed" in result:
            return "Mixed_feelings"
        return result
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def test_mistral(text):
    """Test Mistral with 5-shot"""
    try:
        prompt = create_5shot_prompt(text)
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=20,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        # Clean up response
        if "Positive" in result:
            return "Positive"
        elif "Negative" in result:
            return "Negative"
        elif "Mixed" in result or "mixed" in result:
            return "Mixed_feelings"
        return result
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def calculate_metrics(results_df):
    """Calculate accuracy metrics"""
    # Remove ERROR predictions
    clean_df = results_df[
        (~results_df['llama_pred'].str.contains('ERROR', na=False)) &
        (~results_df['mistral_pred'].str.contains('ERROR', na=False))
    ].copy()
    
    clean_df['llama_correct'] = clean_df['llama_pred'] == clean_df['true_label']
    clean_df['mistral_correct'] = clean_df['mistral_pred'] == clean_df['true_label']
    
    llama_acc = clean_df['llama_correct'].mean() * 100
    mistral_acc = clean_df['mistral_correct'].mean() * 100
    
    metrics = {
        'overall': {
            'llama': llama_acc,
            'mistral': mistral_acc,
            'llama_correct': clean_df['llama_correct'].sum(),
            'mistral_correct': clean_df['mistral_correct'].sum(),
            'total': len(clean_df)
        }
    }
    
    # Per-label metrics
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        label_df = clean_df[clean_df['true_label'] == label]
        if len(label_df) > 0:
            metrics[label] = {
                'llama': (label_df['llama_correct'].sum() / len(label_df)) * 100,
                'mistral': (label_df['mistral_correct'].sum() / len(label_df)) * 100,
                'llama_correct': label_df['llama_correct'].sum(),
                'mistral_correct': label_df['mistral_correct'].sum(),
                'count': len(label_df)
            }
    
    return metrics, clean_df

def main():
    print("=" * 80)
    print("5-SHOT PROMPTING WITH IMPROVED EXAMPLES")
    print("Goal: Achieve 75%+ accuracy on Malayalam-English sentiment")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Select same 100 samples for fair comparison
    print("\n" + "=" * 80)
    print("SELECTING 100 TEST SAMPLES...")
    print("=" * 80)
    
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(50, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(30, random_state=42).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(20, random_state=42).to_dict('records'))
    
    print(f"\n✓ Selected 100 samples:")
    print(f"   - 50 Positive")
    print(f"   - 30 Negative")
    print(f"   - 20 Mixed_feelings")
    
    results = []
    start_time = datetime.now()
    
    print(f"\n⏰ Started at: {start_time.strftime('%H:%M:%S')}")
    print("\n" + "=" * 80)
    print("RUNNING 5-SHOT EXPERIMENT...")
    print("=" * 80)
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        if idx % 25 == 0 or idx == 1:
            print(f"[{idx}/100] Processing...")
        
        # 5-shot predictions
        llama_pred = test_llama(text)
        mistral_pred = test_mistral(text)
        
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred
        })
    
    print("✓ 5-shot experiment complete!")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/5shot_improved_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("CALCULATING RESULTS...")
    print("=" * 80)
    
    metrics, clean_df = calculate_metrics(results_df)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Display results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - 5-SHOT WITH IMPROVED EXAMPLES")
    print("=" * 80)
    
    overall = metrics['overall']
    print(f"\n📊 OVERALL ACCURACY:")
    print(f"  Llama 3.3 70B:  {overall['llama']:.1f}% ({overall['llama_correct']}/{overall['total']})")
    print(f"  Mistral Large:  {overall['mistral']:.1f}% ({overall['mistral_correct']}/{overall['total']})")
    
    # Check if we hit 75%
    best_acc = max(overall['llama'], overall['mistral'])
    if best_acc >= 75:
        print(f"\n🎉 SUCCESS! We achieved {best_acc:.1f}% accuracy!")
    else:
        print(f"\n📈 Current best: {best_acc:.1f}% (Target: 75%)")
    
    print("\n" + "=" * 80)
    print("PER-LABEL BREAKDOWN")
    print("=" * 80)
    
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        if label in metrics:
            m = metrics[label]
            print(f"\n{label} ({m['count']} samples):")
            print(f"  Llama:   {m['llama']:.1f}% ({m['llama_correct']}/{m['count']})")
            print(f"  Mistral: {m['mistral']:.1f}% ({m['mistral_correct']}/{m['count']})")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("=" * 80)
    print("\nLlama 3.3 70B:")
    print("  0-shot:  59% (baseline)")
    print("  3-shot:  67% (+8%)")
    print(f"  5-shot:  {overall['llama']:.1f}% ({overall['llama'] - 59:+.1f}% from baseline)")
    
    print("\nMistral Large:")
    print("  0-shot:  56% (baseline)")
    print("  3-shot:  66% (+10%)")
    print(f"  5-shot:  {overall['mistral']:.1f}% ({overall['mistral'] - 56:+.1f}% from baseline)")
    
    print("\n" + "=" * 80)
    print(f"⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Results saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()