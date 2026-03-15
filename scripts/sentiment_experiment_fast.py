"""
Fast sentiment analysis experiment - Llama + Mistral only
No rate limits! Tests 100 samples in ~3-5 minutes
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

def create_sentiment_prompt(text):
    """Create zero-shot prompt for sentiment analysis"""
    prompt = f"""Analyze the sentiment of this Malayalam-English code-mixed text.

Text: "{text}"

Classify the sentiment as one of:
- Positive
- Negative  
- Mixed_feelings

Respond with ONLY the label, nothing else."""
    return prompt

def test_llama(text):
    """Test Llama via Groq"""
    try:
        prompt = create_sentiment_prompt(text)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def test_mistral(text):
    """Test Mistral"""
    try:
        prompt = create_sentiment_prompt(text)
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def calculate_metrics(results_df):
    """Calculate accuracy and other metrics"""
    # Remove ERROR predictions
    clean_df = results_df[
        (~results_df['llama_pred'].str.contains('ERROR', na=False)) &
        (~results_df['mistral_pred'].str.contains('ERROR', na=False))
    ].copy()
    
    clean_df['llama_correct'] = clean_df['llama_pred'] == clean_df['true_label']
    clean_df['mistral_correct'] = clean_df['mistral_pred'] == clean_df['true_label']
    
    llama_acc = clean_df['llama_correct'].mean() * 100
    mistral_acc = clean_df['mistral_correct'].mean() * 100
    
    # Per-label accuracy
    metrics = {
        'overall': {
            'llama': llama_acc,
            'mistral': mistral_acc,
            'llama_correct': clean_df['llama_correct'].sum(),
            'mistral_correct': clean_df['mistral_correct'].sum(),
            'total': len(clean_df)
        }
    }
    
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        label_df = clean_df[clean_df['true_label'] == label]
        if len(label_df) > 0:
            metrics[label] = {
                'llama': (label_df['llama_correct'].sum() / len(label_df)) * 100,
                'mistral': (label_df['mistral_correct'].sum() / len(label_df)) * 100,
                'count': len(label_df)
            }
    
    return metrics, clean_df

def main():
    print("=" * 80)
    print("FAST SENTIMENT ANALYSIS EXPERIMENT")
    print("Testing: Llama 3.3 70B + Mistral Large on Malayalam-English Code-Mixing")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Select 100 balanced samples
    print("\n" + "=" * 80)
    print("SELECTING 100 TEST SAMPLES...")
    print("=" * 80)
    
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(50).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(30).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(20).to_dict('records'))
    
    print(f"\n✓ Selected 100 samples:")
    print(f"   - 50 Positive")
    print(f"   - 30 Negative")
    print(f"   - 20 Mixed_feelings")
    
    results = []
    start_time = datetime.now()
    
    print(f"\n⏰ Started at: {start_time.strftime('%H:%M:%S')}")
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS (Should complete in 3-5 minutes)")
    print("=" * 80)
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        # Progress indicator every 10 samples
        if idx % 10 == 0 or idx == 1:
            print(f"\n[{idx}/100] Processing samples {idx-9 if idx > 1 else 1}-{idx}...")
        
        # Test both models
        llama_pred = test_llama(text)
        mistral_pred = test_mistral(text)
        
        # Store results
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred
        })
        
        # Save intermediate results every 25 samples
        if idx % 25 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("../results/experiment_progress.csv", index=False)
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"   💾 Progress saved. Elapsed: {elapsed:.1f} minutes")
    
    # Create final results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("CALCULATING RESULTS...")
    print("=" * 80)
    
    metrics, clean_df = calculate_metrics(results_df)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/sentiment_llama_mistral_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Display results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - OVERALL ACCURACY")
    print("=" * 80)
    
    overall = metrics['overall']
    print(f"\nAccuracy on {overall['total']} samples:")
    print(f"  Llama 3.3 70B:     {overall['llama']:.1f}% ({overall['llama_correct']}/{overall['total']})")
    print(f"  Mistral Large:     {overall['mistral']:.1f}% ({overall['mistral_correct']}/{overall['total']})")
    
    print("\n" + "=" * 80)
    print("PER-LABEL BREAKDOWN")
    print("=" * 80)
    
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        if label in metrics:
            m = metrics[label]
            print(f"\n{label} ({m['count']} samples):")
            print(f"  Llama:   {m['llama']:.1f}%")
            print(f"  Mistral: {m['mistral']:.1f}%")
    
    print("\n" + "=" * 80)
    print(f"⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Results saved to: {output_path}")
    print("\n" + "=" * 80)
    print("🎉 EXPERIMENT COMPLETE!")
    print("=" * 80)
    
    # Show sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (first 5):")
    print("=" * 80)
    for idx, row in clean_df.head(5).iterrows():
        print(f"\n{idx+1}. Text: {row['text'][:60]}...")
        print(f"   True: {row['true_label']}")
        print(f"   Llama: {row['llama_pred']} {'✓' if row['llama_correct'] else '✗'}")
        print(f"   Mistral: {row['mistral_pred']} {'✓' if row['mistral_correct'] else '✗'}")

if __name__ == "__main__":
    main()