"""
Improved sentiment analysis experiment with rate limiting
Tests 50 samples with all 3 LLMs
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data
import time
from datetime import datetime

# Import LLM libraries
import google.generativeai as genai
from groq import Groq
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables
load_dotenv()

# Configure APIs
google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

genai.configure(api_key=google_key)
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

def test_gemini(text):
    """Test Gemini with rate limiting"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = create_sentiment_prompt(text)
        response = model.generate_content(prompt)
        # Wait 13 seconds after each call (5 requests/min = 1 every 12 sec, +1 buffer)
        time.sleep(13)
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print("\n⚠️ Rate limit hit, waiting 60 seconds...")
            time.sleep(60)
            return test_gemini(text)  # Retry
        return f"ERROR: {error_msg[:50]}"

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

def main():
    print("=" * 80)
    print("SENTIMENT ANALYSIS EXPERIMENT - 50 SAMPLES")
    print("Testing: Gemini 2.5 Flash, Llama 3.3 70B, Mistral Large")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Select 50 balanced samples
    print("\n" + "=" * 80)
    print("SELECTING 50 TEST SAMPLES (Balanced across labels)...")
    print("=" * 80)
    
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(20).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(20).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(10).to_dict('records'))
    
    print(f"\n✓ Selected 50 samples:")
    print(f"   - 20 Positive")
    print(f"   - 20 Negative")
    print(f"   - 10 Mixed_feelings")
    
    results = []
    start_time = datetime.now()
    
    print(f"\n⏰ Started at: {start_time.strftime('%H:%M:%S')}")
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS (This will take ~15-20 minutes due to rate limits)")
    print("=" * 80)
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        print(f"\n[{idx}/50] Sample {idx}")
        print(f"Text: {text[:50]}...")
        print(f"True: {true_label}")
        
        # Test Gemini (slowest due to rate limit)
        print("  Gemini...", end=" ", flush=True)
        gemini_pred = test_gemini(text)
        print(f"✓ {gemini_pred}")
        
        # Test Llama (fast)
        print("  Llama...", end=" ", flush=True)
        llama_pred = test_llama(text)
        print(f"✓ {llama_pred}")
        
        # Test Mistral (fast)
        print("  Mistral...", end=" ", flush=True)
        mistral_pred = test_mistral(text)
        print(f"✓ {mistral_pred}")
        
        # Store results
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'gemini_pred': gemini_pred,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred
        })
        
        # Save intermediate results every 10 samples
        if idx % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("../results/experiment_progress.csv", index=False)
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"\n  💾 Progress saved. Elapsed: {elapsed:.1f} minutes")
    
    # Create final results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/sentiment_experiment_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    # Calculate accuracy
    print("\n" + "=" * 80)
    print("CALCULATING RESULTS...")
    print("=" * 80)
    
    # Clean predictions (remove ERROR cases for accuracy calculation)
    results_clean = results_df[~results_df['gemini_pred'].str.contains('ERROR', na=False)].copy()
    
    results_clean['gemini_correct'] = results_clean['gemini_pred'] == results_clean['true_label']
    results_clean['llama_correct'] = results_clean['llama_pred'] == results_clean['true_label']
    results_clean['mistral_correct'] = results_clean['mistral_pred'] == results_clean['true_label']
    
    gemini_acc = results_clean['gemini_correct'].mean() * 100
    llama_acc = results_clean['llama_correct'].mean() * 100
    mistral_acc = results_clean['mistral_correct'].mean() * 100
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nAccuracy on 50 test samples:")
    print(f"  Gemini 2.5 Flash:  {gemini_acc:.1f}% ({results_clean['gemini_correct'].sum()}/{len(results_clean)})")
    print(f"  Llama 3.3 70B:     {llama_acc:.1f}% ({results_clean['llama_correct'].sum()}/{len(results_clean)})")
    print(f"  Mistral Large:     {mistral_acc:.1f}% ({results_clean['mistral_correct'].sum()}/{len(results_clean)})")
    
    print(f"\n⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("🎉 EXPERIMENT COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()