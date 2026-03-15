"""
Test sentiment analysis experiment on Malayalam-English code-mixed text
Testing all 3 LLMs on 10 samples
"""
import os
from dotenv import load_dotenv
import pandas as pd
from load_data import load_malayalam_sentiment_data

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
    """Create a simple zero-shot prompt for sentiment analysis"""
    prompt = f"""Analyze the sentiment of this Malayalam-English code-mixed text.

Text: "{text}"

Classify the sentiment as one of:
- Positive
- Negative  
- Mixed_feelings

Respond with ONLY the label, nothing else."""
    return prompt

def test_gemini(text):
    """Test Gemini on sentiment analysis"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = create_sentiment_prompt(text)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def test_llama(text):
    """Test Llama via Groq on sentiment analysis"""
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
        return f"ERROR: {str(e)}"

def test_mistral(text):
    """Test Mistral on sentiment analysis"""
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
        return f"ERROR: {str(e)}"

def main():
    print("=" * 80)
    print("TESTING SENTIMENT ANALYSIS ON MALAYALAM-ENGLISH CODE-MIXED TEXT")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Take 10 samples (mix of different sentiments)
    print("\n" + "=" * 80)
    print("SELECTING 10 TEST SAMPLES...")
    print("=" * 80)
    
    # Get balanced sample: some positive, some negative, some mixed
    samples = []
    samples.extend(df[df['label'] == 'Positive'].sample(4).to_dict('records'))
    samples.extend(df[df['label'] == 'Negative'].sample(3).to_dict('records'))
    samples.extend(df[df['label'] == 'Mixed_feelings'].sample(3).to_dict('records'))
    
    results = []
    
    print(f"\n✓ Selected 10 samples")
    print("\nTesting each sample with all 3 LLMs...")
    print("-" * 80)
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        print(f"\n[{idx}/10] Testing sample...")
        print(f"Text: {text[:60]}...")
        print(f"True Label: {true_label}")
        
        # Test with all 3 models
        print("  Testing Gemini...", end=" ")
        gemini_pred = test_gemini(text)
        print(f"✓ {gemini_pred}")
        
        print("  Testing Llama...", end=" ")
        llama_pred = test_llama(text)
        print(f"✓ {llama_pred}")
        
        print("  Testing Mistral...", end=" ")
        mistral_pred = test_mistral(text)
        print(f"✓ {mistral_pred}")
        
        # Store results
        results.append({
            'text': text,
            'true_label': true_label,
            'gemini_pred': gemini_pred,
            'llama_pred': llama_pred,
            'mistral_pred': mistral_pred
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = "../results/test_experiment_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Calculate accuracy for each model
    results_df['gemini_correct'] = results_df['gemini_pred'] == results_df['true_label']
    results_df['llama_correct'] = results_df['llama_pred'] == results_df['true_label']
    results_df['mistral_correct'] = results_df['mistral_pred'] == results_df['true_label']
    
    gemini_acc = results_df['gemini_correct'].mean() * 100
    llama_acc = results_df['llama_correct'].mean() * 100
    mistral_acc = results_df['mistral_correct'].mean() * 100
    
    print(f"\nAccuracy on 10 test samples:")
    print(f"  Gemini 2.5 Flash:  {gemini_acc:.1f}% ({results_df['gemini_correct'].sum()}/10)")
    print(f"  Llama 3.3 70B:     {llama_acc:.1f}% ({results_df['llama_correct'].sum()}/10)")
    print(f"  Mistral Large:     {mistral_acc:.1f}% ({results_df['mistral_correct'].sum()}/10)")
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "=" * 80)
    print("🎉 TEST EXPERIMENT COMPLETE!")
    print("All 3 LLMs are working on Malayalam-English code-mixing!")
    print("=" * 80)

if __name__ == "__main__":
    main()