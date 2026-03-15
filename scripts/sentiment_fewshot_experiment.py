"""
Few-Shot Prompting Experiment for Malayalam-English Sentiment Analysis
Compares 0-shot vs 3-shot performance
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

# Few-shot examples (carefully selected diverse examples)
FEW_SHOT_EXAMPLES = """
Here are some examples:

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

def create_zeroshot_prompt(text):
    """Create zero-shot prompt (no examples)"""
    prompt = f"""Analyze the sentiment of this Malayalam-English code-mixed text.

Text: "{text}"

Classify the sentiment as one of:
- Positive
- Negative  
- Mixed_feelings

Respond with ONLY the label, nothing else."""
    return prompt

def create_fewshot_prompt(text):
    """Create few-shot prompt (with 3 examples)"""
    prompt = f"""Analyze the sentiment of Malayalam-English code-mixed text.

{FEW_SHOT_EXAMPLES}

Now classify this text:
Text: "{text}"

Classify the sentiment as one of:
- Positive
- Negative  
- Mixed_feelings

Respond with ONLY the label, nothing else."""
    return prompt

def test_llama(text, few_shot=False):
    """Test Llama with zero-shot or few-shot"""
    try:
        prompt = create_fewshot_prompt(text) if few_shot else create_zeroshot_prompt(text)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def test_mistral(text, few_shot=False):
    """Test Mistral with zero-shot or few-shot"""
    try:
        prompt = create_fewshot_prompt(text) if few_shot else create_zeroshot_prompt(text)
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=20,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)[:50]}"

def calculate_metrics(results_df, prompt_type):
    """Calculate accuracy metrics"""
    llama_col = f'llama_{prompt_type}'
    mistral_col = f'mistral_{prompt_type}'
    
    # Remove ERROR predictions
    clean_df = results_df[
        (~results_df[llama_col].str.contains('ERROR', na=False)) &
        (~results_df[mistral_col].str.contains('ERROR', na=False))
    ].copy()
    
    clean_df[f'{llama_col}_correct'] = clean_df[llama_col] == clean_df['true_label']
    clean_df[f'{mistral_col}_correct'] = clean_df[mistral_col] == clean_df['true_label']
    
    llama_acc = clean_df[f'{llama_col}_correct'].mean() * 100
    mistral_acc = clean_df[f'{mistral_col}_correct'].mean() * 100
    
    metrics = {
        'overall': {
            'llama': llama_acc,
            'mistral': mistral_acc,
            'llama_correct': clean_df[f'{llama_col}_correct'].sum(),
            'mistral_correct': clean_df[f'{mistral_col}_correct'].sum(),
            'total': len(clean_df)
        }
    }
    
    # Per-label metrics
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        label_df = clean_df[clean_df['true_label'] == label]
        if len(label_df) > 0:
            metrics[label] = {
                'llama': (label_df[f'{llama_col}_correct'].sum() / len(label_df)) * 100,
                'mistral': (label_df[f'{mistral_col}_correct'].sum() / len(label_df)) * 100,
                'count': len(label_df)
            }
    
    return metrics

def main():
    print("=" * 80)
    print("FEW-SHOT PROMPTING EXPERIMENT")
    print("Comparing Zero-Shot vs 3-Shot on Malayalam-English Sentiment Analysis")
    print("=" * 80)
    
    # Load data
    df = load_malayalam_sentiment_data()
    
    # Select 100 balanced samples (same as before for fair comparison)
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
    
    # PHASE 1: Zero-Shot
    print("\n" + "=" * 80)
    print("PHASE 1: ZERO-SHOT PROMPTING (No examples)")
    print("=" * 80)
    
    for idx, sample in enumerate(samples, 1):
        text = sample['text']
        true_label = sample['label']
        
        if idx % 25 == 0 or idx == 1:
            print(f"[{idx}/100] Processing...")
        
        # Zero-shot predictions
        llama_zero = test_llama(text, few_shot=False)
        mistral_zero = test_mistral(text, few_shot=False)
        
        results.append({
            'sample_id': idx,
            'text': text,
            'true_label': true_label,
            'llama_zero': llama_zero,
            'mistral_zero': mistral_zero,
            'llama_few': None,  # Will fill in Phase 2
            'mistral_few': None
        })
    
    print("✓ Zero-shot phase complete!")
    
    # PHASE 2: Few-Shot
    print("\n" + "=" * 80)
    print("PHASE 2: FEW-SHOT PROMPTING (3 examples)")
    print("=" * 80)
    
    for idx in range(len(results)):
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"[{idx+1}/100] Processing...")
        
        text = results[idx]['text']
        
        # Few-shot predictions
        llama_few = test_llama(text, few_shot=True)
        mistral_few = test_mistral(text, few_shot=True)
        
        results[idx]['llama_few'] = llama_few
        results[idx]['mistral_few'] = mistral_few
    
    print("✓ Few-shot phase complete!")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../results/fewshot_comparison_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)
    
    # Calculate metrics for both approaches
    print("\n" + "=" * 80)
    print("CALCULATING RESULTS...")
    print("=" * 80)
    
    zero_metrics = calculate_metrics(results_df, 'zero')
    few_metrics = calculate_metrics(results_df, 'few')
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Display comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON: ZERO-SHOT vs FEW-SHOT")
    print("=" * 80)
    
    print("\n📊 OVERALL ACCURACY:")
    print("-" * 80)
    
    zo = zero_metrics['overall']
    fw = few_metrics['overall']
    
    print(f"\nLlama 3.3 70B:")
    print(f"  Zero-shot (0 examples): {zo['llama']:.1f}% ({zo['llama_correct']}/{zo['total']})")
    print(f"  Few-shot (3 examples):  {fw['llama']:.1f}% ({fw['llama_correct']}/{fw['total']})")
    llama_improvement = fw['llama'] - zo['llama']
    print(f"  Improvement: {llama_improvement:+.1f}% {'🎉' if llama_improvement > 0 else ''}")
    
    print(f"\nMistral Large:")
    print(f"  Zero-shot (0 examples): {zo['mistral']:.1f}% ({zo['mistral_correct']}/{zo['total']})")
    print(f"  Few-shot (3 examples):  {fw['mistral']:.1f}% ({fw['mistral_correct']}/{fw['total']})")
    mistral_improvement = fw['mistral'] - zo['mistral']
    print(f"  Improvement: {mistral_improvement:+.1f}% {'🎉' if mistral_improvement > 0 else ''}")
    
    print("\n" + "=" * 80)
    print("PER-LABEL COMPARISON")
    print("=" * 80)
    
    for label in ['Positive', 'Negative', 'Mixed_feelings']:
        if label in zero_metrics and label in few_metrics:
            print(f"\n{label}:")
            print(f"  Llama:   {zero_metrics[label]['llama']:.1f}% → {few_metrics[label]['llama']:.1f}% ({few_metrics[label]['llama'] - zero_metrics[label]['llama']:+.1f}%)")
            print(f"  Mistral: {zero_metrics[label]['mistral']:.1f}% → {few_metrics[label]['mistral']:.1f}% ({few_metrics[label]['mistral'] - zero_metrics[label]['mistral']:+.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"⏰ Total time: {duration:.1f} minutes")
    print(f"✓ Results saved to: {output_path}")
    print("\n" + "=" * 80)
    print("🎉 FEW-SHOT EXPERIMENT COMPLETE!")
    print("=" * 80)
    
    # Key finding summary
    avg_improvement = (llama_improvement + mistral_improvement) / 2
    print(f"\n🔑 KEY FINDING:")
    print(f"   Few-shot prompting improves accuracy by {avg_improvement:.1f}% on average")
    print(f"   This demonstrates the value of examples for code-mixed text!")

if __name__ == "__main__":
    main()