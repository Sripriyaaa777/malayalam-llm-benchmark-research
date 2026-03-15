"""
Final Experiment: Ensemble + Comprehensive Error Analysis
Goal: 85%+ accuracy with deep insights
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


# ========================
# Load API keys
# ========================

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

groq_client = Groq(api_key=groq_key)
mistral_client = MistralClient(api_key=mistral_key)


# ========================
# Few-shot examples
# ========================

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


# ========================
# Prompt creation
# ========================

def create_5shot_prompt(text):

    prompt = f"""You are analyzing sentiment in Malayalam-English code-mixed text (Manglish).

{FIVE_SHOT_EXAMPLES}

Now analyze this text:
Text: "{text}"

Classify the sentiment as:

- Positive
- Negative
- Mixed_feelings

Respond with ONLY ONE WORD - the label. Nothing else.
"""

    return prompt


# ========================
# Llama prediction
# ========================

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

        return f"ERROR: {str(e)[:50]}"


# ========================
# Mistral prediction
# ========================

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

        return f"ERROR: {str(e)[:50]}"


# ========================
# Malayalam percentage
# ========================

def calculate_malayalam_percentage(text):

    malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', text))

    total_chars = len(text.replace(' ', ''))

    if total_chars == 0:
        return 0

    return (malayalam_chars / total_chars) * 100


# ========================
# Ensemble
# ========================

def ensemble_predict(llama_pred, mistral_pred):

    if llama_pred == mistral_pred:

        return llama_pred, "agreement"

    else:

        return llama_pred, "llama_fallback"


# ========================
# Confusion matrix
# ========================

def create_confusion_matrix(true_labels, predictions):

    labels = ['Positive', 'Negative', 'Mixed_feelings']

    matrix = {
        true: {pred: 0 for pred in labels}
        for true in labels
    }

    for true, pred in zip(true_labels, predictions):

        if true in labels and pred in labels:

            matrix[true][pred] += 1

    return matrix


# ========================
# Print confusion matrix
# ========================

def print_confusion_matrix(matrix, title):

    labels = ['Positive', 'Negative', 'Mixed_feelings']

    print(f"\n{title}")
    print("=" * 60)

    header = "True \\ Pred"

    print(f"{header:<15} {'Positive':<12} {'Negative':<12} {'Mixed':<12}")

    print("-" * 60)

    for true_label in labels:

        row = f"{true_label:<15}"

        for pred_label in labels:

            count = matrix[true_label][pred_label]

            row += f"{count:<12}"

        print(row)


# ========================
# Error analysis
# ========================

def analyze_errors(results_df):

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    results_df['malayalam_pct'] = results_df['text'].apply(
        calculate_malayalam_percentage)

    results_df['text_length'] = results_df['text'].apply(len)

    ensemble_errors = results_df[
        results_df['ensemble_pred'] != results_df['true_label']
    ].copy()

    print(f"\nTotal errors: {len(ensemble_errors)}")

    if len(results_df) > 0:

        print(
            f"Error rate: {len(ensemble_errors)/len(results_df)*100:.1f}%"
        )

    print("\nSample errors:")

    for idx, row in ensemble_errors.head(5).iterrows():

        print("\nText:", row['text'][:70])
        print("True:", row['true_label'])
        print("Pred:", row['ensemble_pred'])
        print("Llama:", row['llama_pred'])
        print("Mistral:", row['mistral_pred'])

    return ensemble_errors


# ========================
# MAIN
# ========================

def main():

    print("=" * 80)
    print("FINAL EXPERIMENT")
    print("=" * 80)

    df = load_malayalam_sentiment_data()

    samples = []

    samples.extend(
        df[df['label'] == 'Positive'].sample(50, random_state=42)
        .to_dict('records')
    )

    samples.extend(
        df[df['label'] == 'Negative'].sample(30, random_state=42)
        .to_dict('records')
    )

    samples.extend(
        df[df['label'] == 'Mixed_feelings'].sample(20, random_state=42)
        .to_dict('records')
    )

    print("Selected 100 samples")

    results = []

    start_time = datetime.now()

    for idx, sample in enumerate(samples, 1):

        text = sample['text']

        true_label = sample['label']

        print(f"[{idx}/100] Processing")

        llama_pred = test_llama(text)

        mistral_pred = test_mistral(text)

        ensemble_pred, ensemble_method = ensemble_predict(
            llama_pred,
            mistral_pred
        )

        results.append({

            "text": text,

            "true_label": true_label,

            "llama_pred": llama_pred,

            "mistral_pred": mistral_pred,

            "ensemble_pred": ensemble_pred,

            "ensemble_method": ensemble_method
        })

    results_df = pd.DataFrame(results)

    clean_df = results_df[
        (~results_df['llama_pred'].str.contains('ERROR', na=False)) &
        (~results_df['mistral_pred'].str.contains('ERROR', na=False))
    ]

    clean_df['ensemble_correct'] = (
        clean_df['ensemble_pred'] == clean_df['true_label']
    )

    accuracy = clean_df['ensemble_correct'].mean() * 100

    print("\nFINAL ACCURACY:", round(accuracy, 2), "%")

    matrix = create_confusion_matrix(
        clean_df['true_label'],
        clean_df['ensemble_pred']
    )

    print_confusion_matrix(matrix, "Confusion Matrix")

    analyze_errors(clean_df)

    end_time = datetime.now()

    print("\nTime taken:",
          (end_time - start_time).total_seconds() / 60,
          "minutes")


# ========================

if __name__ == "__main__":

    main()
