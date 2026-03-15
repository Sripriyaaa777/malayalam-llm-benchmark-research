"""
Check what invalid predictions look like
"""
import pandas as pd

# Load the results
df = pd.read_csv("../results/large_scale_progress.csv")

# Find invalid predictions
VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

invalid_llama = df[~df['llama_pred'].isin(VALID_LABELS)]
invalid_mistral = df[~df['mistral_pred'].isin(VALID_LABELS)]

print("=" * 80)
print("INVALID PREDICTIONS ANALYSIS")
print("=" * 80)

print(f"\nInvalid Llama predictions: {len(invalid_llama)}")
print(f"Invalid Mistral predictions: {len(invalid_mistral)}")

print("\n" + "=" * 80)
print("SAMPLE INVALID LLAMA PREDICTIONS:")
print("=" * 80)
for idx, row in invalid_llama.head(10).iterrows():
    print(f"\nText: {row['text'][:60]}...")
    print(f"True: {row['true_label']}")
    print(f"Llama predicted: '{row['llama_pred']}'")

print("\n" + "=" * 80)
print("SAMPLE INVALID MISTRAL PREDICTIONS:")
print("=" * 80)
for idx, row in invalid_mistral.head(10).iterrows():
    print(f"\nText: {row['text'][:60]}...")
    print(f"True: {row['true_label']}")
    print(f"Mistral predicted: '{row['mistral_pred']}'")

# Count unique invalid predictions
print("\n" + "=" * 80)
print("UNIQUE INVALID LLAMA RESPONSES:")
print("=" * 80)
print(invalid_llama['llama_pred'].value_counts().head(20))

print("\n" + "=" * 80)
print("UNIQUE INVALID MISTRAL RESPONSES:")
print("=" * 80)
print(invalid_mistral['mistral_pred'].value_counts().head(20))