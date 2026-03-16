"""
Check what Gemma is actually outputting
"""
import pandas as pd

df = pd.read_csv("../results/gemma_500_20260316_193112.csv")

print("=" * 80)
print("GEMMA 2 OUTPUT ANALYSIS")
print("=" * 80)

print(f"\nTotal samples: {len(df)}")

# Show unique outputs
print("\n" + "=" * 80)
print("UNIQUE GEMMA OUTPUTS (Top 20):")
print("=" * 80)
print(df['gemma_pred'].value_counts().head(20))

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS:")
print("=" * 80)

for idx, row in df.head(10).iterrows():
    print(f"\n{idx+1}. Text: {row['text'][:60]}...")
    print(f"   True: {row['true_label']}")
    print(f"   Gemma output: '{row['gemma_pred']}'")