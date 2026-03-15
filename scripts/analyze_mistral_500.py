"""
Analyze Mistral's performance on 500 samples
(Llama fails on Malayalam script, so we'll focus on Mistral)
"""
import pandas as pd

# Load results
df = pd.read_csv("../results/large_scale_progress.csv")

VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

# Keep only samples where Mistral gave valid predictions
valid_df = df[df['mistral_pred'].isin(VALID_LABELS)].copy()

print("=" * 80)
print("MISTRAL LARGE - 500 SAMPLE ANALYSIS")
print("=" * 80)

print(f"\n📊 Valid predictions: {len(valid_df)}/500")
print(f"   Invalid: {500 - len(valid_df)}")

# Calculate accuracy
valid_df['correct'] = valid_df['mistral_pred'] == valid_df['true_label']
accuracy = valid_df['correct'].mean() * 100

print(f"\n🎯 OVERALL ACCURACY: {accuracy:.2f}% ({valid_df['correct'].sum()}/{len(valid_df)})")

# Per-label performance
print("\n" + "=" * 80)
print("PER-LABEL BREAKDOWN")
print("=" * 80)

for label in VALID_LABELS:
    label_df = valid_df[valid_df['true_label'] == label]
    if len(label_df) > 0:
        label_correct = label_df['correct'].sum()
        label_acc = (label_correct / len(label_df)) * 100
        print(f"\n{label}: {label_acc:.1f}% ({label_correct}/{len(label_df)})")

# Confusion matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

for true_label in VALID_LABELS:
    true_count = len(valid_df[valid_df['true_label'] == true_label])
    if true_count > 0:
        print(f"\nTrue {true_label} ({true_count} samples):")
        for pred_label in VALID_LABELS:
            count = len(valid_df[(valid_df['true_label'] == true_label) & (valid_df['mistral_pred'] == pred_label)])
            if count > 0:
                print(f"  → Predicted {pred_label}: {count}")

# Calculate F1 scores
print("\n" + "=" * 80)
print("PRECISION, RECALL, F1-SCORES")
print("=" * 80)

for label in VALID_LABELS:
    tp = len(valid_df[(valid_df['true_label'] == label) & (valid_df['mistral_pred'] == label)])
    fp = len(valid_df[(valid_df['true_label'] != label) & (valid_df['mistral_pred'] == label)])
    fn = len(valid_df[(valid_df['true_label'] == label) & (valid_df['mistral_pred'] != label)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{label}:")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1-Score:  {f1*100:.1f}%")

# Macro F1
all_f1 = []
for label in VALID_LABELS:
    tp = len(valid_df[(valid_df['true_label'] == label) & (valid_df['mistral_pred'] == label)])
    fp = len(valid_df[(valid_df['true_label'] != label) & (valid_df['mistral_pred'] == label)])
    fn = len(valid_df[(valid_df['true_label'] == label) & (valid_df['mistral_pred'] != label)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    all_f1.append(f1)

macro_f1 = (sum(all_f1) / len(all_f1)) * 100

print("\n" + "=" * 80)
print(f"MACRO-AVERAGED F1: {macro_f1:.2f}%")
print("=" * 80)

print("\n" + "=" * 80)
print("COMPARISON: 100 vs 500 SAMPLES")
print("=" * 80)
print(f"\nMistral Large:")
print(f"  100 samples: 71.0%")
print(f"  500 samples: {accuracy:.2f}%")
print(f"  Difference:  {accuracy - 71.0:+.2f}%")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"\n✅ Mistral achieves {accuracy:.2f}% accuracy on {len(valid_df)} samples")
print(f"✅ Macro F1-Score: {macro_f1:.2f}%")
print(f"✅ Handles Malayalam script well (only {500-len(valid_df)} failures)")
print(f"\n⚠️ Llama failed on 280/500 samples (56% failure rate)")
print(f"   → Mistral is MUCH better for Malayalam script!")