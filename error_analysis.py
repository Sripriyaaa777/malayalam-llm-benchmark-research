"""
Comprehensive Error Analysis for Mistral Large
Categorizes and analyzes the 182 errors to understand failure patterns
"""
import pandas as pd
import re
from collections import Counter

print("=" * 80)
print("ERROR ANALYSIS - MISTRAL LARGE")
print("Understanding Why the Best Model Still Fails")
print("=" * 80)

# Load Mistral results
mistral_df = pd.read_csv("results/large_scale_progress.csv")

VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

# Filter to valid predictions only
valid_df = mistral_df[mistral_df['mistral_pred'].isin(VALID_LABELS)].copy()

print(f"\n✓ Loaded 498 valid Mistral predictions")
print(f"  Correct: {(valid_df['mistral_pred'] == valid_df['true_label']).sum()}")
print(f"  Errors: {(valid_df['mistral_pred'] != valid_df['true_label']).sum()}")

# Mark errors
valid_df['is_error'] = valid_df['mistral_pred'] != valid_df['true_label']
errors_df = valid_df[valid_df['is_error']].copy()

print(f"\n📊 Analyzing {len(errors_df)} errors...")

# Calculate Malayalam percentage
def calculate_malayalam_percentage(text):
    malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', str(text)))
    total_chars = len(str(text).replace(' ', ''))
    return (malayalam_chars / total_chars * 100) if total_chars > 0 else 0

errors_df['malayalam_pct'] = errors_df['text'].apply(calculate_malayalam_percentage)
errors_df['text_length'] = errors_df['text'].apply(len)

# ============================================================================
# ANALYSIS 1: Error Distribution by True Label
# ============================================================================

print("\n" + "=" * 80)
print("ERROR DISTRIBUTION BY TRUE LABEL")
print("=" * 80)

for true_label in VALID_LABELS:
    label_total = len(valid_df[valid_df['true_label'] == true_label])
    label_errors = len(errors_df[errors_df['true_label'] == true_label])
    error_rate = (label_errors / label_total * 100) if label_total > 0 else 0
    
    print(f"\n{true_label}:")
    print(f"  Total: {label_total}")
    print(f"  Errors: {label_errors}")
    print(f"  Error Rate: {error_rate:.1f}%")

# ============================================================================
# ANALYSIS 2: Confusion Patterns
# ============================================================================

print("\n" + "=" * 80)
print("CONFUSION PATTERNS")
print("=" * 80)

confusion_patterns = errors_df.groupby(['true_label', 'mistral_pred']).size().reset_index(name='count')
confusion_patterns = confusion_patterns.sort_values('count', ascending=False)

print("\nMost Common Confusions:")
for idx, row in confusion_patterns.iterrows():
    print(f"  {row['true_label']:15} → {row['mistral_pred']:15}: {row['count']:3d} errors")

# ============================================================================
# ANALYSIS 3: Error Categories
# ============================================================================

print("\n" + "=" * 80)
print("ERROR CATEGORIZATION")
print("=" * 80)

def categorize_error(row):
    text = row['text'].lower()
    true_label = row['true_label']
    pred_label = row['mistral_pred']
    
    sarcasm_words = ['comedy', 'adipoli', 'pwoli', 'kidilan', 'mass', 'super', 'kollam']
    if any(word in text for word in sarcasm_words):
        if true_label == 'Positive' and pred_label in ['Negative', 'Mixed_feelings']:
            return 'Sarcasm/Irony Misinterpretation'
    
    if true_label == 'Mixed_feelings' or pred_label == 'Mixed_feelings':
        return 'Mixed Sentiment Confusion'
    
    if (true_label == 'Positive' and pred_label == 'Negative') or \
       (true_label == 'Negative' and pred_label == 'Positive'):
        return 'Clear Sentiment Flip'
    
    return 'Other'

errors_df['error_category'] = errors_df.apply(categorize_error, axis=1)

category_counts = errors_df['error_category'].value_counts()

print("\nError Categories:")
for category, count in category_counts.items():
    percentage = count / len(errors_df) * 100
    print(f"  {category:30s}: {count:3d} ({percentage:5.1f}%)")

# ============================================================================
# ANALYSIS 4: Sample Errors
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE ERRORS BY CATEGORY")
print("=" * 80)

for category in category_counts.index[:3]:
    print(f"\n{category.upper()}")
    print("-" * 80)
    
    category_errors = errors_df[errors_df['error_category'] == category]
    
    for idx, row in category_errors.head(3).iterrows():
        print(f"\nExample {idx+1}:")
        print(f"  Text: {row['text'][:80]}...")
        print(f"  True: {row['true_label']}")
        print(f"  Predicted: {row['mistral_pred']}")
        print(f"  Malayalam %: {row['malayalam_pct']:.1f}%")

# ============================================================================
# ANALYSIS 5: Text Characteristics
# ============================================================================

print("\n" + "=" * 80)
print("TEXT CHARACTERISTICS OF ERRORS")
print("=" * 80)

correct_df = valid_df[~valid_df['is_error']].copy()
correct_df['malayalam_pct'] = correct_df['text'].apply(calculate_malayalam_percentage)

# ✅ FIX APPLIED HERE
correct_df['text_length'] = correct_df['text'].apply(len)

print("\nMalayalam Script Percentage:")
print(f"  Errors:  {errors_df['malayalam_pct'].mean():.1f}% (avg)")
print(f"  Correct: {correct_df['malayalam_pct'].mean():.1f}% (avg)")
print(f"  Difference: {errors_df['malayalam_pct'].mean() - correct_df['malayalam_pct'].mean():.1f}%")

print("\nText Length:")
print(f"  Errors:  {errors_df['text_length'].mean():.1f} chars (avg)")
print(f"  Correct: {correct_df['text_length'].mean():.1f} chars (avg)")
print(f"  Difference: {errors_df['text_length'].mean() - correct_df['text_length'].mean():.1f} chars")

# ============================================================================
# ANALYSIS 6: Hardest Cases
# ============================================================================

print("\n" + "=" * 80)
print("HARDEST CASES (High Malayalam %, Still Errors)")
print("=" * 80)

hard_errors = errors_df[errors_df['malayalam_pct'] > 70].sort_values('malayalam_pct', ascending=False)

print(f"\nFound {len(hard_errors)} errors with >70% Malayalam script:")

for idx, row in hard_errors.head(5).iterrows():
    print(f"\n{idx+1}. Malayalam: {row['malayalam_pct']:.1f}%")
    print(f"   Text: {row['text'][:70]}...")
    print(f"   True: {row['true_label']} | Predicted: {row['mistral_pred']}")

# ============================================================================
# ANALYSIS 7: Per-Label Error Analysis
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED PER-LABEL ANALYSIS")
print("=" * 80)

for true_label in VALID_LABELS:
    label_errors = errors_df[errors_df['true_label'] == true_label]
    
    if len(label_errors) > 0:
        print(f"\n{true_label} Errors ({len(label_errors)} total):")
        print(f"  Misclassified as:")
        
        for pred_label in VALID_LABELS:
            if pred_label != true_label:
                count = len(label_errors[label_errors['mistral_pred'] == pred_label])
                if count > 0:
                    percentage = count / len(label_errors) * 100
                    print(f"    {pred_label:15s}: {count:3d} ({percentage:5.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS - ERROR ANALYSIS")
print("=" * 80)

print(f"\nTotal Errors: {len(errors_df)} out of 498")

print("\n" + "=" * 80)
print("✅ ERROR ANALYSIS COMPLETE!")
print("=" * 80)

# Save results
output_path = "results/error_analysis_details.csv"
errors_df.to_csv(output_path, index=False)
print(f"\n✓ Saved to: {output_path}")