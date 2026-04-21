"""
Create publication-quality figures for the revised paper.

Terminology aligned with revised manuscript:
  - "script-handling success rate"  →  "output validity rate"
  - "catastrophic / unusable"       →  neutral assessment labels
  - "Production Ready / Unusable"   →  "Reliable / Unreliable / No valid output"
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("CREATING PUBLICATION-QUALITY FIGURES (Revised Labels)")
print("=" * 80)

# Load data
import glob
llama_df = pd.read_csv("results/large_scale_progress.csv")
mistral_df = pd.read_csv("results/large_scale_progress.csv")

gemma_files = glob.glob("results/gemma_500_*.csv")
gemma_df = pd.read_csv(gemma_files[0])

# Calculate Malayalam percentage
def calculate_malayalam_percentage(text):
    malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', str(text)))
    total_chars = len(str(text).replace(' ', ''))
    return (malayalam_chars / total_chars * 100) if total_chars > 0 else 0

llama_df['malayalam_pct'] = llama_df['text'].apply(calculate_malayalam_percentage)
mistral_df['malayalam_pct'] = mistral_df['text'].apply(calculate_malayalam_percentage)
gemma_df['malayalam_pct'] = gemma_df['text'].apply(calculate_malayalam_percentage)

# Valid labels
VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

llama_df['llama_valid'] = llama_df['llama_pred'].isin(VALID_LABELS)
mistral_df['mistral_valid'] = mistral_df['mistral_pred'].isin(VALID_LABELS)
gemma_df['gemma_valid'] = gemma_df['gemma_pred'].isin(VALID_LABELS)

# Create bins
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

llama_df['script_bin'] = pd.cut(llama_df['malayalam_pct'], bins=bins, labels=labels, include_lowest=True)
mistral_df['script_bin'] = pd.cut(mistral_df['malayalam_pct'], bins=bins, labels=labels, include_lowest=True)
gemma_df['script_bin'] = pd.cut(gemma_df['malayalam_pct'], bins=bins, labels=labels, include_lowest=True)

# ============================================================================
# FIGURE 1: Output Validity Rate by Malayalam Script Density
# ============================================================================

print("\nCreating Figure 1: Output Validity Rate by Malayalam % ...")

fig, ax = plt.subplots(figsize=(10, 6))

bin_labels_plot = []
mistral_rates = []
llama_rates = []
gemma_rates = []

for bin_label in labels:
    mistral_bin = mistral_df[mistral_df['script_bin'] == bin_label]
    llama_bin = llama_df[llama_df['script_bin'] == bin_label]
    gemma_bin = gemma_df[gemma_df['script_bin'] == bin_label]

    if len(mistral_bin) > 0:
        bin_labels_plot.append(bin_label)
        mistral_rates.append(mistral_bin['mistral_valid'].sum() / len(mistral_bin) * 100)
        llama_rates.append(llama_bin['llama_valid'].sum() / len(llama_bin) * 100)
        gemma_rates.append(gemma_bin['gemma_valid'].sum() / len(gemma_bin) * 100)

x = np.arange(len(bin_labels_plot))
width = 0.25

bars1 = ax.bar(x - width, mistral_rates, width, label='Mistral Large', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x, llama_rates, width, label='Llama 3.3 70B', color='#e74c3c', alpha=0.8)
bars3 = ax.bar(x + width, gemma_rates, width, label='Gemma 2 9B', color='#95a5a6', alpha=0.8)

ax.set_xlabel('Malayalam Script Percentage', fontsize=12, fontweight='bold')
ax.set_ylabel('Output Validity Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Output Validity Rate by Malayalam Script Density\n'
             '(Malayalam-English Code-Mixed Samples)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels_plot)
ax.legend(loc='upper right', frameon=True, shadow=True)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/figure1_script_handling_by_percentage.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figure1_script_handling_by_percentage.pdf', bbox_inches='tight')
print("  Saved: figure1_script_handling_by_percentage.png/.pdf")
plt.close()

# ============================================================================
# FIGURE 2: Overall Output Validity Rate Comparison
# ============================================================================

print("\nCreating Figure 2: Overall Output Validity Rate Comparison ...")

fig, ax = plt.subplots(figsize=(8, 6))

models = ['Mistral\nLarge', 'Llama 3.3\n70B', 'Gemma 2\n9B']
validity_rates = [
    mistral_df['mistral_valid'].sum() / len(mistral_df) * 100,
    llama_df['llama_valid'].sum() / len(llama_df) * 100,
    gemma_df['gemma_valid'].sum() / len(gemma_df) * 100
]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']

bars = ax.bar(models, validity_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Output Validity Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Output Validity Rate\n(500 Malayalam-English Code-Mixed Samples)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar, rate in zip(bars, validity_rates):
    height = bar.get_height()
    label = f'{rate:.1f}%\n({int(round(rate / 100 * 500))}/500)'
    ax.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Neutral assessment labels (no alarmist language)
assessments = ['Reliable', 'Unreliable', 'No valid output']
assess_colors = ['green', 'darkorange', 'red']
for i, (assess, acolor) in enumerate(zip(assessments, assess_colors)):
    ax.text(i, -9, assess, ha='center', fontsize=10, color=acolor, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figure2_overall_success_rate.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figure2_overall_success_rate.pdf', bbox_inches='tight')
print("  Saved: figure2_overall_success_rate.png/.pdf")
plt.close()

# ============================================================================
# FIGURE 3: Few-Shot Prompting Progression
# NOTE: Shows BOTH models with end-to-end accuracy (invalid = wrong),
#       consistent with Table VI in the paper.
# ============================================================================

print("\nCreating Figure 3: Few-Shot Prompting Progression ...")

fig, ax = plt.subplots(figsize=(8, 6))

shots = [0, 3, 5]
llama_acc = [59.0, 67.0, 81.2]
mistral_acc = [56.0, 66.0, 71.0]

ax.plot(shots, llama_acc, 'o-', color='#e74c3c', linewidth=2.5,
        markersize=8, label='Llama 3.3 70B', zorder=3)
ax.plot(shots, mistral_acc, 's--', color='#2ecc71', linewidth=2.5,
        markersize=8, label='Mistral Large', zorder=3)

# Annotate points
for x_val, y_val in zip(shots, llama_acc):
    ax.annotate(f'{y_val:.1f}%', (x_val, y_val),
                textcoords='offset points', xytext=(6, 6), fontsize=9, color='#e74c3c')
for x_val, y_val in zip(shots, mistral_acc):
    ax.annotate(f'{y_val:.1f}%', (x_val, y_val),
                textcoords='offset points', xytext=(6, -14), fontsize=9, color='#2ecc71')

ax.set_xlabel('Number of In-Context Examples', fontsize=12, fontweight='bold')
ax.set_ylabel('End-to-End Accuracy (%)\n(invalid outputs counted as incorrect)',
              fontsize=11, fontweight='bold')
ax.set_title('End-to-End Accuracy by Prompting Condition\n'
             '(100-Sample Evaluation)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks([0, 3, 5])
ax.set_xticklabels(['0-shot', '3-shot', '5-shot'])
ax.set_ylim(45, 90)
ax.legend(loc='upper left', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figure3_fewshot_progression.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figure3_fewshot_progression.pdf', bbox_inches='tight')
print("  Saved: figure3_fewshot_progression.png/.pdf")
plt.close()

# ============================================================================
# FIGURE 4: Confusion Matrix for Mistral (Best Model)
# ============================================================================

print("\nCreating Figure 4: Confusion Matrix ...")

from sklearn.metrics import confusion_matrix

valid_mistral = mistral_df[mistral_df['mistral_pred'].isin(VALID_LABELS)].copy()
cm = confusion_matrix(valid_mistral['true_label'], valid_mistral['mistral_pred'],
                      labels=VALID_LABELS)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Positive', 'Negative', 'Mixed\nfeelings'],
            yticklabels=['Positive', 'Negative', 'Mixed\nfeelings'],
            ax=ax, square=True, linewidths=1, linecolor='gray')

ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix — Mistral Large\n(498 Valid Predictions, 5-Shot)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('results/fig4_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig4_confusion_matrix.pdf', bbox_inches='tight')
print("  Saved: fig4_confusion_matrix.png/.pdf")
plt.close()

# ============================================================================
# FIGURE 5: Model Size vs Output Validity Rate
# ============================================================================

print("\nCreating Figure 5: Model Size vs Output Validity Rate ...")

fig, ax = plt.subplots(figsize=(8, 6))

model_sizes = [9, 70, 100]
model_names = ['Gemma 2\n9B', 'Llama 3.3\n70B', 'Mistral Large\n(~100B est.)']
val_rates = [0, 44, 99.6]
colors_size = ['#95a5a6', '#e74c3c', '#2ecc71']

ax.scatter(model_sizes, val_rates, s=400, c=colors_size,
           alpha=0.8, edgecolors='black', linewidth=2, zorder=5)

for size, rate, name in zip(model_sizes, val_rates, model_names):
    offset = (-15, 12) if size == 9 else (8, 8)
    ax.annotate(name, (size, rate), xytext=offset, textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Dashed trend line (note: with only 3 points this is illustrative)
z = np.polyfit(model_sizes, val_rates, 2)
p = np.poly1d(z)
x_smooth = np.linspace(5, 105, 200)
ax.plot(x_smooth, p(x_smooth), "--", color='gray', linewidth=1.5,
        alpha=0.5, label='Trend (illustrative)')

ax.set_xlabel('Approximate Model Size (Billion Parameters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Output Validity Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Size vs Output Validity Rate\n'
             '(Note: only three data points; trend is illustrative)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(-2, 115)
ax.set_ylim(-5, 108)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/figure5_model_size_correlation.png', dpi=300, bbox_inches='tight')
plt.savefig('results/figure5_model_size_correlation.pdf', bbox_inches='tight')
print("  Saved: figure5_model_size_correlation.png/.pdf")
plt.close()

# ============================================================================
# FIGURE 6: Bootstrap Confidence Intervals
# ============================================================================

print("\nCreating Figure 6: Bootstrap Confidence Intervals ...")

fig, ax = plt.subplots(figsize=(8, 5))

model_labels = ['Mistral Large', 'Llama 3.3 70B', 'Gemma 2 9B']
rates_ci = [99.6, 44.0, 0.0]
ci_low = [99.0, 39.4, 0.0]
ci_high = [100.0, 48.0, 0.0]
colors_ci = ['#2ecc71', '#e74c3c', '#95a5a6']

y_pos = np.arange(len(model_labels))
for i, (rate, low, high, color) in enumerate(zip(rates_ci, ci_low, ci_high, colors_ci)):
    ax.hlines(i, low, high, colors=color, linewidth=4, alpha=0.6)
    ax.plot(rate, i, 'o', color=color, markersize=10, zorder=5)
    ax.annotate(f'{rate:.1f}%\n[{low:.1f}%, {high:.1f}%]', (rate, i),
                xytext=(4, 6), textcoords='offset points', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels, fontsize=11)
ax.set_xlabel('Output Validity Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('95% Bootstrap Confidence Intervals for Output Validity Rate\n'
             '(10,000 resamples; N=500)', fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(-5, 110)
ax.grid(True, alpha=0.3, linestyle='--', axis='x')

plt.tight_layout()
plt.savefig('results/fig6_confidence_intervals.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig6_confidence_intervals.pdf', bbox_inches='tight')
print("  Saved: fig6_confidence_intervals.png/.pdf")
plt.close()

# ============================================================================
# FIGURE 7: Romanization Effect on Output Validity
# ============================================================================

print("\nCreating Figure 7: Romanization Effect ...")

fig, ax = plt.subplots(figsize=(7, 5))

conditions = ['Original\n(Malayalam Script)', 'Romanized\n(Latin Script)']
validity = [44.0, 93.2]
bar_colors = ['#e74c3c', '#2ecc71']

bars = ax.bar(conditions, validity, color=bar_colors, alpha=0.85,
              edgecolor='black', linewidth=1.5, width=0.4)

for bar, val in zip(bars, validity):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

# Annotate gain
ax.annotate('', xy=(1, 93.2), xytext=(0, 44.0),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='steelblue'))
ax.text(0.5, 68, '+49.2 pp', ha='center', fontsize=12, color='steelblue', fontweight='bold')

ax.set_ylabel('Output Validity Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Effect of Romanization on Output Validity Rate\n'
             '(Llama 3.3 70B, Same 500 Samples, 5-Shot)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/fig7_romanization.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig7_romanization.pdf', bbox_inches='tight')
print("  Saved: fig7_romanization.png/.pdf")
plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("ALL FIGURES CREATED SUCCESSFULLY")
print("=" * 80)
print("\nGenerated figures:")
print("  1. Output validity rate by Malayalam script density (bar chart)")
print("  2. Overall output validity rate comparison (bar chart)")
print("  3. End-to-end accuracy by prompting condition (line chart)")
print("  4. Confusion matrix for Mistral Large (heatmap)")
print("  5. Model size vs output validity rate (scatter)")
print("  6. Bootstrap confidence intervals (horizontal CI plot)")
print("  7. Romanization effect on output validity (bar chart)")
print("\nFormats: PNG (300 dpi) + PDF (vector)")
print("Location: results/")



# """
# Create publication-quality figures for script-handling analysis
# """
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import re

# # Set publication style
# plt.style.use('seaborn-v0_8-paper')
# sns.set_palette("husl")
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['font.size'] = 10
# plt.rcParams['font.family'] = 'serif'

# print("=" * 80)
# print("CREATING PUBLICATION-QUALITY FIGURES")
# print("=" * 80)

# # Load data
# llama_df = pd.read_csv("results/large_scale_progress.csv")
# mistral_df = pd.read_csv("results/large_scale_progress.csv")

# import glob
# gemma_files = glob.glob("results/gemma_500_*.csv")
# gemma_df = pd.read_csv(gemma_files[0])

# # Calculate Malayalam percentage
# def calculate_malayalam_percentage(text):
#     malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', str(text)))
#     total_chars = len(str(text).replace(' ', ''))
#     return (malayalam_chars / total_chars * 100) if total_chars > 0 else 0

# llama_df['malayalam_pct'] = llama_df['text'].apply(calculate_malayalam_percentage)
# mistral_df['malayalam_pct'] = mistral_df['text'].apply(calculate_malayalam_percentage)
# gemma_df['malayalam_pct'] = gemma_df['text'].apply(calculate_malayalam_percentage)

# # Valid labels
# VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

# llama_df['llama_valid'] = llama_df['llama_pred'].isin(VALID_LABELS)
# mistral_df['mistral_valid'] = mistral_df['mistral_pred'].isin(VALID_LABELS)
# gemma_df['gemma_valid'] = gemma_df['gemma_pred'].isin(VALID_LABELS)

# # Create bins
# bins = [0, 20, 40, 60, 80, 100]
# labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

# llama_df['script_bin'] = pd.cut(llama_df['malayalam_pct'], bins=bins, labels=labels, include_lowest=True)
# mistral_df['script_bin'] = pd.cut(mistral_df['malayalam_pct'], bins=bins, labels=labels, include_lowest=True)
# gemma_df['script_bin'] = pd.cut(gemma_df['malayalam_pct'], bins=bins, labels=labels, include_lowest=True)

# # ============================================================================
# # FIGURE 1: Script-Handling Success Rate by Malayalam Percentage
# # ============================================================================

# print("\n📊 Creating Figure 1: Success Rate by Malayalam %...")

# fig, ax = plt.subplots(figsize=(10, 6))

# bin_labels_plot = []
# mistral_rates = []
# llama_rates = []
# gemma_rates = []

# for bin_label in labels:
#     mistral_bin = mistral_df[mistral_df['script_bin'] == bin_label]
#     llama_bin = llama_df[llama_df['script_bin'] == bin_label]
#     gemma_bin = gemma_df[gemma_df['script_bin'] == bin_label]
    
#     if len(mistral_bin) > 0:
#         bin_labels_plot.append(bin_label)
#         mistral_rates.append(mistral_bin['mistral_valid'].sum() / len(mistral_bin) * 100)
#         llama_rates.append(llama_bin['llama_valid'].sum() / len(llama_bin) * 100)
#         gemma_rates.append(gemma_bin['gemma_valid'].sum() / len(gemma_bin) * 100)

# x = np.arange(len(bin_labels_plot))
# width = 0.25

# bars1 = ax.bar(x - width, mistral_rates, width, label='Mistral Large', color='#2ecc71', alpha=0.8)
# bars2 = ax.bar(x, llama_rates, width, label='Llama 3.3 70B', color='#e74c3c', alpha=0.8)
# bars3 = ax.bar(x + width, gemma_rates, width, label='Gemma 2 9B', color='#95a5a6', alpha=0.8)

# ax.set_xlabel('Malayalam Script Percentage', fontsize=12, fontweight='bold')
# ax.set_ylabel('Script-Handling Success Rate (%)', fontsize=12, fontweight='bold')
# ax.set_title('LLM Script-Handling Performance on Malayalam-English Code-Mixing', 
#              fontsize=14, fontweight='bold', pad=20)
# ax.set_xticks(x)
# ax.set_xticklabels(bin_labels_plot)
# ax.legend(loc='upper right', frameon=True, shadow=True)
# ax.set_ylim(0, 105)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels on bars
# for bars in [bars1, bars2, bars3]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.0f}%',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=8)

# plt.tight_layout()
# plt.savefig('results/figure1_script_handling_by_percentage.png', dpi=300, bbox_inches='tight')
# plt.savefig('results/figure1_script_handling_by_percentage.pdf', bbox_inches='tight')
# print("✓ Saved: figure1_script_handling_by_percentage.png/.pdf")
# plt.close()

# # ============================================================================
# # FIGURE 2: Overall Success Rate Comparison
# # ============================================================================

# print("\n📊 Creating Figure 2: Overall Success Rate Comparison...")

# fig, ax = plt.subplots(figsize=(8, 6))

# models = ['Mistral\nLarge', 'Llama 3.3\n70B', 'Gemma 2\n9B']
# success_rates = [
#     mistral_df['mistral_valid'].sum() / len(mistral_df) * 100,
#     llama_df['llama_valid'].sum() / len(llama_df) * 100,
#     gemma_df['gemma_valid'].sum() / len(gemma_df) * 100
# ]
# colors = ['#2ecc71', '#e74c3c', '#95a5a6']

# bars = ax.bar(models, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# ax.set_ylabel('Script-Handling Success Rate (%)', fontsize=12, fontweight='bold')
# ax.set_title('Overall Script-Handling Performance\n(500 Malayalam-English Code-Mixed Samples)', 
#              fontsize=14, fontweight='bold', pad=20)
# ax.set_ylim(0, 105)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels
# for i, (bar, rate) in enumerate(zip(bars, success_rates)):
#     height = bar.get_height()
#     label = f'{rate:.1f}%\n({int(rate/100*500)}/500)'
#     ax.text(bar.get_x() + bar.get_width()/2., height + 2,
#             label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# # Add success/failure indicators
# for i, rate in enumerate(success_rates):
#     if rate > 90:
#         status = '✓ Production Ready'
#         color = 'green'
#     elif rate > 60:
#         status = '⚠ Moderate'
#         color = 'orange'
#     else:
#         status = '✗ Unusable'
#         color = 'red'
#     ax.text(i, -8, status, ha='center', fontsize=10, color=color, fontweight='bold')

# plt.tight_layout()
# plt.savefig('results/figure2_overall_success_rate.png', dpi=300, bbox_inches='tight')
# plt.savefig('results/figure2_overall_success_rate.pdf', bbox_inches='tight')
# print("✓ Saved: figure2_overall_success_rate.png/.pdf")
# plt.close()

# # ============================================================================
# # FIGURE 3: Few-Shot Prompting Progression (Mistral only)
# # ============================================================================

# print("\n📊 Creating Figure 3: Few-Shot Prompting Effectiveness...")

# fig, ax = plt.subplots(figsize=(8, 6))

# prompting_strategies = ['0-shot\n(baseline)', '3-shot\n(generic)', '5-shot\n(improved)']
# accuracies = [40, 66, 63.5]  # Estimated 0-shot, actual 3-shot, actual 5-shot
# colors_gradient = ['#e74c3c', '#f39c12', '#2ecc71']

# bars = ax.bar(prompting_strategies, accuracies, color=colors_gradient, alpha=0.8, 
#               edgecolor='black', linewidth=1.5)

# ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax.set_title('Few-Shot Prompting Effectiveness on Malayalam-English Sentiment\n(Mistral Large)', 
#              fontsize=14, fontweight='bold', pad=20)
# ax.set_ylim(0, 80)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add value labels
# for bar, acc in zip(bars, accuracies):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height + 1,
#             f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# # Add improvement arrows
# ax.annotate('', xy=(1, 66), xytext=(0, 40),
#             arrowprops=dict(arrowstyle='->', lw=2, color='green'))
# ax.text(0.5, 53, '+26%', fontsize=11, color='green', fontweight='bold')

# ax.annotate('', xy=(2, 63.5), xytext=(1, 66),
#             arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
# ax.text(1.5, 65, '-2.5%\n(variance)', fontsize=9, color='gray')

# plt.tight_layout()
# plt.savefig('results/figure3_fewshot_progression.png', dpi=300, bbox_inches='tight')
# plt.savefig('results/figure3_fewshot_progression.pdf', bbox_inches='tight')
# print("✓ Saved: figure3_fewshot_progression.png/.pdf")
# plt.close()

# # ============================================================================
# # FIGURE 4: Confusion Matrix for Mistral (Best Model)
# # ============================================================================

# print("\n📊 Creating Figure 4: Confusion Matrix...")

# from sklearn.metrics import confusion_matrix

# # Get valid Mistral predictions
# valid_mistral = mistral_df[mistral_df['mistral_pred'].isin(VALID_LABELS)].copy()

# # Create confusion matrix
# cm = confusion_matrix(valid_mistral['true_label'], valid_mistral['mistral_pred'], 
#                       labels=VALID_LABELS)

# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot heatmap
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
#             xticklabels=VALID_LABELS, yticklabels=VALID_LABELS,
#             ax=ax, square=True, linewidths=1, linecolor='gray')

# ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
# ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
# ax.set_title('Confusion Matrix - Mistral Large\n(498 valid predictions)', 
#              fontsize=14, fontweight='bold', pad=20)

# plt.tight_layout()
# plt.savefig('results/figure4_confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.savefig('results/figure4_confusion_matrix.pdf', bbox_inches='tight')
# print("✓ Saved: figure4_confusion_matrix.png/.pdf")
# plt.close()

# # ============================================================================
# # FIGURE 5: Model Size vs Script-Handling Performance
# # ============================================================================

# print("\n📊 Creating Figure 5: Model Size Correlation...")

# fig, ax = plt.subplots(figsize=(8, 6))

# model_sizes = [9, 70, 100]  # Approximate: Gemma 9B, Llama 70B, Mistral Large (~100B estimated)
# model_names = ['Gemma 2\n9B', 'Llama 3.3\n70B', 'Mistral Large\n(~100B)']
# success_rates_size = [0, 44, 99.6]
# colors_size = ['#95a5a6', '#e74c3c', '#2ecc71']

# scatter = ax.scatter(model_sizes, success_rates_size, s=500, c=colors_size, 
#                      alpha=0.7, edgecolors='black', linewidth=2)

# # Add labels
# for i, (size, rate, name) in enumerate(zip(model_sizes, success_rates_size, model_names)):
#     ax.annotate(name, (size, rate), xytext=(0, -25), textcoords='offset points',
#                 ha='center', fontsize=10, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors_size[i], alpha=0.3))

# # Add trend line
# z = np.polyfit(model_sizes, success_rates_size, 2)
# p = np.poly1d(z)
# x_smooth = np.linspace(9, 100, 100)
# ax.plot(x_smooth, p(x_smooth), "--", color='gray', linewidth=2, alpha=0.5, label='Trend')

# ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12, fontweight='bold')
# ax.set_ylabel('Script-Handling Success Rate (%)', fontsize=12, fontweight='bold')
# ax.set_title('Model Size vs Malayalam Script-Handling Performance', 
#              fontsize=14, fontweight='bold', pad=20)
# ax.set_xlim(0, 110)
# ax.set_ylim(-5, 105)
# ax.grid(True, alpha=0.3, linestyle='--')

# plt.tight_layout()
# plt.savefig('results/figure5_model_size_correlation.png', dpi=300, bbox_inches='tight')
# plt.savefig('results/figure5_model_size_correlation.pdf', bbox_inches='tight')
# print("✓ Saved: figure5_model_size_correlation.png/.pdf")
# plt.close()

# # ============================================================================
# # Summary
# # ============================================================================

# print("\n" + "=" * 80)
# print("✅ ALL FIGURES CREATED SUCCESSFULLY!")
# print("=" * 80)
# print("\nGenerated figures:")
# print("  1. Script-handling by Malayalam % (bar chart)")
# print("  2. Overall success rate comparison (bar chart)")
# print("  3. Few-shot prompting progression (bar chart)")
# print("  4. Confusion matrix for Mistral (heatmap)")
# print("  5. Model size correlation (scatter plot)")
# print("\nFormats: PNG (for viewing) + PDF (for paper)")
# print("Location: results/ folder")
# print("\n🎨 Ready for publication!")
