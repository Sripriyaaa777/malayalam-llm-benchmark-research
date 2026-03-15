"""
Data loader for Malayalam-English code-mixed sentiment dataset
"""
import pandas as pd
import os

def load_malayalam_sentiment_data():
    """Load the Malayalam sentiment analysis dataset"""
    
    # Path to dataset
    data_path = "../data/DravidianCodeMix-Dataset/DravidianCodeMix/mal_full_sentiment_train.csv"
    
    print("Loading Malayalam-English sentiment dataset...")
    print("-" * 60)
    
    # Read file line by line to handle inconsistent separators
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 2:
                text = parts[0]
                label = parts[1]
                if text and label:  # Skip empty lines
                    data.append({'text': text, 'label': label})
    
    df = pd.DataFrame(data)
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Filter to keep only clear sentiment labels
    sentiment_labels = ['Positive', 'Negative', 'Mixed_feelings']
    df_clean = df[df['label'].isin(sentiment_labels)].copy()
    
    print(f"\n✓ After filtering: {len(df_clean)} samples with clear sentiment labels")
    print(f"\nFiltered label distribution:")
    print(df_clean['label'].value_counts())
    
    # Show sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 60)
    for idx, row in df_clean.head(5).iterrows():
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        print(f"\n{idx+1}. Text: {text_preview}")
        print(f"   Label: {row['label']}")
    
    return df_clean

if __name__ == "__main__":
    # Test the data loader
    df = load_malayalam_sentiment_data()
    
    print("\n" + "=" * 60)
    print(f"✅ DATA LOADING SUCCESSFUL!")
    print(f"   Total samples: {len(df)}")
    print(f"   Ready for LLM experiments!")
    print("=" * 60)