"""
Data loader for Malayalam-English code-mixed sentiment dataset (FIRE 2020 DravidianCodeMix)
"""
import pandas as pd
import os
import re

VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

def compute_malayalam_density(text):
    """Ratio of Malayalam Unicode chars (U+0D00–U+0D7F) to total non-whitespace chars."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    mal_chars = [c for c in non_ws if '\u0D00' <= c <= '\u0D7F']
    return len(mal_chars) / len(non_ws)

def load_malayalam_sentiment_data(data_path=None):
    """Load and clean the Malayalam sentiment dataset."""
    if data_path is None:
        # Default path relative to scripts/
        candidates = [
            "../data/DravidianCodeMix-Dataset/DravidianCodeMix/mal_full_sentiment_train.csv",
            "data/DravidianCodeMix-Dataset/DravidianCodeMix/mal_full_sentiment_train.csv",
        ]
        for p in candidates:
            if os.path.exists(p):
                data_path = p
                break
        if data_path is None:
            raise FileNotFoundError(
                "Dataset not found. Expected: data/DravidianCodeMix-Dataset/DravidianCodeMix/mal_full_sentiment_train.csv"
            )

    print(f"Loading dataset from: {data_path}")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 2:
                text = parts[0].strip()
                label = parts[1].strip()
                if text and label:
                    data.append({'text': text, 'label': label})

    df = pd.DataFrame(data)
    df_clean = df[df['label'].isin(VALID_LABELS)].copy()
    df_clean = df_clean.drop_duplicates(subset='text').reset_index(drop=True)
    df_clean['malayalam_density'] = df_clean['text'].apply(compute_malayalam_density)

    print(f"Loaded {len(df_clean)} samples after filtering.")
    print(df_clean['label'].value_counts().to_string())
    return df_clean

if __name__ == "__main__":
    df = load_malayalam_sentiment_data()
    print(f"\nTotal: {len(df)} samples ready.")
