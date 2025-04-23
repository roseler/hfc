import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_ids2018_dataset(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Binary classification: Normal = 0, Attack = 1
    df['label'] = df['attack_cat'].apply(lambda x: 0 if str(x).strip() == 'Benign' else 1)

    # Drop original attack_cat and keep only numeric columns
    df = df.drop(columns=['attack_cat'])
    df = df.select_dtypes(include=['number'])

    # Replace inf/-inf with NaN, then fill with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Optional: clip extreme values (you can comment this if unnecessary)
    df = df.clip(lower=-1e10, upper=1e10)

    # Extract features and labels
    X = df.drop(columns=['label'], errors='ignore')
    y = df['label']

    # Print class distribution before split (for debugging)
    print("Label distribution before split:\n", y.value_counts())

    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)