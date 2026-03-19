# src/data/sample_new_data.py
import pandas as pd
import os

INPUT_FILE = "data/raw/new_data_urls.csv"
OUTPUT_FILE = "data/processed/new_data_sampled.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    # Ensure label column exists
    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column (0=benign, 1=phishing)")

    # Separate by class
    benign_df = df[df["label"] == 0]
    phishing_df = df[df["label"] == 1]

    n = min(len(benign_df), len(phishing_df), 7500)  # balance, total 15k
    benign_sample = benign_df.sample(n=n, random_state=42)
    phishing_sample = phishing_df.sample(n=n, random_state=42)

    sampled_df = pd.concat([benign_sample, phishing_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    sampled_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved balanced 15,000 dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
