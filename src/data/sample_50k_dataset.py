import pandas as pd
import os

INPUT_FILE = "data/raw/new_data_urls.csv"
OUTPUT_FILE = "data/processed/new_data_50k.csv"
SAMPLE_SIZE = 50000  # total

def main():
    df = pd.read_csv(INPUT_FILE)

    # Expecting a column "url" and "label" (0=benign, 1=phishing)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'url' and 'label' columns")

    phishing = df[df["label"] == 1]
    benign = df[df["label"] == 0]

    # Sample equally
    n = SAMPLE_SIZE // 2
    phishing_sample = phishing.sample(n=n, random_state=42)
    benign_sample = benign.sample(n=n, random_state=42)

    balanced_df = pd.concat([phishing_sample, benign_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Balanced dataset created: {OUTPUT_FILE}, shape={balanced_df.shape}")

if __name__ == "__main__":
    main()
