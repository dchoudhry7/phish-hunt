# prepare_features_dataset.py
import pandas as pd
from sklearn.utils import resample

INPUT_FILE = "data/All.csv"
OUTPUT_FILE = "data/final_features_dataset.csv"

def main():
    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    # Map labels to binary
    df["label"] = df["URL_Type_obf_Type"].apply(lambda x: 0 if x == "benign" else 1)

    # Drop the old label column
    df = df.drop(columns=["URL_Type_obf_Type"])

    # Split benign vs malicious
    benign_df = df[df["label"] == 0]
    malicious_df = df[df["label"] == 1]

    print(f"Original counts → Benign: {len(benign_df)}, Malicious: {len(malicious_df)}")

    # Balance dataset
    min_count = min(len(benign_df), len(malicious_df))

    benign_balanced = resample(benign_df, n_samples=min_count, random_state=42)
    malicious_balanced = resample(malicious_df, n_samples=min_count, random_state=42)

    # Combine
    balanced_df = pd.concat([benign_balanced, malicious_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset size: {len(balanced_df)} (Benign={min_count}, Malicious={min_count})")

    # Save
    balanced_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved balanced dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
