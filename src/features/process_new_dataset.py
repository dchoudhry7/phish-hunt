import pandas as pd
import os
from tqdm import tqdm
from extract_features import extract_url_features

INPUT_FILE = "data/processed/new_data_sampled.csv"
OUTPUT_FILE = "data/processed/new_features_dataset.csv"

def main():
    df = pd.read_csv(INPUT_FILE)
    features_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["url"]
        label = row["label"]
        feats = extract_url_features(url)
        if feats:
            feats["label"] = label
            features_list.append(feats)

    features_df = pd.DataFrame(features_list)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Extracted features for {len(features_df)} URLs -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
