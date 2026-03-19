# src/features/merge_features.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def merge_datasets():
    """Merges the two feature datasets into a single file."""
    
    final_features_path = "data/processed/final_features_dataset.csv"
    new_features_path = "data/processed/new_features_dataset.csv"
    output_path = "data/processed/merged_features_dataset.csv"

    logging.info(f"Loading first feature set from {final_features_path}...")
    final_df = pd.read_csv(final_features_path)
    
    logging.info(f"Loading second feature set from {new_features_path}...")
    new_df = pd.read_csv(new_features_path)

    # --- Standardize the 'tld' column in the first dataset ---
    if 'tld' in final_df.columns and pd.api.types.is_string_dtype(final_df['tld']):
        logging.info("Converting 'tld' column in first dataset to numeric length...")
        final_df['tld'] = final_df['tld'].astype(str).apply(len)
    
    # Find common columns to avoid errors if one set has extra columns
    common_cols = list(set(final_df.columns) & set(new_df.columns))
    
    logging.info(f"Merging datasets on {len(common_cols)} common columns...")
    merged_df = pd.concat([final_df[common_cols], new_df[common_cols]], ignore_index=True)
    
    # Shuffle the final dataset
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    merged_df.to_csv(output_path, index=False)
    logging.info(f"✅ Saved final merged dataset with {len(merged_df)} rows to {output_path}")

if __name__ == "__main__":
    merge_datasets()