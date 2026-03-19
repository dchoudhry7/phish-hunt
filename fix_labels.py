import pandas as pd
import os

# Define the path to your dataset
file_path = "data/processed/final_features_dataset.csv"

def swap_labels(path):
    """
    Swaps the values in the 'label' column (0 becomes 1, 1 becomes 0)
    and overwrites the original CSV file.
    """
    if not os.path.exists(path):
        print(f"Error: File not found at '{path}'. Please check the path.")
        return

    print(f"Loading dataset from '{path}'...")
    df = pd.read_csv(path, low_memory=False)

    # Check if 'label' column exists
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the CSV file.")
        return
        
    print("Original label distribution:")
    print(df['label'].value_counts())

    print("\nSwapping labels (0 -> 1, 1 -> 0)...")
    # A simple and fast way to swap 0s and 1s is to subtract from 1
    df['label'] = 1 - df['label']

    print("\nNew label distribution:")
    print(df['label'].value_counts())

    # Save the modified DataFrame back to the original file
    df.to_csv(path, index=False)
    print(f"\n✅ Successfully updated labels in '{path}'")

if __name__ == "__main__":
    swap_labels(file_path)