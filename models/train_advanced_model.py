# src/models/train_advanced_model.py
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_and_evaluate():
    """
    Loads the advanced feature set, trains a RandomForest model,
    and saves the entire preprocessing and model pipeline.
    """
    
    input_path = "data/processed/advanced_features_50k.csv"
    model_output_path = "models/advanced_model_50k.pkl"
    
    logging.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at {input_path}. Please ensure the file exists.")
        return

    df.dropna(inplace=True)

    drop_cols = ['url', 'label', 'domain', 'subdomain']
    X = df.drop(columns=drop_cols)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
    
    logging.info(f"Identified {len(numeric_features)} numeric features.")
    logging.info(f"Identified {len(categorical_features)} categorical feature(s): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    logging.info("Training the RandomForest model pipeline...")
    model_pipeline.fit(X_train, y_train)
    logging.info("Training complete.")

    logging.info("Evaluating model on the test set...")
    y_pred = model_pipeline.predict(X_test)

    print("\n--- Final Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    # --- MODIFIED LINE ---
    # Swapped order to match: 0=Phishing, 1=Legitimate
    print(classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))

    joblib.dump(model_pipeline, model_output_path)
    logging.info(f"Final model pipeline saved to {model_output_path}")

if __name__ == '__main__':
    train_and_evaluate()