# src/models/train_lexical_ensemble.py (Optimized with Top 30 Features)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- The list of the top 30 features you discovered ---
TOP_FEATURES = [
    'delimeter_path', 'tld', 'Entropy_Extension', 'subDirLen', 'path_token_count', 
    'SymbolCount_Extension', 'Entropy_DirectoryName', 'pathurlRatio', 'pathLength', 
    'dld_filename', 'domain_token_count', 'SymbolCount_Domain', 'delimeter_Count', 
    'CharacterContinuityRate', 'Extension_LetterCount', 'fileNameLen', 'pathDomainRatio', 
    'Entropy_Domain', 'URL_sensitiveWord', 'this.fileExtLen', 'Entropy_Filename', 
    'avgpathtokenlen', 'SymbolCount_Directoryname', 'Entropy_URL', 'Directory_LetterCount', 
    'domainUrlRatio', 'urlLen', 'ldl_filename', 'LongestPathTokenLength', 'NumberofDotsinURL'
]

def train_and_evaluate(input_path, model_output_path):
    logging.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # --- Use only the top features for training ---
    logging.info(f"Selecting the top {len(TOP_FEATURES)} features for training...")
    X = df[TOP_FEATURES]
    y = df["label"]
    
    # Ensure 'tld' is treated as a string for the preprocessor
    if 'tld' in X.columns:
        X['tld'] = X['tld'].astype(str)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define which columns are which type from our selected features
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Define the models
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    
    # Create the full pipeline with the VotingClassifier
    voting_clf = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft', n_jobs=-1)
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', voting_clf)])

    logging.info("Training the final model pipeline on the optimized feature set...")
    model_pipeline.fit(X_train, y_train)

    logging.info("Evaluating final model...")
    y_pred = model_pipeline.predict(X_test)
    
    print("\n--- Final Optimized Model Evaluation ---")
    print(classification_report(y_test, y_pred))

    joblib.dump(model_pipeline, model_output_path)
    logging.info(f"✅ Final optimized model saved to {model_output_path}")

if __name__ == '__main__':
    train_and_evaluate(
        "data/processed/merged_features_dataset.csv",
        "models/lexical_ensemble_model_optimized.pkl"
    )