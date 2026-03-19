# predict_lexical.py (Optimized with Top 30 Features)
import joblib
import pandas as pd
import numpy as np
import re
import tldextract
from urllib.parse import urlparse
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
MODEL_PATH = "models/lexical_ensemble_model_optimized.pkl" # Use the new optimized model

# --- Feature Extraction Logic (This part is correct) ---
def entropy(s):
    if not s: return 0
    prob = [s.count(c) / len(s) for c in set(s)]; return -sum(p * np.log2(p) for p in prob)
def longest_token(tokens):
    return max((len(t) for t in tokens if t), default=0)
def extract_lexical_features(url):
    try:
        parsed = urlparse(url); ext = tldextract.extract(url)
        domain = ext.domain; suffix = ext.suffix; path = parsed.path or ""; query = parsed.query or ""
        domain_tokens = re.split(r"\W+", domain); path_tokens = [p for p in path.split("/") if p]
        arg_tokens = query.split("&") if query else []
        features = {
            "Querylength": len(query), "domain_token_count": len(domain_tokens), "path_token_count": len(path_tokens),
            "avgdomaintokenlen": np.mean([len(t) for t in domain_tokens]) if domain_tokens else 0,
            "longdomaintokenlen": max([len(t) for t in domain_tokens]) if domain_tokens else 0,
            "avgpathtokenlen": np.mean([len(t) for t in path_tokens]) if path_tokens else 0,
            "tld": suffix, "charcompvowels": sum(c in "aeiou" for c in domain.lower()),
            "charcompace": domain.count("-"), "ldl_url": len(url.split("/")), "ldl_domain": len(domain),
            "ldl_path": len(path), "ldl_filename": len(path.split("/")[-1]) if "/" in path else 0,
            "ldl_getArg": len(query), "dld_url": len(url.split(".")), "dld_domain": len(domain.split(".")),
            "dld_path": len(path.split(".")), "dld_filename": len(path.split(".")[-1]) if "." in path else 0,
            "dld_getArg": len(query.split(".")) if query else 0, "urlLen": len(url),
            "domainlength": len(domain), "pathLength": len(path), "subDirLen": len(path_tokens),
            "fileNameLen": len(path.split("/")[-1]) if "/" in path else 0,
            "this.fileExtLen": len(path.split(".")[-1]) if "." in path else 0,
            "ArgLen": len(query), "pathurlRatio": len(path) / len(url) if url else 0,
            "ArgUrlRatio": len(query) / len(url) if url else 0,
            "argDomanRatio": len(query) / len(domain) if domain else 0,
            "domainUrlRatio": len(domain) / len(url) if url else 0,
            "pathDomainRatio": len(path) / len(domain) if domain else 0,
            "argPathRatio": len(query) / len(path) if path else 0,
            "executable": 1 if url.endswith(".exe") else 0,
            "isPortEighty": 1 if parsed.port == 80 else 0,
            "NumberofDotsinURL": url.count("."),
            "ISIpAddressInDomainName": 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc) else 0,
            "CharacterContinuityRate": max([len(m.group()) for m in re.finditer(r"(.)\1*", url)]) / len(url) if url else 0,
            "LongestVariableValue": longest_token(arg_tokens),
            "URL_DigitCount": sum(c.isdigit() for c in url), "host_DigitCount": sum(c.isdigit() for c in parsed.netloc),
            "Directory_DigitCount": sum(c.isdigit() for c in path),
            "File_name_DigitCount": sum(c.isdigit() for c in path.split("/")[-1]),
            "Extension_DigitCount": sum(c.isdigit() for c in path.split(".")[-1]) if "." in path else 0,
            "Query_DigitCount": sum(c.isdigit() for c in query), "URL_Letter_Count": sum(c.isalpha() for c in url),
            "host_letter_count": sum(c.isalpha() for c in parsed.netloc),
            "Directory_LetterCount": sum(c.isalpha() for c in path),
            "Filename_LetterCount": sum(c.isalpha() for c in path.split("/")[-1]),
            "Extension_LetterCount": sum(c.isalpha() for c in path.split(".")[-1]) if "." in path else 0,
            "Query_LetterCount": sum(c.isalpha() for c in query), "LongestPathTokenLength": longest_token(path_tokens),
            "Domain_LongestWordLength": longest_token(domain_tokens), "Path_LongestWordLength": longest_token(path_tokens),
            "sub-Directory_LongestWordLength": longest_token(path.split("/")),
            "Arguments_LongestWordLength": longest_token(arg_tokens),
            "URL_sensitiveWord": 1 if any(w in url.lower() for w in ["secure","account","login","bank","update"]) else 0,
            "URLQueries_variable": len(arg_tokens), "spcharUrl": len(re.findall(r'[!@#$%^&*(),?":{}|<>]', url)),
            "delimeter_Domain": domain.count("-"), "delimeter_path": path.count("/"),
            "delimeter_Count": url.count("/"), "NumberRate_URL": sum(c.isdigit() for c in url) / len(url) if url else 0,
            "NumberRate_Domain": sum(c.isdigit() for c in domain) / len(domain) if domain else 0,
            "NumberRate_DirectoryName": sum(c.isdigit() for c in path) / len(path) if path else 0,
            "NumberRate_FileName": sum(c.isdigit() for c in path.split("/")[-1]) / len(path.split("/")[-1]) if path.split("/")[-1] else 0,
            "NumberRate_Extension": sum(c.isdigit() for c in path.split(".")[-1]) / len(path.split(".")[-1]) if "." in path and path.split(".")[-1] else 0,
            "NumberRate_AfterPath": sum(c.isdigit() for c in query) / len(query) if query else 0,
            "SymbolCount_URL": len(re.findall(r'\W', url)), "SymbolCount_Domain": len(re.findall(r'\W', domain)),
            "SymbolCount_Directoryname": len(re.findall(r'\W', path)),
            "SymbolCount_FileName": len(re.findall(r'\W', path.split("/")[-1])),
            "SymbolCount_Extension": len(re.findall(r'\W', path.split(".")[-1])) if "." in path else 0,
            "SymbolCount_Afterpath": len(re.findall(r'\W', query)), "Entropy_URL": entropy(url),
            "Entropy_Domain": entropy(domain), "Entropy_DirectoryName": entropy(path),
            "Entropy_Filename": entropy(path.split("/")[-1]),
            "Entropy_Extension": entropy(path.split(".")[-1]) if "." in path else 0,
            "Entropy_Afterpath": entropy(query)
        }
        return features
    except Exception as e:
        logging.error(f"Error extracting features for {url}: {e}")
        return None

def main():
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("Optimized lexical model loaded successfully!")
    except FileNotFoundError:
        logging.error(f"Model file not found at {MODEL_PATH}. Please train the model first."); return

    print("\n--- Optimized Lexical Phishing URL Detector ---")
    print("Enter a URL to analyze or type 'exit' to quit.")

    while True:
        url_to_check = input("\nURL> ").strip()
        if not url_to_check: continue
        if url_to_check.lower() == 'exit': break
        if not url_to_check.startswith(('http://', 'https://')):
            url_to_check = 'http://' + url_to_check
        
        features = extract_lexical_features(url_to_check)
        if features:
            df_to_predict = pd.DataFrame([features])
            
            # --- MODIFIED SECTION ---
            prediction = model.predict(df_to_predict)[0]
            # predict_proba returns [[prob_of_0, prob_of_1]]
            # Since Phishing is now 0, its probability is at index 0.
            probabilities = model.predict_proba(df_to_predict)[0]
            
            if prediction == 0: # Phishing is label 0
                phishing_confidence = probabilities[0]
                print(f"🚨 Result: PHISHING (Confidence: {phishing_confidence:.2%})")
            else: # Legitimate is label 1
                legitimate_confidence = probabilities[1]
                print(f"✅ Result: LEGITIMATE (Confidence: {legitimate_confidence:.2%})")
        else:
            print("Could not extract features from the URL.")

if __name__ == "__main__":
    main()