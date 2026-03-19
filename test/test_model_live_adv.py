# test/test_model_live_adv.py (High-Performance Version with Percentage Report)
import joblib
import pandas as pd
import numpy as np
import logging
import math
import requests
import whois
import tldextract
import json
import os
import sys
from urllib.parse import urlparse
from datetime import datetime
from dateutil.parser import parse as parse_date
from tqdm import tqdm
from sklearn.metrics import classification_report

# --- NEW HELPER FUNCTION ---
def format_report_as_percentage(report_dict):
    """Formats a classification report dictionary to display percentages."""
    output = "\n"
    header = f"{'':<12} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>12}\n\n"
    output += header

    for label, metrics in report_dict.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        row = f"{label:<12} "
        row += f"{metrics['precision'] * 100:>10.0f}% "
        row += f"{metrics['recall'] * 100:>10.0f}% "
        row += f"{metrics['f1-score'] * 100:>10.0f}% "
        row += f"{metrics['support']:>12}\n"
        output += row
    
    output += "\n"
    if 'accuracy' in report_dict:
        acc = report_dict['accuracy']
        total_support = report_dict['macro avg']['support']
        output += f"{'accuracy':<12} {'':>12} {'':>12} {acc * 100:>10.0f}% {total_support:>12}\n"

    for label in ["macro avg", "weighted avg"]:
        if label in report_dict:
            metrics = report_dict[label]
            row = f"{label:<12} "
            row += f"{metrics['precision'] * 100:>10.0f}% "
            row += f"{metrics['recall'] * 100:>10.0f}% "
            row += f"{metrics['f1-score'] * 100:>10.0f}% "
            row += f"{metrics['support']:>12}\n"
            output += row
            
    return output

# --- CONFIGURATION ---
TEST_URLS_FILE = "test/test_urls_500_adv.csv"
MODEL_PATH = "models/advanced_model_50k.pkl"
CACHE_PATH = "data/whois_rdap_cache.json"

# --- CACHE LOGIC ---
def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            print(f"✅ Loaded cache with {len(data)} entries.")
            return data
        except json.JSONDecodeError:
            print("⚠️ Cache file corrupted. Starting with empty cache.")
            return {}
    print("No existing cache found. Starting fresh.")
    return {}

def save_cache(cache, path=CACHE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(cache, f, indent=2, default=str)
        print(f"✅ Saved updated cache with {len(cache)} entries.")
    except Exception as e: print(f"❌ Failed to save cache: {e}")

# --- FEATURE EXTRACTION LOGIC ---
def query_domain(domain):
    try:
        w = whois.whois(domain)
        c_date = getattr(w, 'creation_date', None); e_date = getattr(w, 'expiration_date', None)
        if isinstance(c_date, list): c_date = c_date[0]
        if isinstance(e_date, list): e_date = e_date[0]
        return {"source": "whois", "creation_date": c_date, "expiration_date": e_date}
    except Exception: return None

def entropy(text):
    if not text: return 0
    freq = {char: text.count(char) for char in set(text)}
    return -sum((c / len(text)) * math.log2(c / len(text)) for c in freq.values())

def digit_ratio(text):
    return sum(c.isdigit() for c in text) / len(text) if text else 0

def get_domain_features(url):
    ext = tldextract.extract(url)
    return {'domain': ext.domain, 'subdomain': ext.subdomain, 'tld': ext.suffix,'domain_len': len(ext.domain),'subdomain_count': len(ext.subdomain.split('.')) if ext.subdomain else 0}

def get_path_features(url):
    path = urlparse(url).path
    return {'path_len': len(path),'path_token_count': len([token for token in path.split('/') if token]),'path_entropy': entropy(path),'path_digit_ratio': digit_ratio(path)}

def get_url_wide_features(url):
    return {'url_len': len(url), 'has_at_symbol': int('@' in url),'hyphen_count': url.count('-'), 'param_count': url.count('?') + url.count('&'),'is_https': int(urlparse(url).scheme == "https")}

def get_domain_age_features(domain_info):
    age_days, exp_days = -1, -1
    try:
        if domain_info and domain_info.get('creation_date'):
            creation_dt = parse_date(str(domain_info['creation_date']))
            if creation_dt.tzinfo: creation_dt = creation_dt.replace(tzinfo=None)
            age_days = (datetime.now() - creation_dt).days
    except: pass
    try:
        if domain_info and domain_info.get('expiration_date'):
            exp_dt = parse_date(str(domain_info['expiration_date']))
            if exp_dt.tzinfo: exp_dt = exp_dt.replace(tzinfo=None)
            exp_days = (exp_dt - datetime.now()).days
    except: pass
    return {'domain_age_days': age_days, 'domain_exp_days': exp_days}

def extract_all_features_with_cache(url, domain_cache):
    domain_feats = get_domain_features(url); path_feats = get_path_features(url)
    url_wide_feats = get_url_wide_features(url)
    full_domain = f"{domain_feats['domain']}.{domain_feats['tld']}"
    if full_domain in domain_cache:
        domain_info = domain_cache[full_domain]
    else:
        domain_info = query_domain(full_domain)
        domain_cache[full_domain] = domain_info
    age_feats = get_domain_age_features(domain_info)
    return {**domain_feats, **path_feats, **url_wide_feats, **age_feats}

# --- MAIN TEST FUNCTION ---
def run_test():
    model = joblib.load(MODEL_PATH)
    whois_cache = load_cache()
    test_df = pd.read_csv(TEST_URLS_FILE)
    
    print(f"\n--- Starting Automated Test Run on {len(test_df)} URLs ---")
    
    features_list = []
    for url in tqdm(test_df['url'], desc="Extracting features"):
        features = extract_all_features_with_cache(url, whois_cache)
        features_list.append(features)

    X_test = pd.DataFrame(features_list)
    y_test = test_df['label']

    print("\nAll features extracted. Making predictions...")
    
    predictions = model.predict(X_test)
    
    # --- Final Summary (MODIFIED SECTION) ---
    print("\n--- Advanced Model Test Run Summary ---")
    report_dict = classification_report(y_test, predictions, target_names=["Phishing", "Legitimate"], output_dict=True)
    formatted_report = format_report_as_percentage(report_dict)
    print(formatted_report)
    
    save_cache(whois_cache)
    
    results_df = pd.DataFrame({
        'url': test_df['url'],
        'expected_label': y_test,
        'predicted_label': predictions
    })
    results_df.to_csv("test/advanced_test_run_results.csv", index=False)
    print("Detailed results saved to 'test/advanced_test_run_results.csv'")

if __name__ == "__main__":
    run_test()