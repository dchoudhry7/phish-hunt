# src/features/extract_advanced_features.py
import os
import json
import math
import time
import logging
import requests
import whois
import tldextract
import pandas as pd

from urllib.parse import urlparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.parser import parse as parse_date
from tqdm import tqdm

# ================= CONFIG =================
CACHE_PATH = "data/whois_rdap_cache.json"
LOG_PATH = "data/logs/feature_extraction.log"
MAX_WORKERS = 8
HTTP_TIMEOUT = 10
RETRY_ATTEMPTS = 3
CHECKPOINT_INTERVAL = 1000   # rows
CACHE_SAVE_INTERVAL = 500    # domains

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
tqdm.pandas(desc="Extracting Features")

# ================= CACHE =================
def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded cache with {len(data)} entries from {path}")
            return data
        except json.JSONDecodeError:
            logger.warning("Cache file corrupted. Starting with empty cache.")
            return {}
    return {}

def save_cache(cache, path=CACHE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
        logger.info(f"Saved cache with {len(cache)} entries to {path}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# ================= WHOIS / RDAP =================
def query_domain(domain):
    """Query RDAP/WHOIS for domain with retries & backoff"""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            rdap_url = f"https://rdap.org/domain/{domain}"
            resp = requests.get(rdap_url, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                rdap_data = resp.json()
                creation_date, expiration_date = None, None
                for event in rdap_data.get('events', []):
                    if event.get('eventAction') == 'registration':
                        creation_date = event.get('eventDate')
                    if event.get('eventAction') == 'expiration':
                        expiration_date = event.get('eventDate')
                if creation_date or expiration_date:
                    return {
                        "source": "rdap",
                        "creation_date": creation_date,
                        "expiration_date": expiration_date
                    }
        except Exception:
            pass

        try:
            w = whois.whois(domain)
            c_date = getattr(w, 'creation_date', None)
            e_date = getattr(w, 'expiration_date', None)
            if isinstance(c_date, list): c_date = c_date[0]
            if isinstance(e_date, list): e_date = e_date[0]
            return {
                "source": "whois",
                "creation_date": str(c_date) if c_date else None,
                "expiration_date": str(e_date) if e_date else None
            }
        except Exception:
            logger.debug(f"WHOIS failed for {domain}, attempt {attempt+1}")

        time.sleep(2 ** attempt)

    return None

# ================= FEATURE HELPERS =================
def entropy(text):
    if not text: return 0
    freq = {char: text.count(char) for char in set(text)}
    return -sum((c / len(text)) * math.log2(c / len(text)) for c in freq.values())

def digit_ratio(text):
    return sum(c.isdigit() for c in text) / len(text) if text else 0

def get_domain_features(url):
    ext = tldextract.extract(url)
    return {
        'domain': ext.domain,
        'subdomain': ext.subdomain,
        'tld': ext.suffix,
        'domain_len': len(ext.domain),
        'subdomain_count': len(ext.subdomain.split('.')) if ext.subdomain else 0
    }

def get_path_features(url):
    path = urlparse(url).path
    return {
        'path_len': len(path),
        'path_token_count': len([token for token in path.split('/') if token]),
        'path_entropy': entropy(path),
        'path_digit_ratio': digit_ratio(path)
    }

def get_url_wide_features(url):
    return {
        'url_len': len(url),
        'has_at_symbol': int('@' in url),
        'hyphen_count': url.count('-'),
        'param_count': url.count('?') + url.count('&'),
        'is_https': int(urlparse(url).scheme == "https")
    }

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

# ================= FEATURE EXTRACTION =================
def extract_all_features(url, domain_cache):
    domain_feats = get_domain_features(url)
    path_feats = get_path_features(url)
    url_wide_feats = get_url_wide_features(url)

    full_domain = f"{domain_feats['domain']}.{domain_feats['tld']}"
    domain_info = domain_cache.get(full_domain)
    age_feats = get_domain_age_features(domain_info)

    return {**domain_feats, **path_feats, **url_wide_feats, **age_feats}

def generate_features(input_path, output_path, cache):
    logger.info(f"Processing file: {input_path}")
    df = pd.read_csv(input_path)

    df['full_domain'] = df['url'].apply(lambda u: f"{tldextract.extract(u).domain}.{tldextract.extract(u).suffix}")
    unique_domains = df['full_domain'].unique()
    domains_to_query = [d for d in unique_domains if d and d not in cache]

    logger.info(f"Found {len(unique_domains)} unique domains total.")
    logger.info(f"{len(domains_to_query)} new domains will be queried (others cached).")

    if domains_to_query:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_domain = {executor.submit(query_domain, domain): domain for domain in domains_to_query}
            for i, future in enumerate(tqdm(as_completed(future_to_domain), total=len(domains_to_query), desc="WHOIS/RDAP Lookups"), 1):
                domain = future_to_domain[future]
                cache[domain] = future.result()
                if i % CACHE_SAVE_INTERVAL == 0:
                    save_cache(cache)
                    logger.info(f"Checkpoint: saved cache after {i} domains.")
        save_cache(cache)

    logger.info("Extracting features with checkpointing...")
    features = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Feature Extraction"):
        feats = extract_all_features(row['url'], cache)
        features.append(feats)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            partial_df = pd.concat([df.loc[:i, ['url', 'label']], pd.DataFrame(features)], axis=1)
            checkpoint_path = output_path.replace(".csv", f"_checkpoint_{i+1}.csv")
            partial_df.to_csv(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {checkpoint_path} (rows={i+1})")

    final_df = pd.concat([df[['url', 'label']], pd.DataFrame(features)], axis=1)
    final_df.to_csv(output_path, index=False)
    # --- MODIFIED: Removed emoji to prevent error ---
    logger.info(f"Features saved to {output_path}")

# ================= MAIN =================
if __name__ == '__main__':
    whois_cache = load_cache()
    # --- MODIFIED: Updated input and output file paths ---
    generate_features(
        'data/processed/new_data_50k.csv',
        'data/processed/advanced_features_50k.csv',
        whois_cache
    )
    # --- MODIFIED: Removed emoji to prevent error ---
    logger.info("All feature extraction complete!")