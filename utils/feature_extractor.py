# utils/feature_extractor.py
import numpy as np
import math
import requests
import whois
import tldextract
from urllib.parse import urlparse
from datetime import datetime
from dateutil.parser import parse as parse_date

def query_domain(domain):
    """Performs a live RDAP/WHOIS lookup for a domain."""
    try:
        w = whois.whois(domain)
        c_date = getattr(w, 'creation_date', None)
        e_date = getattr(w, 'expiration_date', None)
        if isinstance(c_date, list): c_date = c_date[0]
        if isinstance(e_date, list): e_date = e_date[0]
        return {"source": "whois", "creation_date": c_date, "expiration_date": e_date}
    except Exception:
        return None

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

def extract_features(url):
    """Orchestrates the extraction of all advanced features for a single URL."""
    domain_feats = get_domain_features(url)
    path_feats = get_path_features(url)
    url_wide_feats = get_url_wide_features(url)
   
    full_domain = f"{domain_feats['domain']}.{domain_feats['tld']}"
    domain_info = query_domain(full_domain)
    age_feats = get_domain_age_features(domain_info)
   
    all_features = {**domain_feats, **path_feats, **url_wide_feats, **age_feats}
    return all_features