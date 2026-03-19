import os
import argparse
import logging
import pandas as pd
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------
# URL Standardization
# ----------------------------
def standardize_url(url: str) -> str | None:
    """Standardize a URL: add scheme if missing, lowercase netloc, remove spaces."""
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        standardized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path or ''}"
        if parsed.query:
            standardized += "?" + parsed.query
        return standardized
    except Exception:
        return None


def load_benign(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    url_col = "url" if "url" in df.columns else df.columns[0]
    df = df[[url_col]].dropna()
    df["url"] = df[url_col].apply(standardize_url)
    df = df.dropna()
    df["label"] = 0
    return df[["url", "label"]]


def load_tranco(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["rank", "domain"])
    df["url"] = df["domain"].apply(lambda d: standardize_url("http://" + d))
    df = df.dropna()
    df["label"] = 0
    return df[["url", "label"]]


def load_openphish(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["url"])
    df["url"] = df["url"].apply(standardize_url)
    df = df.dropna()
    df["label"] = 1
    return df[["url", "label"]]


def load_phishtank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    url_col = "url" if "url" in df.columns else df.columns[0]
    df = df[[url_col]].dropna()
    df["url"] = df[url_col].apply(standardize_url)
    df = df.dropna()
    df["label"] = 1
    return df[["url", "label"]]


def log_class_balance(df: pd.DataFrame, name: str):
    counts = df["label"].value_counts().to_dict()
    legit = counts.get(0, 0)
    phish = counts.get(1, 0)
    total = legit + phish
    logging.info(f"{name} → Total: {total} | Legitimate: {legit} | Phishing: {phish}")


def balance_dataset(df: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    """
    Balance dataset by undersampling legitimate URLs.
    ratio=1.0 → equal phishing and legit
    ratio=2.0 → legit count = 2 × phishing count
    """
    phishing = df[df["label"] == 1]
    legit = df[df["label"] == 0]

    target_legit_count = int(len(phishing) * ratio)
    legit_sampled = legit.sample(
        n=min(target_legit_count, len(legit)),
        random_state=42
    )

    balanced = pd.concat([phishing, legit_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


def main(args):
    logging.info("Loading datasets...")

    benign = load_benign(args.benign)
    tranco = load_tranco(args.tranco)
    openphish = load_openphish(args.openphish)
    phishtank = load_phishtank(args.phishtank)

    all_data = pd.concat([benign, tranco, openphish, phishtank], ignore_index=True)
    all_data = all_data.drop_duplicates(subset=["url"])
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    log_class_balance(all_data, "Raw master dataset")

    # Balance dataset
    balanced = balance_dataset(all_data, ratio=args.ratio)
    log_class_balance(balanced, f"Balanced dataset (ratio={args.ratio})")

    os.makedirs(os.path.dirname(args.master_out), exist_ok=True)
    balanced.to_csv(args.master_out, index=False)
    logging.info(f"Saved master dataset → {args.master_out}")

    # Train/test split
    train_size = int(0.8 * len(balanced))
    train = balanced.iloc[:train_size]
    test = balanced.iloc[train_size:]

    log_class_balance(train, "Train set")
    log_class_balance(test, "Test set")

    train.to_csv(args.train_out, index=False)
    test.to_csv(args.test_out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare phishing dataset")
    parser.add_argument("--benign", default="data/raw/Benign_list_big_final.csv")
    parser.add_argument("--openphish", default="data/raw/openphish.csv")
    parser.add_argument("--tranco", default="data/raw/top-1m.csv")
    parser.add_argument("--phishtank", default="data/raw/verified_online.csv")
    parser.add_argument("--master_out", default="data/processed/master_urls.csv")
    parser.add_argument("--train_out", default="data/processed/train_urls.csv")
    parser.add_argument("--test_out", default="data/processed/test_urls.csv")
    parser.add_argument("--ratio", type=float, default=1.0, help="Legit:Phishing ratio (default=1.0)")
    args = parser.parse_args()

    main(args)
