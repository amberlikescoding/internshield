"""Prepare our ONE dataset: English/Hinglish synthetic Indian job messages."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import ssl
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited/resolve/main/data/synthetic_audited.csv"
SOURCE_PAGE = "https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited"

# Conservative content review BEFORE model fitting, not based on model errors.
# Exclusion is not a claim that the source label is wrong.
EXCLUSIONS = {
    "826f635812fa": "Document request: insufficient text-only evidence for the scam label.",
    "134dbb8d1d60": "OTP/login wording on an official-looking URL: ambiguous without external evidence.",
    "7dc004693e4b": "OTP/login wording: ambiguous without external evidence.",
    "83967ac187fc": "Verification-link reminder: insufficient evidence for the scam label.",
    "f066103e5d99": "Account-block/payment threat, not a clear recruitment message.",
    "a71d09fe8795": "Standalone identity-verification link, not a clear recruitment message.",
    "60343c2f86d9": "Existing-employee tax notice, not a recruitment message.",
}


def prepare_source(raw):
    required = {"row_id", "text", "label", "topic", "language"}
    if not required <= set(raw):
        raise ValueError(f"Missing source columns: {sorted(required - set(raw))}")
    selected = raw[raw["topic"].eq("job_offer") & raw["language"].isin(["en", "hinglish"])].copy()
    selected = selected[~selected["row_id"].isin(EXCLUSIONS)]
    selected["label"] = selected["label"].map({"safe": 0, "scam": 1})
    if selected.empty or selected["label"].isna().any() or selected["text"].isna().any():
        raise ValueError("Empty subset, missing text or unexpected source labels.")
    selected["text"] = selected["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    if selected["text"].eq("").any():
        raise ValueError("Empty source message.")
    selected["label"] = selected["label"].astype(int)
    return selected[["row_id", "text", "label", "language"]].rename(columns={"row_id": "source_id"})


def save_dataset(content):
    raw = pd.read_csv(BytesIO(content))
    data = prepare_source(raw)
    destination = ROOT / "data" / "india_jobs.csv"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".download")
    data.to_csv(temporary, index=False)
    temporary.replace(destination)
    metadata = {"name": "Indian recruitment messages — synthetic pilot", "source": SOURCE_PAGE,
                "raw_sha256": hashlib.sha256(content).hexdigest(),
                "prepared_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "source_rows": len(raw), "topic": "job_offer", "languages": ["en", "hinglish"],
                "excluded_source_ids": EXCLUSIONS, "rows": len(data),
                "class_counts": {str(k): int(v) for k, v in data.label.value_counts().items()},
                "synthetic": True, "review": "Assistant content review; not independent human or real-world verification.",
                "attribution": "Ridham115, Indian Scam SMS (synthetic, audited), CC BY 4.0; see source card for provider terms."}
    (ROOT / "data" / "dataset_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def download():
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urlopen(SOURCE_URL, timeout=60, context=context) as response:
        return save_dataset(response.read())


if __name__ == "__main__":
    try:
        print("Prepared:", download())
    except Exception as error:
        raise SystemExit(f"Download failed: {error}. The included data/india_jobs.csv can be used offline; see data/README.md.")
