"""
Noise-pattern standardization utilities for Business Entity Resolution.
Built from patterns observed in train_source1/2/3.tsv cross-referenced
against train_ground_truth.tsv.
"""
import re

# ---------------------------------------------------------------------------
# 1. Legal-suffix canonicalization
#    Observed: "Private Limited", "Pvt Ltd", "Pvt. Ltd.", "Pvt Ltd.",
#    "प्राइवेट लिमिटेड", "LLP", "एलएलपी", "Inc", "Inc.", "Ltd", "Ltd.", "Corp", "Corp."
# ---------------------------------------------------------------------------
SUFFIX_MAP = {
    r"\bpvt\.?\s*ltd\.?\b": "private limited",
    r"\bप्राइवेट\s+लिमिटेड\b": "private limited",
    r"\bprivate\s+limited\b": "private limited",
    r"\bएलएलपी\b": "llp",
    r"\binc\.?\b": "inc",
    r"\bltd\.?\b": "ltd",
    r"\bcorp\.?\b": "corp",
    r"\bllc\b": "llc",
}

def canonicalize_suffix(name: str) -> str:
    out = name.lower()
    for pattern, repl in SUFFIX_MAP.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# 2. Street-type abbreviation normalization
#    Observed: St/Street, Rd/Road, Ave/Avenue, Dr/Drive, Blvd
# ---------------------------------------------------------------------------
STREET_ABBR_MAP = {
    r"\bst\.?\b": "street",
    r"\brd\.?\b": "road",
    r"\bave\.?\b": "avenue",
    r"\bdr\.?\b": "drive",
    r"\bblvd\.?\b": "boulevard",
}

def canonicalize_street(address: str) -> str:
    out = address
    for pattern, repl in STREET_ABBR_MAP.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# 3. Landmark-phrase isolation (do NOT delete blindly — the landmark is a
#    weak but real signal; strip it into a side field instead of destroying it)
#    Observed triggers: "near", "opp.", "opposite", "behind", "landmark"
# ---------------------------------------------------------------------------
LANDMARK_PATTERN = re.compile(
    r"(near|opp\.?|opposite|behind|landmark)\b.*?(?=,|$)",
    flags=re.IGNORECASE
)

def extract_landmark(address: str):
    """Returns (address_without_landmark, landmark_text_or_None)."""
    match = LANDMARK_PATTERN.search(address)
    if not match:
        return address.strip(), None
    landmark = match.group(0).strip()
    cleaned = LANDMARK_PATTERN.sub("", address)
    cleaned = re.sub(r"\s*,\s*,", ",", cleaned)          # collapse double commas
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned, landmark


# ---------------------------------------------------------------------------
# 4. Domain-name -> business-name token decomposition
#    Observed: "maurewilliamscolombier.com", "prprivate.com"
# ---------------------------------------------------------------------------
DOMAIN_PATTERN = re.compile(r"^([a-z0-9\-]+)\.(com|net|in|org|co)$", re.IGNORECASE)

def domain_to_tokens(name: str):
    m = DOMAIN_PATTERN.match(name.strip())
    if not m:
        return None
    stem = m.group(1).replace("-", " ")
    # naive camel/word split heuristic for concatenated domains
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem).lower()


# ---------------------------------------------------------------------------
# 5. PIN code note
#    IMPORTANT: audit of train_source1/2/3.tsv found 0% coverage of 6-digit
#    Indian PIN codes across ALL three sources on India-labeled rows. Do NOT
#    build a PIN-code blocking key or feature — it will always be null/absent.
#    This is left here only to document the decision, not to extract anything.
# ---------------------------------------------------------------------------
PIN_CODE_PATTERN = re.compile(r"\b\d{6}\b")  # kept for completeness / future data


# ---------------------------------------------------------------------------
# 6. Example pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        ("Global Developers Medicals Private Limited",
         "H.NO 008 GOVIND BUILDING OPP.HOTEL VRINDHAVAN AARAY ROAD GOREGAON (EAST), MUMBAI, Maharashtra"),
        ("M/s Forgings Devi Ltd",
         "SURYANARAYAN VILLA, HATTALA ROAD, NEAR SBI ATM, KANKSA, West Bengal"),
        ("Raj Investments LLP",
         "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu"),
        ("prprivate.com",
         "ROYAL ARCHID SECOND FLOR, GITGUJARI 4/8 CORNER, RAJKOT, Gujarat"),
    ]
    for name, addr in samples:
        name_out = canonicalize_suffix(name)
        addr_no_landmark, landmark = extract_landmark(canonicalize_street(addr))
        print(f"NAME:    {name}  ->  {name_out}")
        print(f"ADDRESS: {addr}\n  -> cleaned: {addr_no_landmark}\n  -> landmark: {landmark}")
        print()
