#!/usr/bin/env python3
"""
Phase 2: Data Preprocessing and Normalization
Implements robust normalization for business names and addresses across multiple countries
(including open-set countries such as US, India, France).
Uses unicodedata and unidecode for transliteration of Indian, Cyrillic, and accented scripts.
"""

import re
import unicodedata
import unidecode
from typing import Tuple, List, Set, Optional

# Legal suffix dictionary for standardized normalization / stripping
# Includes US, Indian (English & transliterated), French, and international legal forms
LEGAL_REPLACEMENTS = [
    (r'\b(private\s+limited|pvt\s+ltd|p\s+ltd|pvt\s+limited|praaivett\s+limittedd|praa\s+li)\b', 'pvtltd'),
    (r'\b(private|pvt|praaivett)\b', 'pvt'),
    (r'\b(limited|ltd|limittedd)\b', 'ltd'),
    (r'\b(corporation|corp|incorporated|inc)\b', 'inc'),
    (r'\b(limited\s+liability\s+company|llc)\b', 'llc'),
    (r'\b(limited\s+liability\s+partnership|llp|elelpi)\b', 'llp'),
    (r'\b(company|co)\b', 'co'),
    (r'\b(societe\s+anonyme|s\.?a\.?)\b', 'sa'),
    (r'\b(societe\s+par\s+actions\s+simplifiee|s\.?a\.?s\.?)\b', 'sas'),
    (r'\b(societe\s+a\s+responsabilite\s+limitee|s\.?a\.?r\.?l\.?)\b', 'sarl'),
    (r'\b(gesellschaft\s+mit\s+beschrankter\s+haftung|gmbh)\b', 'gmbh'),
]

LEGAL_TOKENS_SET = {
    'pvtltd', 'pvt', 'ltd', 'inc', 'llc', 'llp', 'co', 'sa', 'sas', 'sarl', 'gmbh',
    'corp', 'limited', 'private', 'incorporated', 'company', 'enterprises', 'services',
    'praaivett', 'limittedd', 'elelpi'
}

ADDR_ABBREVIATIONS = [
    (r'\b(street|str)\b', 'st'),
    (r'\b(road)\b', 'rd'),
    (r'\b(avenue|av)\b', 'ave'),
    (r'\b(boulevard|blv)\b', 'blvd'),
    (r'\b(drive)\b', 'dr'),
    (r'\b(lane)\b', 'ln'),
    (r'\b(court)\b', 'ct'),
    (r'\b(place)\b', 'pl'),
    (r'\b(terrace|terr)\b', 'ter'),
    (r'\b(highway|hwy)\b', 'hwy'),
    (r'\b(apartment|apartments|apts?)\b', 'apt'),
    (r'\b(suite|suites)\b', 'ste'),
    (r'\b(floor|flr)\b', 'fl'),
    (r'\b(building|bldgs?)\b', 'bldg'),
    (r'\b(room)\b', 'rm'),
    (r'\b(north)\b', 'n'),
    (r'\b(south)\b', 's'),
    (r'\b(east)\b', 'e'),
    (r'\b(west)\b', 'w'),
    (r'\b(opposite|oppo?)\b', 'opp'),
    (r'\b(near|nr)\b', 'near'),
    (r'\b(post\s*office\s*box|p\.?o\.?\s*box)\b', 'pobox'),
    (r'\b(rue|r\.)\b', 'rue'),
    (r'\b(chemin|ch\.)\b', 'chemin'),
    (r'\b(all[eé]e)\b', 'allee'),
]

RE_CLEAN_PUNCT = re.compile(r'[^a-z0-9\s]')
RE_WHITESPACE = re.compile(r'\s+')
RE_AMP = re.compile(r'[\&]')
RE_WEB_EXT = re.compile(r'\.(com|org|net|in|co|us|fr|io|biz|info)\b')

def base_text_clean(text: str) -> str:
    """Basic unicode normalization and transliteration to ASCII lowercase."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    # unidecode transliterates Hindi, Tamil, Cyrillic, accented Latin, etc.
    text = unidecode.unidecode(text).lower()
    return text

def normalize_business_name(name: str) -> Tuple[str, str, List[str]]:
    """
    Normalizes a business name.
    Returns:
        norm_name: cleaned name preserving standardized legal suffix
        stem_name: cleaned name with legal suffix stripped (core brand name)
        tokens: list of meaningful tokens (length >= 2, non-generic)
    """
    if not name:
        return "", "", []
    
    text = base_text_clean(name)
    # Strip web extensions & email/handle artifacts
    text = RE_WEB_EXT.sub('', text)
    text = text.replace('@', ' ')
    # Normalize & to and
    text = RE_AMP.sub(' and ', text)
    # Standardize legal suffixes
    for pattern, repl in LEGAL_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    # Remove punctuation
    text = RE_CLEAN_PUNCT.sub(' ', text)
    text = RE_WHITESPACE.sub(' ', text).strip()
    
    tokens = text.split()
    stem_tokens = [t for t in tokens if t not in LEGAL_TOKENS_SET]
    
    norm_name = " ".join(tokens)
    stem_name = " ".join(stem_tokens)
    
    meaningful_tokens = [t for t in stem_tokens if len(t) >= 2]
    return norm_name, stem_name, meaningful_tokens

def extract_postal_code(text: str) -> Optional[str]:
    """
    Extracts 5-digit (US, France) or 6-digit (India) postal/PIN code.
    Returns the first valid postal code or None.
    """
    # India 6-digit PIN code (110001 - 899999)
    in_pins = re.findall(r'\b[1-9]\d{5}\b', text)
    if in_pins:
        return in_pins[0]
    # US / France 5-digit zip code (01000 - 99999)
    us_pins = re.findall(r'\b\d{5}\b', text)
    if us_pins:
        return us_pins[0]
    return None

def extract_numeric_house_info(tokens: List[str]) -> Set[str]:
    """
    Extracts house/building/unit numbers, stripping leading zeros so '01604' matches '1604'.
    Also extracts numbers from alphanumeric units like 'a702' -> '702'.
    """
    nums = set()
    for t in tokens:
        if t.isdigit():
            norm_num = t.lstrip('0')
            if norm_num and len(norm_num) <= 8:
                nums.add(norm_num)
        else:
            # Check for patterns like a702, 105b, or unit-12
            m = re.findall(r'\b[a-z]?(\d{1,6})[a-z]?\b', t)
            for sub in m:
                s_norm = sub.lstrip('0')
                if s_norm:
                    nums.add(s_norm)
    return nums

def normalize_address(address: str) -> Tuple[str, List[str], Optional[str], Set[str]]:
    """
    Normalizes a business address.
    Returns:
        norm_address: standard cleaned address string
        tokens: list of cleaned address tokens
        postal_code: extracted postal/PIN code (or None)
        numeric_info: set of extracted house/building numeric identifiers
    """
    if not address:
        return "", [], None, set()
    
    raw_clean = base_text_clean(address)
    postal_code = extract_postal_code(raw_clean)
    
    text = RE_AMP.sub(' and ', raw_clean)
    for pattern, repl in ADDR_ABBREVIATIONS:
        text = re.sub(pattern, repl, text)
        
    text = RE_CLEAN_PUNCT.sub(' ', text)
    text = RE_WHITESPACE.sub(' ', text).strip()
    
    tokens = text.split()
    numeric_info = extract_numeric_house_info(tokens)
    
    return text, tokens, postal_code, numeric_info

if __name__ == "__main__":
    test_cases_names = [
        "Orelee's Barbershop",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Marina École France Sàrl",
        "Payne Enterprises Inc.",
        "PAYNE-ENRTPRMISES",
        "maurewilliamscolombier.com",
        "Raj Investments LLP",
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
    ]
    print("Testing Business Name Normalization:")
    for n in test_cases_names:
        norm, stem, toks = normalize_business_name(n)
        print(f"  Orig: {n}\n  Norm: {norm} | Stem: {stem} | Toks: {toks}\n")

    test_cases_addr = [
        "1795 Westchester Drive, High Point, NC 27262",
        "AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश 201001",
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine 33000",
        "01604 Washington Street, PMB 841, Greensboro, North Carolina",
        "Flat No. A702, 1St Main, Bangalore, KA",
    ]
    print("\nTesting Address Normalization:")
    for a in test_cases_addr:
        norm, toks, pin, nums = normalize_address(a)
        print(f"  Orig: {a}\n  Norm: {norm}\n  PIN: {pin} | Nums: {nums}\n")
