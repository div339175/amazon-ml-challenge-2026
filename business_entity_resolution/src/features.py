#!/usr/bin/env python3
"""
Phase 4: Feature Engineering
Calculates comprehensive lexical, phonetic, n-gram, edit-distance, and domain features
for candidate pairs (S1 record <-> S2/S3 record).
"""

import math
import collections
from typing import Dict, Any, List, Tuple
from rapidfuzz import fuzz, distance

def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram cosine similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    v1 = collections.Counter(s1[i:i+n] for i in range(len(s1)-n+1))
    v2 = collections.Counter(s2[i:i+n] for i in range(len(s2)-n+1))
    dot = sum(v1[k] * v2[k] for k in v1 if k in v2)
    norm1 = math.sqrt(sum(v*v for v in v1.values()))
    norm2 = math.sqrt(sum(v*v for v in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def jaccard_similarity(s1_set: set, s2_set: set) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not s1_set or not s2_set:
        return 0.0
    intersection = len(s1_set & s2_set)
    union = len(s1_set | s2_set)
    return float(intersection / union) if union > 0 else 0.0

FEATURE_NAMES = [
    # Name features
    "name_exact_match",
    "name_stem_exact_match",
    "name_fuzz_ratio",
    "name_fuzz_partial_ratio",
    "name_fuzz_token_sort",
    "name_fuzz_token_set",
    "name_levenshtein_sim",
    "name_char3gram_cosine",
    "name_token_jaccard",
    "name_len_diff",
    "name_tok_cnt_diff",
    
    # Address features
    "addr_exact_match",
    "addr_fuzz_ratio",
    "addr_fuzz_token_set",
    "addr_levenshtein_sim",
    "addr_char3gram_cosine",
    "addr_token_jaccard",
    "pin_match_score",
    "numeric_house_jaccard",
    "numeric_house_overlap_cnt",
    "addr_len_diff",
    "addr_missing",
    
    # Combined / Interaction features
    "source_is_s2",
    "combined_name_addr_sim",
    "strong_exact_name_addr",
    "strong_name_and_num",
    "strong_name_and_pin"
]

def extract_pair_features(
    s1_preprocessed: Tuple[str, str, List[str], str, List[str], Optional[str], set],
    target_id: str,
    target_preprocessed: Tuple[str, str, List[str], str, List[str], Optional[str], set]
) -> List[float]:
    """
    Extracts numerical features for a single candidate pair.
    
    Tuple structure for preprocessed entity:
    (norm_name, stem_name, name_tokens, norm_addr, addr_tokens, pin, numeric_info)
    """
    (s1_norm_name, s1_stem_name, s1_name_toks, s1_norm_addr, s1_addr_toks, s1_pin, s1_nums) = s1_preprocessed
    (tr_norm_name, tr_stem_name, tr_name_toks, tr_norm_addr, tr_addr_toks, tr_pin, tr_nums) = target_preprocessed
    
    # --- NAME FEATURES ---
    name_exact = 1.0 if s1_norm_name == tr_norm_name and s1_norm_name else 0.0
    name_stem_exact = 1.0 if s1_stem_name == tr_stem_name and s1_stem_name else 0.0
    name_ratio = fuzz.ratio(s1_stem_name, tr_stem_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_stem_name, tr_stem_name) / 100.0
    name_sort = fuzz.token_sort_ratio(s1_norm_name, tr_norm_name) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm_name, tr_norm_name) / 100.0
    name_lev = distance.Levenshtein.normalized_similarity(s1_stem_name, tr_stem_name)
    name_c3 = char_ngram_cosine(s1_stem_name, tr_stem_name, n=3)
    name_jacc = jaccard_similarity(set(s1_name_toks), set(tr_name_toks))
    name_len_diff = float(abs(len(s1_norm_name) - len(tr_norm_name)))
    name_tok_diff = float(abs(len(s1_name_toks) - len(tr_name_toks)))
    
    # --- ADDRESS FEATURES ---
    addr_missing = 1.0 if not tr_norm_addr else 0.0
    if not tr_norm_addr:
        addr_exact = 0.0
        addr_ratio = 0.0
        addr_set = 0.0
        addr_lev = 0.0
        addr_c3 = 0.0
        addr_jacc = 0.0
        pin_score = 0.0
        num_jacc = 0.0
        num_cnt = 0.0
        addr_len_diff = float(len(s1_norm_addr))
    else:
        addr_exact = 1.0 if s1_norm_addr == tr_norm_addr else 0.0
        addr_ratio = fuzz.ratio(s1_norm_addr, tr_norm_addr) / 100.0
        addr_set = fuzz.token_set_ratio(s1_norm_addr, tr_norm_addr) / 100.0
        addr_lev = distance.Levenshtein.normalized_similarity(s1_norm_addr, tr_norm_addr)
        addr_c3 = char_ngram_cosine(s1_norm_addr, tr_norm_addr, n=3)
        addr_jacc = jaccard_similarity(set(s1_addr_toks), set(tr_addr_toks))
        
        # PIN code comparison
        if s1_pin and tr_pin:
            pin_score = 1.0 if s1_pin == tr_pin else -1.0
        else:
            pin_score = 0.0
            
        # House/Building numbers
        if s1_nums and tr_nums:
            num_jacc = jaccard_similarity(s1_nums, tr_nums)
            num_cnt = float(len(s1_nums & tr_nums))
        else:
            num_jacc = 0.0
            num_cnt = 0.0
        addr_len_diff = float(abs(len(s1_norm_addr) - len(tr_norm_addr)))
        
    # --- INTERACTION & METADATA FEATURES ---
    source_is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    combined_sim = (0.6 * name_set) + (0.4 * addr_set)
    strong_exact = 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0
    strong_name_num = 1.0 if (name_set > 0.85 and num_cnt >= 1.0) else 0.0
    strong_name_pin = 1.0 if (name_set > 0.85 and pin_score == 1.0) else 0.0
    
    return [
        name_exact,
        name_stem_exact,
        name_ratio,
        name_partial,
        name_sort,
        name_set,
        name_lev,
        name_c3,
        name_jacc,
        name_len_diff,
        name_tok_diff,
        addr_exact,
        addr_ratio,
        addr_set,
        addr_lev,
        addr_c3,
        addr_jacc,
        pin_score,
        num_jacc,
        num_cnt,
        addr_len_diff,
        addr_missing,
        source_is_s2,
        combined_sim,
        strong_exact,
        strong_name_num,
        strong_name_pin
    ]

if __name__ == "__main__":
    from business_entity_resolution.src.preprocessing import normalize_business_name, normalize_address
    
    def prep(name, addr):
        n_norm, n_stem, n_toks = normalize_business_name(name)
        a_norm, a_toks, pin, nums = normalize_address(addr)
        return (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)
    
    s1 = prep("Payne Enterprises", "3315 Fremont Street, Peoria, IL")
    s2 = prep("Payne Énterprises", "3315 FREMONT ST, PEORIA, IL")
    feats = extract_pair_features(s1, "S2-12345", s2)
    print("Extracted", len(feats), "features:")
    for fn, fv in zip(FEATURE_NAMES, feats):
        print(f"  {fn:28s}: {fv:.4f}")
