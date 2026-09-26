#!/usr/bin/env python3
"""
Phase 7: Enhanced Feature Pipeline (36 Features)
Adds:
- Jaro-Winkler name and address similarities
- Consonant skeleton matching
- Token containment (subset ratios)
- Character 2-gram cosine similarity
- Domain-specific interaction features:
  * exact_name_with_empty_addr (resolves missing address FNs)
  * exact_addr_with_diff_name (resolves DBA / different brand FNs)
  * country indicators
"""

import math
import collections
from typing import Dict, Any, List, Tuple, Optional, Set
from rapidfuzz import fuzz, distance
from rapidfuzz.distance import JaroWinkler, Levenshtein

from business_entity_resolution.src.features import char_ngram_cosine, jaccard_similarity

FEATURE_NAMES_V2 = [
    # Baseline Name Features
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
    
    # Baseline Address Features
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
    
    # Baseline Interaction Features
    "source_is_s2",
    "combined_name_addr_sim",
    "strong_exact_name_addr",
    "strong_name_and_num",
    "strong_name_and_pin",

    # NEW ENHANCED FEATURES (Phase 7)
    "name_jaro_winkler",
    "name_consonant_skeleton_match",
    "name_token_containment",
    "name_char2gram_cosine",
    "addr_jaro_winkler",
    "addr_token_containment",
    "exact_name_with_empty_addr",
    "exact_addr_with_diff_name",
    "country_is_india"
]

def get_consonants(s: str) -> str:
    return "".join([c for c in s.lower() if c in "bcdfghjklmnpqrstvwxyz"])

def token_containment_ratio(toks1: List[str], toks2: List[str]) -> float:
    if not toks1 or not toks2:
        return 0.0
    s1, s2 = set(toks1), set(toks2)
    inter = len(s1 & s2)
    return float(max(inter / len(s1), inter / len(s2)))

def extract_pair_features_v2(
    s1_prep: Tuple,
    target_id: str,
    tr_prep: Tuple,
    country: str = "US"
) -> List[float]:
    """
    Extracts 36 numerical features for a single candidate pair.
    """
    (s1_norm_n, s1_stem_n, s1_toks_n, s1_norm_a, s1_toks_a, s1_pin, s1_nums) = s1_prep[:7]
    (tr_norm_n, tr_stem_n, tr_toks_n, tr_norm_a, tr_toks_a, tr_pin, tr_nums) = tr_prep[:7]

    # --- Baseline 27 Features ---
    name_exact = 1.0 if s1_norm_n == tr_norm_n and s1_norm_n else 0.0
    name_stem_exact = 1.0 if s1_stem_n == tr_stem_n and s1_stem_n else 0.0
    name_ratio = fuzz.ratio(s1_stem_n, tr_stem_n) / 100.0
    name_partial = fuzz.partial_ratio(s1_stem_n, tr_stem_n) / 100.0
    name_sort = fuzz.token_sort_ratio(s1_norm_n, tr_norm_n) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm_n, tr_norm_n) / 100.0
    name_lev = Levenshtein.normalized_similarity(s1_stem_n, tr_stem_n)
    name_c3 = char_ngram_cosine(s1_stem_n, tr_stem_n, n=3)
    name_jacc = jaccard_similarity(set(s1_toks_n), set(tr_toks_n))
    name_len_diff = float(abs(len(s1_norm_n) - len(tr_norm_n)))
    name_tok_diff = float(abs(len(s1_toks_n) - len(tr_toks_n)))

    addr_missing = 1.0 if not tr_norm_a else 0.0
    if not tr_norm_a:
        addr_exact = 0.0
        addr_ratio = 0.0
        addr_set = 0.0
        addr_lev = 0.0
        addr_c3 = 0.0
        addr_jacc = 0.0
        pin_score = 0.0
        num_jacc = 0.0
        num_cnt = 0.0
        addr_len_diff = float(len(s1_norm_a))
    else:
        addr_exact = 1.0 if s1_norm_a == tr_norm_a else 0.0
        addr_ratio = fuzz.ratio(s1_norm_a, tr_norm_a) / 100.0
        addr_set = fuzz.token_set_ratio(s1_norm_a, tr_norm_a) / 100.0
        addr_lev = Levenshtein.normalized_similarity(s1_norm_a, tr_norm_a)
        addr_c3 = char_ngram_cosine(s1_norm_a, tr_norm_a, n=3)
        addr_jacc = jaccard_similarity(set(s1_toks_a), set(tr_toks_a))

        if s1_pin and tr_pin:
            pin_score = 1.0 if s1_pin == tr_pin else -1.0
        else:
            pin_score = 0.0

        if s1_nums and tr_nums:
            num_jacc = jaccard_similarity(s1_nums, tr_nums)
            num_cnt = float(len(s1_nums & tr_nums))
        else:
            num_jacc = 0.0
            num_cnt = 0.0
        addr_len_diff = float(abs(len(s1_norm_a) - len(tr_norm_a)))

    source_is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    combined_sim = (0.6 * name_set) + (0.4 * addr_set)
    strong_exact = 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0
    strong_name_num = 1.0 if (name_set > 0.85 and num_cnt >= 1.0) else 0.0
    strong_name_pin = 1.0 if (name_set > 0.85 and pin_score == 1.0) else 0.0

    # --- NEW ENHANCED FEATURES (Phase 7) ---
    name_jw = float(JaroWinkler.similarity(s1_stem_n, tr_stem_n))
    
    # Consonant skeleton match
    skel1 = get_consonants(s1_stem_n)
    skel2 = get_consonants(tr_stem_n)
    skel_match = 1.0 if (skel1 == skel2 and len(skel1) >= 3) else 0.0

    # Token containment
    name_contain = token_containment_ratio(s1_toks_n, tr_toks_n)
    name_c2 = char_ngram_cosine(s1_stem_n, tr_stem_n, n=2)

    if not tr_norm_a:
        addr_jw = 0.0
        addr_contain = 0.0
    else:
        addr_jw = float(JaroWinkler.similarity(s1_norm_a, tr_norm_a))
        addr_contain = token_containment_ratio(s1_toks_a, tr_toks_a)

    # Missing address interaction: Exact name match with empty address
    exact_name_empty_addr = 1.0 if (name_stem_exact == 1.0 and addr_missing == 1.0) else 0.0

    # DBA interaction: Exact address & number match with differing trade name
    exact_addr_diff_name = 1.0 if (addr_set > 0.85 and num_cnt >= 1.0 and name_set < 0.5) else 0.0

    country_is_in = 1.0 if country == "India" else 0.0

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
        strong_name_pin,
        # Enhanced features
        name_jw,
        skel_match,
        name_contain,
        name_c2,
        addr_jw,
        addr_contain,
        exact_name_empty_addr,
        exact_addr_diff_name,
        country_is_in
    ]

if __name__ == "__main__":
    from business_entity_resolution.src.preprocessing import normalize_business_name, normalize_address
    s1 = normalize_business_name("George Saul Inc") + normalize_address("1400 Great Wolf Drive, WI")
    s2 = normalize_business_name("Úmbrayuma") + normalize_address("1400 Great Wolf Dr, Wisconsin")
    f = extract_pair_features_v2(s1, "S3-001", s2, country="US")
    print(f"Extracted {len(f)} features for DBA match with identical address:")
    for fn, fv in zip(FEATURE_NAMES_V2[-9:], f[-9:]):
        print(f"  {fn:30s}: {fv:.4f}")
