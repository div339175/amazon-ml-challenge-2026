#!/usr/bin/env python3
"""
Phase 5 & 8: High-Recall Multi-Pass Blocking Index v2
Combines:
1. Exact normalized name
2. Concatenated name string (handles merged words, domains)
3. Phonetic consonant skeleton
4. Distinctive name tokens (length >= 3)
5. House number + Locality/Word (strong spatial signal)
6. Postal/PIN + name prefix
7. Compound address token pairs (for DBAs)
"""

import collections
from typing import Dict, List, Tuple, Set, Optional
from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
    LEGAL_TOKENS_SET,
)

ADDR_STOPWORDS = {
    'st', 'rd', 'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'ter', 'hwy',
    'apt', 'ste', 'fl', 'bldg', 'rm', 'n', 's', 'e', 'w', 'opp', 'near',
    'pobox', 'rue', 'chemin', 'allee', 'india', 'us', 'usa', 'france',
    'state', 'city', 'dist', 'district', 'region', 'floor', 'suite', 'block',
    'phase', 'sector', 'nagar', 'colony', 'road', 'street', 'main', 'cross',
    'layout', 'delhi', 'mumbai', 'bangalore', 'chennai', 'paris', 'new'
}

def get_consonants(s: str) -> str:
    return "".join([c for c in s.lower() if c in "bcdfghjklmnpqrstvwxyz"])

def extract_blocking_keys_v2(
    stem_name: str,
    name_tokens: List[str],
    norm_addr: str,
    addr_tokens: List[str],
    postal_code: Optional[str],
    numeric_info: Set[str]
) -> List[Tuple[str, float]]:
    """
    Generates multi-pass blocking keys with calibrated intrinsic relevance weights.
    Returns: list of (key_string, weight)
    """
    keys = []

    # 1. Exact normalized stem name (high confidence)
    if stem_name and len(stem_name) >= 3:
        keys.append(('nx:' + stem_name, 14.0))

    # 2. Concatenated name string (handles merged words, domains like company.com)
    if len(name_tokens) >= 2:
        concat = ''.join(name_tokens)
        if len(concat) >= 5:
            keys.append(('nc:' + concat[:16], 11.0))
    elif stem_name and len(stem_name) >= 5:
        concat = ''.join(stem_name.split())
        keys.append(('nc:' + concat[:16], 10.0))

    # 3. Phonetic consonant skeleton (catches typos and transliterated Indic names)
    if name_tokens:
        skel = "_".join([get_consonants(t)[:4] for t in name_tokens[:2] if len(get_consonants(t)) >= 2])
        if len(skel) >= 4:
            keys.append(('psk:' + skel, 9.0))

    # 4. Individual distinctive name tokens (length >= 3, non-generic)
    for tok in name_tokens:
        if len(tok) >= 3 and tok not in LEGAL_TOKENS_SET:
            w = 8.0 if len(tok) >= 6 else (5.5 if len(tok) >= 4 else 3.5)
            keys.append(('nt:' + tok, w))

    # 5. First token prefix
    if name_tokens:
        first = name_tokens[0]
        if len(first) >= 4 and first not in LEGAL_TOKENS_SET:
            keys.append(('np:' + first[:4], 3.5))

    # 6. Compound Address: Number + Locality/Word (strong spatial signal)
    sig_addr = [w for w in addr_tokens if len(w) >= 4 and w not in ADDR_STOPWORDS][:4]
    nums = sorted(list(numeric_info))[:2]

    for n in nums:
        for w in sig_addr[:2]:
            keys.append((f'anw:{n}_{w}', 8.0))
        if postal_code:
            keys.append((f'anp:{n}_{postal_code}', 9.0))

    # 7. Postal/PIN + name prefix
    if postal_code and name_tokens:
        first = name_tokens[0]
        if len(first) >= 3:
            keys.append((f'pnp:{postal_code}_{first[:3]}', 7.5))

    # 8. Compound Address words (handles DBAs / alternative brand names)
    if len(sig_addr) >= 2:
        keys.append((f'amp:{sig_addr[0]}_{sig_addr[1]}', 6.0))

    return keys

class CountryBlockingIndexV2:
    """
    Enhanced Multi-Pass Inverted Index partitioned strictly by country.
    """
    def __init__(self, max_posting_size: int = 350):
        self.max_posting_size = max_posting_size
        self.indices: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self.target_data: Dict[str, Tuple] = {}

    def add_target_entity(
        self,
        entity_id: str,
        name: str,
        address: str,
        country: str
    ):
        norm_n, stem_n, toks_n = normalize_business_name(name)
        norm_a, toks_a, pin, nums = normalize_address(address)

        prepped = (norm_n, stem_n, toks_n, norm_a, toks_a, pin, nums, country)
        self.target_data[entity_id] = prepped

        c_index = self.indices[country]
        bkeys = extract_blocking_keys_v2(stem_n, toks_n, norm_a, toks_a, pin, nums)

        for k, _ in bkeys:
            pl = c_index[k]
            if len(pl) < self.max_posting_size:
                pl.append(entity_id)

    def retrieve_candidates(
        self,
        s1_prep: Tuple,
        country: str,
        max_candidates: int = 30,
        min_score: float = 2.0
    ) -> List[str]:
        """
        Retrieves top candidate IDs for a query entity.
        Returns ranked list of target IDs.
        """
        c_index = self.indices.get(country)
        if not c_index:
            return []

        (norm_n, stem_n, toks_n, norm_a, toks_a, pin, nums) = s1_prep[:7]
        bkeys = extract_blocking_keys_v2(stem_n, toks_n, norm_a, toks_a, pin, nums)

        scores = collections.defaultdict(float)
        exact_hits = set()
        strong_addr_hits = set()

        for k, weight in bkeys:
            pl = c_index.get(k)
            if not pl:
                continue
            discount = 1.0 / (1.0 + 0.03 * len(pl))
            effective_w = weight * discount

            for tid in pl:
                scores[tid] += effective_w
                if k.startswith('nx:'):
                    exact_hits.add(tid)
                elif k.startswith('anw:') or k.startswith('anp:'):
                    strong_addr_hits.add(tid)

        if not scores:
            return []

        # Prioritize exact normalized name hits
        for tid in exact_hits:
            scores[tid] += 25.0

        # Prioritize strong spatial hits
        for tid in strong_addr_hits:
            scores[tid] += 10.0

        filtered = [(tid, sc) for tid, sc in scores.items() if sc >= min_score]
        filtered.sort(key=lambda x: x[1], reverse=True)

        return [tid for tid, _ in filtered[:max_candidates]]
