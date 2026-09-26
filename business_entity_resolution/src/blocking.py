#!/usr/bin/env python3
"""
Phase 3: Candidate Generation / Blocking System
Multi-stage blocking pipeline combining:
1. Exact normalized name
2. Name prefix and token blocking
3. Concatenated name blocking (handles spacing/punctuation and web domains)
4. Address token blocking
5. Postal/PIN code blocking
6. Compound building number + locality blocking
7. Country-aware partitioning (open-set: works for any country including US, India, France)
8. Frequency-capped posting lists to eliminate stopword noise
9. Candidate relevance scoring with inverse document frequency discounting
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

def extract_blocking_keys(
    stem_name: str,
    name_tokens: List[str],
    norm_addr: str,
    addr_tokens: List[str],
    postal_code: Optional[str],
    numeric_info: Set[str]
) -> List[Tuple[str, float]]:
    """
    Generates blocking keys with intrinsic relevance weights.
    Returns: list of (key_string, weight)
    """
    keys = []
    
    # 1. Exact normalized stem name (high confidence)
    if stem_name and len(stem_name) >= 3:
        keys.append(('nx:' + stem_name, 12.0))
        
    # 2. Concatenated name string (handles merged words, domains like company.com)
    if len(name_tokens) >= 2:
        concat = ''.join(name_tokens)
        if len(concat) >= 5:
            keys.append(('nc:' + concat[:16], 10.0))
    elif stem_name and len(stem_name) >= 5:
        concat = ''.join(stem_name.split())
        keys.append(('nc:' + concat[:16], 9.0))
        
    # 3. Individual distinctive name tokens (length >= 3, non-generic)
    for tok in name_tokens:
        if len(tok) >= 3 and tok not in LEGAL_TOKENS_SET:
            w = 7.0 if len(tok) >= 6 else (5.0 if len(tok) >= 4 else 3.0)
            keys.append(('nt:' + tok, w))
            
    # 4. Name prefix (for slight typos in first word)
    if name_tokens:
        first = name_tokens[0]
        if len(first) >= 5 and first not in LEGAL_TOKENS_SET:
            keys.append(('np:' + first[:4], 3.0))
            
    # 5. Compound Address: Number + Locality/Word (strong spatial signal)
    sig_addr = [w for w in addr_tokens if len(w) >= 4 and w not in ADDR_STOPWORDS][:4]
    nums = sorted(list(numeric_info))[:2]
    
    for n in nums:
        for w in sig_addr[:2]:
            keys.append((f'anw:{n}_{w}', 7.0))
        if postal_code:
            keys.append((f'anp:{n}_{postal_code}', 8.0))
            
    # 6. Compound Address words (when name is very brief or missing)
    if len(sig_addr) >= 2 and (not stem_name or len(name_tokens) <= 1):
        keys.append((f'amw:{sig_addr[0]}_{sig_addr[1]}', 6.0))
        
    return keys

class CountryBlockingIndex:
    """
    Inverted index for target records partitioned strictly by country.
    Guarantees country consistency, sub-second lookup, and bounded memory usage.
    """
    def __init__(self, max_posting_size: int = 250):
        self.max_posting_size = max_posting_size
        # index[country][key] -> list of target_id
        self.indices: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        # target_data[target_id] -> preprocessed entity tuple
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
        bkeys = extract_blocking_keys(stem_n, toks_n, norm_a, toks_a, pin, nums)
        
        for k, _ in bkeys:
            pl = c_index[k]
            if len(pl) < self.max_posting_size:
                pl.append(entity_id)

    def retrieve_candidates(
        self,
        s1_preprocessed: Tuple,
        country: str,
        max_candidates: int = 25,
        min_score: float = 2.0
    ) -> List[str]:
        """
        Retrieves top candidate IDs for a query entity.
        Returns ranked list of target IDs.
        """
        c_index = self.indices.get(country)
        if not c_index:
            return []
            
        (norm_n, stem_n, toks_n, norm_a, toks_a, pin, nums) = s1_preprocessed
        bkeys = extract_blocking_keys(stem_n, toks_n, norm_a, toks_a, pin, nums)
        
        scores = collections.defaultdict(float)
        exact_hits = set()
        
        for k, weight in bkeys:
            pl = c_index.get(k)
            if not pl:
                continue
            # Apply posting list length discount (penalize overly frequent tokens)
            discount = 1.0 / (1.0 + 0.05 * len(pl))
            effective_w = weight * discount
            
            for tid in pl:
                scores[tid] += effective_w
                if k.startswith('nx:'):
                    exact_hits.add(tid)
                    
        if not scores:
            return []
            
        # Guarantee exact normalized name matches are prioritized
        for tid in exact_hits:
            scores[tid] += 20.0
            
        filtered = [(tid, sc) for tid, sc in scores.items() if sc >= min_score]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return [tid for tid, _ in filtered[:max_candidates]]

if __name__ == "__main__":
    print("Testing CountryBlockingIndex...")
    cbi = CountryBlockingIndex()
    cbi.add_target_entity("S2-001", "Payne Enterprises Inc.", "3315 Fremont St, Peoria, IL", "US")
    cbi.add_target_entity("S2-002", "Payne Bakery", "12 Main St, Peoria, IL", "US")
    cbi.add_target_entity("S3-001", "Payne Énterprises", "3315 Fremont St, Peoria, Illinois", "US")
    
    n_n, s_n, t_n = normalize_business_name("Payne Enterprises")
    n_a, t_a, pin, nums = normalize_address("3315 Fremont Street, Peoria, IL")
    s1_prep = (n_n, s_n, t_n, n_a, t_a, pin, nums)
    
    cands = cbi.retrieve_candidates(s1_prep, "US", max_candidates=10)
    print("Retrieved candidates for Payne Enterprises (US):", cands)
