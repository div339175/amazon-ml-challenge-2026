#!/usr/bin/env python3
"""
Phase 2: V4 High-Recall Multi-Pass Blocking Index
Amazon ML Challenge 2026 - Business Entity Resolution

Comprehensive Channels:
1. Channel A: Exact Normalized Name (cap 2,000)
2. Channel B: Exact Normalized Address (cap 2,000)
3. Channel C: House Number + Distinctive Street/Locality Tokens (cap 500)
4. Channel D: House Number + City / Region / State Token (num_loc) (cap 500)
5. Channel E: House Number + Postal/PIN (cap 500)
6. Channel F: Name Stem + Postal/PIN (cap 500)
7. Channel G: Rare Distinctive Name Tokens (IDF-weighted, cap 400)
8. Channel H: Rare Distinctive Address Tokens (IDF-weighted, cap 400)
9. Channel I: Compound Distinctive Address Words (cap 400)
10. Channel J: Compound Name + Address Token (cap 400)
11. Channel K: Transliteration & Phonetic Name Tokens (cap 400)
12. Channel L: Word Consonant Skeletons (cap 400)
13. Channel M: Name Prefix (4 chars) (cap 350)
"""

import collections
import math
import re
import unidecode
from typing import Dict, List, Tuple, Set, Optional

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
    LEGAL_TOKENS_SET,
)

LEETSPEAK_TABLE = str.maketrans({
    "5": "s", "6": "g", "0": "o", "1": "i", "3": "e", "@": "a", "$": "s",
    "+": " ", "-": " ", "#": " ", "_": " "
})

# City & State tokens for spatial num_loc pairing
MAJOR_CITIES_STATES = {
    # India
    'delhi', 'mumbai', 'bangalore', 'bengaluru', 'chennai', 'madras', 'hyderabad',
    'kolkata', 'calcutta', 'pune', 'ahmedabad', 'coimbatore', 'jaipur', 'surat',
    'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
    'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad',
    'meerut', 'rajkot', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar',
    'navi', 'allahabad', 'prayagraj', 'ranchi', 'howrah', 'jabalpur', 'gwalior',
    'vijayawada', 'jodhpur', 'madurai', 'raipur', 'kota', 'guwahati', 'chandigarh',
    'solapur', 'hubli', 'dharwad', 'bareilly', 'moradabad', 'mysore', 'gurgaon',
    'gurugram', 'aligarh', 'jalandhar', 'tiruchirappalli', 'bhubaneswar', 'salem',
    'warangal', 'mira', 'bhayandar', 'jalgaon', 'guntur', 'thiruvananthapuram',
    'tamil', 'nadu', 'maharashtra', 'karnataka', 'gujarat', 'telangana', 'andhra',
    'pradesh', 'bengal', 'rajasthan', 'kerala', 'punjab', 'haryana', 'bihar',
    # US
    'york', 'chicago', 'houston', 'phoenix', 'philadelphia', 'antonio', 'diego',
    'dallas', 'jose', 'austin', 'jacksonville', 'fort', 'worth', 'columbus',
    'charlotte', 'francisco', 'indianapolis', 'seattle', 'denver', 'washington',
    'boston', 'el', 'paso', 'nashville', 'detroit', 'oklahoma', 'portland',
    'las', 'vegas', 'memphis', 'louisville', 'baltimore', 'milwaukee', 'albuquerque',
    'tucson', 'fresno', 'sacramento', 'kansas', 'mesa', 'atlanta', 'omaha',
    'colorado', 'raleigh', 'miami', 'virginia', 'oakland', 'minneapolis', 'tulsa',
    'california', 'texas', 'florida', 'illinois', 'pennsylvania', 'ohio', 'georgia',
    'carolina', 'michigan', 'tennessee', 'indiana', 'missouri', 'maryland', 'wisconsin',
    # France
    'paris', 'marseille', 'lyon', 'toulouse', 'nice', 'nantes', 'strasbourg',
    'montpellier', 'bordeaux', 'lille', 'rennes', 'reims', 'saint', 'etienne',
    'toulon', 'le', 'havre', 'grenoble', 'dijon', 'angers', 'nimes', 'villeurbanne'
}

ADDR_GENERIC_STOPWORDS = {
    'st', 'rd', 'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'ter', 'hwy',
    'apt', 'ste', 'fl', 'bldg', 'rm', 'n', 's', 'e', 'w', 'opp', 'near',
    'pobox', 'rue', 'chemin', 'allee', 'state', 'city', 'dist', 'district',
    'region', 'floor', 'suite', 'block', 'phase', 'sector', 'nagar', 'colony',
    'road', 'street', 'main', 'cross', 'layout', 'lane', 'court', 'circle',
    'building', 'house', 'no', 'hno', 'plot', 'flat', 'part', 'shop', 'door'
} | MAJOR_CITIES_STATES

GENERIC_NAME_STOPWORDS = {
    'inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co', 'limited',
    'private', 'pvt', 'services', 'enterprises', 'solutions', 'holdings',
    'group', 'industries', 'international', 'systems', 'consulting', 'management',
    'technologies', 'store', 'shop', 'market', 'hotel', 'restaurant', 'cafe',
    'express', 'center', 'centre', 'trading', 'agency', 'associates', 'care',
    'service', 'global', 'prime', 'national', 'world', 'enterprises', 'business',
    'products', 'developers', 'finance', 'logistics', 'hospitality', 'infrastructure'
}

def clean_ocr_leetspeak(s: str) -> str:
    if not s: return ""
    cleaned = s.lower().translate(LEETSPEAK_TABLE)
    return " ".join(cleaned.split())

def normalize_indic_phonetics(s: str) -> str:
    """Normalizes Indic/transliterated phonetic variation (k->c, ph->f, ii->ee, etc.)."""
    if not s: return ""
    t = unidecode.unidecode(s).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = t.replace("ph", "f").replace("sh", "s").replace("ii", "ee").replace("oo", "u")
    t = t.replace("k", "c").replace("v", "w").replace("z", "j")
    # collapse consecutive repeated letters
    t = re.sub(r"(.)\1+", r"\1", t)
    return " ".join(t.split())

def get_consonants(s: str) -> str:
    return "".join([c for c in s.lower() if c in "bcdfghjklmnpqrstvwxyz"])

class CountryBlockingIndexV4:
    def __init__(self):
        self.indices: Dict[str, Dict[str, Dict[str, List[str]]]] = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list))
        )
        self.target_data: Dict[str, Tuple] = {}
        self.token_dfs: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.doc_counts: Dict[str, int] = collections.defaultdict(int)

    def add_target_entity(
        self,
        entity_id: str,
        name: str,
        address: str,
        country: str
    ):
        norm_n, stem_n, toks_n = normalize_business_name(name)
        norm_a, toks_a, pin, nums = normalize_address(address)
        clean_n = clean_ocr_leetspeak(norm_n)
        clean_toks_n = [t for t in clean_n.split() if t not in LEGAL_TOKENS_SET and len(t) >= 2]
        phon_n = normalize_indic_phonetics(name)
        phon_toks_n = [t for t in phon_n.split() if t not in LEGAL_TOKENS_SET and len(t) >= 3]

        prepped = (norm_n, stem_n, toks_n, norm_a, toks_a, pin, nums, country, clean_n, clean_toks_n, phon_n, phon_toks_n)
        self.target_data[entity_id] = prepped

        self.doc_counts[country] += 1
        dfs = self.token_dfs[country]
        for t in toks_n: dfs[t] += 1
        for t in clean_toks_n: dfs[t] += 1
        for t in toks_a: dfs[t] += 1

        c_channels = self.indices[country]

        # Channel A: Exact Normalized Name (cap 2,000)
        if norm_n and len(norm_n) >= 3:
            pl = c_channels["exact_name"][norm_n]
            if len(pl) < 2000: pl.append(entity_id)
        if clean_n and clean_n != norm_n and len(clean_n) >= 3:
            pl = c_channels["exact_name"][clean_n]
            if len(pl) < 2000: pl.append(entity_id)

        # Channel B: Exact Normalized Address (cap 2,000)
        if norm_a and len(norm_a) >= 5:
            pl = c_channels["exact_addr"][norm_a]
            if len(pl) < 2000: pl.append(entity_id)

        # Address tokens
        sig_addr = [w for w in toks_a if len(w) >= 4 and w not in ADDR_GENERIC_STOPWORDS]
        sig_addr_sorted = sorted(sig_addr, key=lambda x: len(x), reverse=True)[:5]
        city_tokens = [w for w in toks_a if w in MAJOR_CITIES_STATES][:2]

        # Channel C: House Number + Distinctive Street/Locality Tokens (cap 500)
        for num in sorted(list(nums))[:2]:
            for w in sig_addr_sorted:
                k = f"{num}_{w}"
                pl = c_channels["num_addr"][k]
                if len(pl) < 500: pl.append(entity_id)

            # Channel D: House Number + City / State (cap 500) - Catches minimal addresses!
            for loc in city_tokens:
                k = f"{num}_{loc}"
                pl = c_channels["num_loc"][k]
                if len(pl) < 500: pl.append(entity_id)

            # Channel E: House Number + Postal/PIN (cap 500)
            if pin:
                k = f"{num}_{pin}"
                pl = c_channels["num_pin"][k]
                if len(pl) < 500: pl.append(entity_id)

        # Channel F: Name Stem + Postal/PIN (cap 500)
        if pin and stem_n:
            k = f"{pin}_{stem_n[:10]}"
            pl = c_channels["name_pin"][k]
            if len(pl) < 500: pl.append(entity_id)

        # Channel G: Rare Distinctive Name Tokens (cap 400)
        for t in set(toks_n) | set(clean_toks_n):
            if len(t) >= 4 and t not in GENERIC_NAME_STOPWORDS:
                pl = c_channels["name_tokens"][t]
                if len(pl) < 400: pl.append(entity_id)

        # Channel H: Rare Distinctive Address Tokens (cap 400)
        for t in sig_addr_sorted:
            pl = c_channels["addr_tokens"][t]
            if len(pl) < 400: pl.append(entity_id)

        # Channel I: Compound Address Words (cap 400)
        if len(sig_addr_sorted) >= 2:
            pair_k = f"{sig_addr_sorted[0]}_{sig_addr_sorted[1]}"
            pl = c_channels["compound_addr"][pair_k]
            if len(pl) < 400: pl.append(entity_id)

        # Channel J: Compound Name + Address Token (cap 400)
        first_n = [t for t in clean_toks_n if t not in GENERIC_NAME_STOPWORDS]
        if first_n and sig_addr_sorted:
            k = f"{first_n[0][:6]}_{sig_addr_sorted[0][:6]}"
            pl = c_channels["compound_na"][k]
            if len(pl) < 400: pl.append(entity_id)

        # Channel K: Transliteration & Phonetic Name Tokens (cap 400)
        for t in phon_toks_n:
            if len(t) >= 4 and t not in GENERIC_NAME_STOPWORDS:
                pl = c_channels["phonetic_tokens"][t]
                if len(pl) < 400: pl.append(entity_id)

        # Channel L: Word Consonant Skeletons (cap 400)
        for t in (first_n[:2] if first_n else clean_toks_n[:2]):
            skel = get_consonants(t)
            if len(skel) >= 3:
                pl = c_channels["consonant_skel"][skel]
                if len(pl) < 400: pl.append(entity_id)

        # Channel M: Name Prefix (4 chars)
        if first_n and len(first_n[0]) >= 4:
            pfx = first_n[0][:4]
            pl = c_channels["name_prefix"][pfx]
            if len(pl) < 350: pl.append(entity_id)

    def retrieve_candidates(
        self,
        s1_prep: Tuple,
        country: str,
        max_candidates: int = 50
    ) -> List[str]:
        c_channels = self.indices.get(country)
        if not c_channels:
            return []

        (norm_n, stem_n, toks_n, norm_a, toks_a, pin, nums) = s1_prep[:7]
        clean_n = clean_ocr_leetspeak(norm_n)
        clean_toks_n = [t for t in clean_n.split() if t not in LEGAL_TOKENS_SET and len(t) >= 2]
        phon_n = normalize_indic_phonetics(norm_n)
        phon_toks_n = [t for t in phon_n.split() if t not in LEGAL_TOKENS_SET and len(t) >= 3]

        channel_hits = collections.defaultdict(float)
        channel_counts = collections.defaultdict(int)
        exact_name_hits = set()
        exact_addr_hits = set()

        dfs = self.token_dfs[country]
        total_docs = max(1, self.doc_counts[country])

        # 1. Exact Name Channel (weight 30.0)
        for n_key in set([norm_n, clean_n]):
            if n_key and n_key in c_channels["exact_name"]:
                for tid in c_channels["exact_name"][n_key]:
                    channel_hits[tid] += 30.0
                    channel_counts[tid] += 1
                    exact_name_hits.add(tid)

        # 2. Exact Address Channel (weight 25.0)
        if norm_a and norm_a in c_channels["exact_addr"]:
            for tid in c_channels["exact_addr"][norm_a]:
                channel_hits[tid] += 25.0
                channel_counts[tid] += 1
                exact_addr_hits.add(tid)

        # Address distinctive words
        sig_addr = [w for w in toks_a if len(w) >= 4 and w not in ADDR_GENERIC_STOPWORDS]
        sig_addr_sorted = sorted(sig_addr, key=lambda x: len(x), reverse=True)[:5]
        city_tokens = [w for w in toks_a if w in MAJOR_CITIES_STATES][:2]

        # 3. House Number + Address Token / Loc / PIN Channel
        for num in sorted(list(nums))[:2]:
            for w in sig_addr_sorted:
                k = f"{num}_{w}"
                if k in c_channels["num_addr"]:
                    for tid in c_channels["num_addr"][k]:
                        channel_hits[tid] += 16.0
                        channel_counts[tid] += 1

            for loc in city_tokens:
                k = f"{num}_{loc}"
                if k in c_channels["num_loc"]:
                    for tid in c_channels["num_loc"][k]:
                        channel_hits[tid] += 14.0
                        channel_counts[tid] += 1

            if pin:
                k = f"{num}_{pin}"
                if k in c_channels["num_pin"]:
                    for tid in c_channels["num_pin"][k]:
                        channel_hits[tid] += 18.0
                        channel_counts[tid] += 1

        # 4. Name Stem + Postal/PIN (weight 16.0)
        if pin and stem_n:
            k = f"{pin}_{stem_n[:10]}"
            if k in c_channels["name_pin"]:
                for tid in c_channels["name_pin"][k]:
                    channel_hits[tid] += 16.0
                    channel_counts[tid] += 1

        # 5. Compound Address Words (weight 14.0)
        if len(sig_addr_sorted) >= 2:
            pair_k = f"{sig_addr_sorted[0]}_{sig_addr_sorted[1]}"
            if pair_k in c_channels["compound_addr"]:
                for tid in c_channels["compound_addr"][pair_k]:
                    channel_hits[tid] += 14.0
                    channel_counts[tid] += 1

        # 6. Compound Name + Address Token (weight 15.0)
        first_n = [t for t in clean_toks_n if t not in GENERIC_NAME_STOPWORDS]
        if first_n and sig_addr_sorted:
            k = f"{first_n[0][:6]}_{sig_addr_sorted[0][:6]}"
            if k in c_channels["compound_na"]:
                for tid in c_channels["compound_na"][k]:
                    channel_hits[tid] += 15.0
                    channel_counts[tid] += 1

        # 7. Rare Distinctive Name Tokens with IDF (weight up to 14.0)
        for t in set(toks_n) | set(clean_toks_n):
            if len(t) >= 4 and t not in GENERIC_NAME_STOPWORDS:
                if t in c_channels["name_tokens"]:
                    pl = c_channels["name_tokens"][t]
                    doc_freq = max(1, dfs[t])
                    idf = math.log((total_docs + 1) / (doc_freq + 1))
                    w = min(14.0, max(2.5, idf * 1.6))
                    for tid in pl:
                        channel_hits[tid] += w
                        channel_counts[tid] += 1

        # 8. Rare Distinctive Address Tokens with IDF (weight up to 12.0)
        for t in sig_addr_sorted:
            if t in c_channels["addr_tokens"]:
                pl = c_channels["addr_tokens"][t]
                doc_freq = max(1, dfs[t])
                idf = math.log((total_docs + 1) / (doc_freq + 1))
                w = min(12.0, max(2.0, idf * 1.3))
                for tid in pl:
                    channel_hits[tid] += w
                    channel_counts[tid] += 1

        # 9. Transliteration & Phonetic Tokens (weight 12.0)
        for t in phon_toks_n:
            if len(t) >= 4 and t not in GENERIC_NAME_STOPWORDS:
                if t in c_channels["phonetic_tokens"]:
                    for tid in c_channels["phonetic_tokens"][t]:
                        channel_hits[tid] += 11.0
                        channel_counts[tid] += 1

        # 10. Consonant Skeletons (weight 9.0)
        for t in (first_n[:2] if first_n else clean_toks_n[:2]):
            skel = get_consonants(t)
            if len(skel) >= 3 and skel in c_channels["consonant_skel"]:
                for tid in c_channels["consonant_skel"][skel]:
                    channel_hits[tid] += 9.0
                    channel_counts[tid] += 1

        # 11. Name Prefix (weight 5.0)
        if first_n and len(first_n[0]) >= 4:
            pfx = first_n[0][:4]
            if pfx in c_channels["name_prefix"]:
                for tid in c_channels["name_prefix"][pfx]:
                    channel_hits[tid] += 5.0
                    channel_counts[tid] += 1

        if not channel_hits:
            return []

        ranked = []
        for tid, base_score in channel_hits.items():
            num_ch = channel_counts[tid]
            bonus = 15.0 if num_ch >= 4 else (10.0 if num_ch >= 3 else (5.0 if num_ch >= 2 else 0.0))
            score = base_score + bonus

            if tid in exact_name_hits and tid in exact_addr_hits:
                score += 50.0
            elif tid in exact_name_hits:
                score += 20.0
            elif tid in exact_addr_hits:
                score += 15.0

            ranked.append((tid, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in ranked[:max_candidates]]
