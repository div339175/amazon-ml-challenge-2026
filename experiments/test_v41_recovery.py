#!/usr/bin/env python3
"""
Test recovery of the 409 missed true pairs using:
1. Indic script / phonetic transliteration normalization
2. Address-centric compound key (house_num + street_token / city)
3. Name-token prefix / rare token matching for empty/truncated addresses
"""

import os
import sys
import re
import time
import random
import collections
import unidecode
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
    LEGAL_TOKENS_SET
)
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.blocking_v4 import (
    CountryBlockingIndexV4,
    clean_ocr_leetspeak,
    normalize_indic_phonetics,
    get_consonants
)

ADDR_STOPWORDS = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "pl", "fl", "bldg", "rm", "apt", "ste", "near", "opp", "north", "south", "east", "west"}

INDIAN_HONORIFICS = {"smt", "sri", "shri", "shree", "ms", "messrs", "dr", "prof"}

def clean_indic_name(raw_name: str) -> tuple:
    """Cleans Indic script names, removes honorifics, transliterates to phonetic Latin."""
    # 1. Transliterate if non-ascii
    u = unidecode.unidecode(raw_name).lower()
    # 2. Leetspeak clean
    u = clean_ocr_leetspeak(u)
    # 3. Clean legal forms and honorifics
    u = re.sub(r'[^a-z0-9\s]', ' ', u)
    toks = [t for t in u.split() if len(t) >= 2]
    # Strip leading honorifics
    if toks and toks[0] in INDIAN_HONORIFICS:
        toks = toks[1:]
    # Stem / clean legal tokens
    stem_toks = [t for t in toks if t not in LEGAL_TOKENS_SET and t not in {"picinnns", "piraiveett", "limittett", "limittedd", "bilddrs", "bhaart"}]
    stem_str = " ".join(stem_toks)
    consonants = get_consonants(stem_str)
    return " ".join(toks), stem_str, consonants

def extract_address_compounds(addr_norm: str, nums: set, country: str) -> list:
    """Extracts high-specificity house_num + street_token compound keys."""
    keys = []
    if not nums or not addr_norm:
        return keys
    toks = [t for t in addr_norm.split() if len(t) >= 3 and not t.isdigit() and t not in ADDR_STOPWORDS]
    # Street / distinctive words
    for n in nums:
        for t in toks[:3]:  # First 3 distinctive street tokens
            keys.append(f"{n}_{t}")
    return keys

def main():
    print("=" * 70)
    print("TESTING V4.1 TARGETED RECOVERY CHANNELS ON 409 MISSED PAIRS")
    print("=" * 70)
    random.seed(42)

    # 1. Load exact 3,000 validation queries
    gt_all = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
            gt_all[p[0].strip()] = mids
            if idx >= 150000: break

    s1_all = list(gt_all.keys())
    random.shuffle(s1_all)
    val_s1_ids = s1_all[:3000]
    gt_val = {s: gt_all[s] for s in val_s1_ids}

    # Load S1 records
    needed_s1 = set(val_s1_ids)
    s1_records = {}
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0].strip()
            if eid in needed_s1:
                s1_records[eid] = p
                if len(s1_records) == len(needed_s1): break

    s1_prepped = {}
    for sid in val_s1_ids:
        p = s1_records[sid]
        nn, sn, tn = normalize_business_name(p[1])
        na, ta, pin, nums = normalize_address(p[2])
        ctry = p[3].strip() if len(p) > 3 else "US"
        u_norm, u_stem, u_cons = clean_indic_name(p[1])
        s1_prepped[sid] = (nn, sn, tn, na, ta, pin, nums, ctry, p[1], p[2], u_norm, u_stem, u_cons)

    val_true_targets = set()
    for mids in gt_val.values(): val_true_targets |= mids

    # Load Targets: true targets + 400k distractors
    target_records = {}
    bg_limit = 400000
    bg_added = 0
    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0].strip()
                if eid in val_true_targets:
                    target_records[eid] = p
                elif bg_added < bg_limit:
                    target_records[eid] = p
                    bg_added += 1

    # Build V4 indices + NEW Targeted V4.1 channels
    print("\nBuilding V4 + Targeted Recovery Indices...")
    cbi_v2 = CountryBlockingIndexV2(max_posting_size=350)
    cbi_v4 = CountryBlockingIndexV4()
    exact_name_index = collections.defaultdict(lambda: collections.defaultdict(list))
    exact_addr_index = collections.defaultdict(lambda: collections.defaultdict(list))
    pin_num_index = collections.defaultdict(lambda: collections.defaultdict(list))

    # Channel A: Indic Cleaned Consonant Skeleton & Stem index
    indic_cons_index = collections.defaultdict(lambda: collections.defaultdict(list))
    indic_stem_index = collections.defaultdict(lambda: collections.defaultdict(list))

    # Channel B: House Number + Street Token Compound Index (cap 50)
    house_street_index = collections.defaultdict(lambda: collections.defaultdict(list))

    # Channel C: First 2 Name Tokens Index for Missing Address (cap 100)
    name_prefix2_index = collections.defaultdict(lambda: collections.defaultdict(list))

    for eid, p in target_records.items():
        name = p[1] if len(p) > 1 else ""
        addr = p[2] if len(p) > 2 else ""
        ctry = p[3].strip() if len(p) > 3 else "US"
        cbi_v2.add_target_entity(eid, name, addr, ctry)
        cbi_v4.add_target_entity(eid, name, addr, ctry)

        nn, sn, tn = normalize_business_name(name)
        na, ta, pin, nums = normalize_address(addr)

        if nn and len(nn) >= 3:
            exact_name_index[ctry][nn].append(eid)
        if na and len(na) >= 5:
            exact_addr_index[ctry][na].append(eid)
        if pin and nums:
            for n in nums:
                pin_num_index[ctry][f"{pin}_{n}"].append(eid)

        # V4.1 Channel A: Indic cleaned
        u_norm, u_stem, u_cons = clean_indic_name(name)
        if u_cons and len(u_cons) >= 4:
            indic_cons_index[ctry][u_cons].append(eid)
        if u_stem and len(u_stem) >= 4:
            indic_stem_index[ctry][u_stem].append(eid)

        # V4.1 Channel B: House + Street compounds
        compounds = extract_address_compounds(na, nums, ctry)
        for comp in compounds:
            house_street_index[ctry][comp].append(eid)

        # V4.1 Channel C: Name 2-token prefix
        if tn and len(tn) >= 2:
            p2 = f"{tn[0]}_{tn[1]}"
            name_prefix2_index[ctry][p2].append(eid)
        elif tn and len(tn) == 1 and len(tn[0]) >= 4:
            name_prefix2_index[ctry][tn[0]].append(eid)

    print("Indices built successfully.")

    # Retrieve with V4 vs V4.1
    v4_hits = 0
    v41_hits = 0
    tot_true = 0

    recovered_a = 0
    recovered_b = 0
    recovered_c = 0

    cands_count_v4 = 0
    cands_count_v41 = 0

    for sid in val_s1_ids:
        prep = s1_prepped[sid]
        ctry = prep[7]
        true_m = gt_val[sid]
        tot_true += len(true_m)

        # V4 retrieval
        (nn, sn, tn, na, ta, pin, nums) = prep[:7]
        c2 = cbi_v2.retrieve_candidates(prep[:7], ctry, max_candidates=25)
        c4 = cbi_v4.retrieve_candidates(prep[:7], ctry, max_candidates=25)
        c_exact_n = exact_name_index[ctry].get(nn, [])[:50]
        c_exact_a = exact_addr_index[ctry].get(na, [])[:50]
        c_pin_num = []
        if pin and nums:
            for n in nums:
                c_pin_num.extend(pin_num_index[ctry].get(f"{pin}_{n}", [])[:30])

        counts_v4 = collections.Counter(c2 + c4 + c_exact_n + c_exact_a + c_pin_num)
        scored_v4 = sorted(counts_v4.items(), key=lambda x: x[1], reverse=True)[:35]
        v4_cands = {tid for tid, _ in scored_v4}
        cands_count_v4 += len(v4_cands)

        cur_v4_hits = len(true_m & v4_cands)
        v4_hits += cur_v4_hits

        # V4.1 retrieval: Add targeted channels
        u_norm, u_stem, u_cons = prep[10], prep[11], prep[12]
        c_indic_c = indic_cons_index[ctry].get(u_cons, [])[:30] if u_cons else []
        c_indic_s = indic_stem_index[ctry].get(u_stem, [])[:30] if u_stem else []

        compounds = extract_address_compounds(na, nums, ctry)
        c_house_street = []
        for comp in compounds:
            c_house_street.extend(house_street_index[ctry].get(comp, [])[:20])

        c_name_prefix = []
        if not na or len(na) < 10:
            if tn and len(tn) >= 2:
                p2 = f"{tn[0]}_{tn[1]}"
                c_name_prefix.extend(name_prefix2_index[ctry].get(p2, [])[:30])

        counts_v41 = collections.Counter(
            c2 + c4 + c_exact_n + c_exact_a + c_pin_num +
            c_indic_c + c_indic_s + c_house_street + c_name_prefix
        )

        scored_v41 = sorted(counts_v41.items(), key=lambda x: x[1], reverse=True)[:45]
        v41_cands = {tid for tid, _ in scored_v41}
        cands_count_v41 += len(v41_cands)

        cur_v41_hits = len(true_m & v41_cands)
        v41_hits += cur_v41_hits

        # Check what channel recovered any new hits
        new_hits = (true_m & v41_cands) - v4_cands
        for nh in new_hits:
            if nh in set(c_indic_c + c_indic_s): recovered_a += 1
            if nh in set(c_house_street): recovered_b += 1
            if nh in set(c_name_prefix): recovered_c += 1

    print("\n" + "=" * 70)
    print("V4 vs V4.1 RECALL COMPARISON RESULTS")
    print("=" * 70)
    print(f"Total True Pairs: {tot_true:,}")
    print(f"V4 Candidate Recall:   {(v4_hits / tot_true)*100:.2f}% ({v4_hits:,}/{tot_true:,}) | Avg Candidates/S1: {cands_count_v4/len(val_s1_ids):.1f}")
    print(f"V4.1 Candidate Recall: {(v41_hits / tot_true)*100:.2f}% ({v41_hits:,}/{tot_true:,}) | Avg Candidates/S1: {cands_count_v41/len(val_s1_ids):.1f}")
    print(f"Net Newly Recovered True Pairs: +{v41_hits - v4_hits:,} ({(v41_hits-v4_hits)/tot_true*100:.2f}% increase)")
    print(f"  - Channel A (Indic/Phonetic): +{recovered_a} recovered")
    print(f"  - Channel B (House+Street/DBA): +{recovered_b} recovered")
    print(f"  - Channel C (Truncated Name Prefix): +{recovered_c} recovered")

if __name__ == "__main__":
    main()
