#!/usr/bin/env python3
"""
Audit V4 False Negatives and Missed Candidates
Amazon ML Challenge 2026 - Business Entity Resolution

Categorizes missed true matches into:
1. Transliteration gaps (Indic script, phonetic differences)
2. DBA / holding-company names (different names, matching addresses)
3. Severely truncated address (missing/short addresses, name overlap)
4. Other
"""

import os
import sys
import time
import math
import random
import collections
import unicodedata
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import unidecode

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.blocking_v4 import (
    CountryBlockingIndexV4,
    clean_ocr_leetspeak,
    normalize_indic_phonetics,
    get_consonants
)

def is_indic(text: str) -> bool:
    for ch in text:
        # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
        cp = ord(ch)
        if (0x0900 <= cp <= 0x0D7F):
            return True
    return False

def main():
    print("=" * 70)
    print("AUDITING V4 FALSE NEGATIVES & CANDIDATE GENERATION LOSSES")
    print("=" * 70)
    random.seed(42)

    # 1. Load exact same 3,000 validation queries
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
        s1_prepped[sid] = (nn, sn, tn, na, ta, pin, nums, ctry, p[1], p[2])

    val_true_targets = set()
    for mids in gt_val.values(): val_true_targets |= mids

    # Load Targets: true targets + 400k distractors (sufficient for FN analysis)
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

    # Multi-Channel Indices from V4
    cbi_v2 = CountryBlockingIndexV2(max_posting_size=350)
    cbi_v4 = CountryBlockingIndexV4()
    exact_name_index = collections.defaultdict(lambda: collections.defaultdict(list))
    exact_addr_index = collections.defaultdict(lambda: collections.defaultdict(list))
    pin_num_index = collections.defaultdict(lambda: collections.defaultdict(list))

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

    def retrieve_union_v4(s1_prep, country, top_k=35):
        (nn, sn, tn, na, ta, pin, nums) = s1_prep[:7]
        c2 = cbi_v2.retrieve_candidates(s1_prep[:7], country, max_candidates=25)
        c4 = cbi_v4.retrieve_candidates(s1_prep[:7], country, max_candidates=25)
        c_exact_n = exact_name_index[country].get(nn, [])[:50]
        c_exact_a = exact_addr_index[country].get(na, [])[:50]
        c_pin_num = []
        if pin and nums:
            for n in nums:
                c_pin_num.extend(pin_num_index[country].get(f"{pin}_{n}", [])[:30])

        counts = collections.Counter(c2 + c4 + c_exact_n + c_exact_a + c_pin_num)
        scored = []
        for tid, cnt in counts.items():
            sc = cnt * 10.0
            if tid in c_exact_n: sc += 30.0
            if tid in c_exact_a: sc += 25.0
            scored.append((tid, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in scored[:top_k]]

    missed_pairs = []
    total_true = 0

    for sid in val_s1_ids:
        prep = s1_prepped[sid]
        ctry = prep[7]
        true_m = gt_val[sid]
        total_true += len(true_m)
        cands = set(retrieve_union_v4(prep, ctry, top_k=35))
        for tid in true_m:
            if tid not in cands:
                tr_p = target_records.get(tid, [])
                missed_pairs.append((sid, tid, prep, tr_p))

    print(f"Total True Pairs: {total_true:,}")
    print(f"Total Missed True Pairs in V4 Retrieval: {len(missed_pairs):,} ({(len(missed_pairs)/total_true)*100:.2f}%)")

    # Categorize misses
    cat_translit = []
    cat_dba = []
    cat_trunc_addr = []
    cat_other = []

    for sid, tid, s1_p, tr_p in missed_pairs:
        s1_raw_name = s1_p[8]
        s1_raw_addr = s1_p[9]
        tr_raw_name = tr_p[1] if len(tr_p) > 1 else ""
        tr_raw_addr = tr_p[2] if len(tr_p) > 2 else ""

        s1_nn = s1_p[0]
        s1_sn = s1_p[1]
        s1_na = s1_p[3]
        tr_nn, tr_sn, tr_tn = normalize_business_name(tr_raw_name)
        tr_na, tr_ta, tr_pin, tr_nums = normalize_address(tr_raw_addr)

        name_sim = fuzz.token_set_ratio(s1_nn, tr_nn)
        addr_sim = fuzz.token_set_ratio(s1_na, tr_na)

        has_indic = is_indic(s1_raw_name) or is_indic(tr_raw_name)
        s1_unidec = unidecode.unidecode(s1_raw_name).lower()
        tr_unidec = unidecode.unidecode(tr_raw_name).lower()
        unidec_sim = fuzz.token_set_ratio(s1_unidec, tr_unidec)

        if has_indic or (s1_p[7] == "India" and (unidec_sim > 60 or fuzz.token_sort_ratio(get_consonants(s1_sn), get_consonants(tr_sn)) > 70)):
            cat_translit.append((sid, tid, s1_raw_name, tr_raw_name, s1_raw_addr, tr_raw_addr))
        elif name_sim < 40 and (addr_sim > 65 or (s1_p[5] and s1_p[5] == tr_pin and len(s1_p[6] & tr_nums) > 0)):
            cat_dba.append((sid, tid, s1_raw_name, tr_raw_name, s1_raw_addr, tr_raw_addr))
        elif len(s1_na) < 10 or len(tr_na) < 10 or not s1_na or not tr_na:
            cat_trunc_addr.append((sid, tid, s1_raw_name, tr_raw_name, s1_raw_addr, tr_raw_addr))
        else:
            cat_other.append((sid, tid, s1_raw_name, tr_raw_name, s1_raw_addr, tr_raw_addr))

    print("\n--- Missed Pairs Categorization ---")
    print(f"1. Transliteration / Indic gaps: {len(cat_translit)} ({len(cat_translit)/len(missed_pairs)*100:.1f}%)")
    print(f"2. DBA / Disjoint Trade Names:  {len(cat_dba)} ({len(cat_dba)/len(missed_pairs)*100:.1f}%)")
    print(f"3. Truncated / Missing Address:  {len(cat_trunc_addr)} ({len(cat_trunc_addr)/len(missed_pairs)*100:.1f}%)")
    print(f"4. Other (noisy token collisions): {len(cat_other)} ({len(cat_other)/len(missed_pairs)*100:.1f}%)")

    print("\n[Samples: Transliteration Gaps]")
    for s in cat_translit[:5]:
        print(f"  S1 Name: {s[2]!r:30s} | Tr Name: {s[3]!r}")
        print(f"     S1 Addr: {s[4]!r}")
        print(f"     Tr Addr: {s[5]!r}")

    print("\n[Samples: DBA / Trade Name Discrepancies]")
    for s in cat_dba[:5]:
        print(f"  S1 Name: {s[2]!r:30s} | Tr Name: {s[3]!r}")
        print(f"     S1 Addr: {s[4]!r}")
        print(f"     Tr Addr: {s[5]!r}")

    print("\n[Samples: Truncated Address]")
    for s in cat_trunc_addr[:5]:
        print(f"  S1 Name: {s[2]!r:30s} | Tr Name: {s[3]!r}")
        print(f"     S1 Addr: {s[4]!r}")
        print(f"     Tr Addr: {s[5]!r}")

if __name__ == "__main__":
    main()
