#!/usr/bin/env python3
"""
Compare candidate recall between Baseline Blocking (v1) and Multi-Pass Blocking (v2)
on a held-out validation set with realistic background distractors.
"""

import os
import sys
import collections
import numpy as np

from business_entity_resolution.src.preprocessing import normalize_business_name, normalize_address
from business_entity_resolution.src.blocking import CountryBlockingIndex
from experiments.blocking_v2 import CountryBlockingIndexV2

def main():
    print("Comparing Blocking v1 vs Blocking v2...")
    # Load 4000 validation S1 entities (2000 US, 2000 India)
    gt_map = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            gt_map[parts[0].strip()] = set(parts[1].strip().split(",")) if len(parts) > 1 and parts[1].strip() else set()
            if idx >= 80000: break

    val_s1 = {}
    val_us = 0
    val_in = 0
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            ctry = parts[3].strip() if len(parts) > 3 else ""
            if eid in gt_map:
                if ctry == "US" and val_us < 2000:
                    val_s1[eid] = parts; val_us += 1
                elif ctry == "India" and val_in < 2000:
                    val_s1[eid] = parts; val_in += 1
            if val_us >= 2000 and val_in >= 2000: break

    needed_targets = set()
    for eid in val_s1:
        needed_targets |= gt_map[eid]

    # Load targets (true targets + 100k distractors)
    targets = {}
    bg_added = 0
    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn) as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0].strip()
                if eid in needed_targets:
                    targets[eid] = parts
                elif bg_added < 100000:
                    targets[eid] = parts
                    bg_added += 1

    total_gt = sum(len(gt_map[s]) for s in val_s1)
    print(f"Validation: {len(val_s1)} entities, {total_gt:,} true matches to retrieve, {len(targets):,} indexed targets.")

    # Index v1
    cbi_v1 = CountryBlockingIndex(max_posting_size=250)
    # Index v2
    cbi_v2 = CountryBlockingIndexV2(max_posting_size=350)

    for tid, parts in targets.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        cbi_v1.add_target_entity(tid, name, addr, ctry)
        cbi_v2.add_target_entity(tid, name, addr, ctry)

    # Evaluate v1 vs v2
    v1_hits = 0
    v2_hits = 0
    v1_cands = []
    v2_cands = []

    v1_hits_in = 0; in_total = 0
    v2_hits_in = 0

    for s1_id, parts in val_s1.items():
        true_mids = gt_map[s1_id]
        ctry = parts[3].strip() if len(parts) > 3 else ""
        n_norm, n_stem, n_toks = normalize_business_name(parts[1])
        a_norm, a_toks, pin, nums = normalize_address(parts[2])
        s1_prep = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)

        c1 = cbi_v1.retrieve_candidates(s1_prep, ctry, max_candidates=25)
        c2 = cbi_v2.retrieve_candidates(s1_prep, ctry, max_candidates=30)

        v1_hits += len(true_mids & set(c1))
        v2_hits += len(true_mids & set(c2))
        v1_cands.append(len(c1))
        v2_cands.append(len(c2))

        if ctry == "India":
            in_total += len(true_mids)
            v1_hits_in += len(true_mids & set(c1))
            v2_hits_in += len(true_mids & set(c2))

    print("\n--- BLOCKING RECALL COMPARISON ---")
    print(f"Overall Recall:")
    print(f"  v1 Baseline:  {v1_hits}/{total_gt} = {(v1_hits/total_gt)*100:.2f}% (Avg {np.mean(v1_cands):.1f} cands/S1)")
    print(f"  v2 Multi-Pass: {v2_hits}/{total_gt} = {(v2_hits/total_gt)*100:.2f}% (Avg {np.mean(v2_cands):.1f} cands/S1)")
    print(f"India Recall:")
    print(f"  v1 Baseline:  {v1_hits_in}/{in_total} = {(v1_hits_in/in_total)*100:.2f}%")
    print(f"  v2 Multi-Pass: {v2_hits_in}/{in_total} = {(v2_hits_in/in_total)*100:.2f}%")

if __name__ == "__main__":
    main()
