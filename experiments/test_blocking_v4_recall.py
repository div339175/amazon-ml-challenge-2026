#!/usr/bin/env python3
"""
Benchmark Candidate Recall for CountryBlockingIndexV4
Tests on 3,000 S1 validation queries against 1,500,000+ target records.
Compares V2 vs V4.
"""

import os
import sys
import time
import json
import random
import collections
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v4 import CountryBlockingIndexV4

def main():
    print("=" * 70)
    print("BENCHMARKING V4 HIGH-RECALL MULTI-PASS BLOCKING (1.5M TARGET POOL)")
    print("=" * 70)
    random.seed(42)
    np.random.seed(42)

    # 1. Load exact same 3,000 validation queries
    gt_all = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
            gt_all[p[0].strip()] = mids
            if idx >= 150000:
                break

    s1_all = list(gt_all.keys())
    random.seed(42)
    random.shuffle(s1_all)
    val_s1_ids = s1_all[:3000]
    gt_val = {s: gt_all[s] for s in val_s1_ids}

    needed_true_targets = set()
    for mids in gt_val.values():
        needed_true_targets |= mids

    # 2. Load S1 source records
    s1_records = {}
    needed_s1 = set(val_s1_ids)
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0].strip()
            if eid in needed_s1:
                s1_records[eid] = p
                if len(s1_records) == len(needed_s1):
                    break

    s1_prepped = {}
    for sid in val_s1_ids:
        p = s1_records[sid]
        n_norm, n_stem, n_toks = normalize_business_name(p[1])
        a_norm, a_toks, pin, nums = normalize_address(p[2])
        s1_prepped[sid] = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums, p[3].strip(), p[1], p[2])

    # 3. Load Target Records: true targets + 1,500,000 distractors
    print("\n[1] Loading target records (true targets + 1.5M distractors)...")
    target_records = {}
    bg_target_count = 1500000
    bg_added = 0
    t0 = time.time()

    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0].strip()
                if eid in needed_true_targets:
                    target_records[eid] = p
                elif bg_added < bg_target_count:
                    target_records[eid] = p
                    bg_added += 1

    print(f"  Loaded {len(target_records):,} target records in {time.time()-t0:.2f}s.")

    # 4. Build CountryBlockingIndexV4
    print("\n[2] Indexing target records into CountryBlockingIndexV4...")
    t0 = time.time()
    cbi = CountryBlockingIndexV4()
    for eid, p in target_records.items():
        name = p[1] if len(p) > 1 else ""
        addr = p[2] if len(p) > 2 else ""
        ctry = p[3] if len(p) > 3 else "US"
        cbi.add_target_entity(eid, name, addr, ctry)
    print(f"  Indexing completed in {time.time()-t0:.2f}s.")

    # 5. Retrieve candidates across max_cands settings: 30, 40, 50
    for max_k in [30, 40, 50]:
        t0 = time.time()
        hits = 0
        total_gt = 0
        cands_tot = 0
        by_country = collections.defaultdict(lambda: {"hits": 0, "total": 0})

        for sid in val_s1_ids:
            prep = s1_prepped[sid]
            ctry = prep[7]
            true_mids = gt_val[sid]
            cands = set(cbi.retrieve_candidates(prep[:7], ctry, max_candidates=max_k))
            cands_tot += len(cands)

            for tid in true_mids:
                if tid in target_records:
                    total_gt += 1
                    by_country[ctry]["total"] += 1
                    if tid in cands:
                        hits += 1
                        by_country[ctry]["hits"] += 1

        rec = (hits / total_gt) * 100.0 if total_gt > 0 else 0
        avg_c = cands_tot / len(val_s1_ids)
        print(f"\nV4 Candidate Recall (max_candidates={max_k}): {rec:.2f}% ({hits:,}/{total_gt:,} true matches) | Avg Candidates/S1: {avg_c:.1f} in {time.time()-t0:.2f}s")
        for ctry, st in sorted(by_country.items()):
            c_rec = (st["hits"] / st["total"]) * 100.0 if st["total"] > 0 else 0
            print(f"  - {ctry}: {c_rec:.2f}% ({st['hits']:,}/{st['total']:,})")

if __name__ == "__main__":
    main()
