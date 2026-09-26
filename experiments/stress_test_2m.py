#!/usr/bin/env python3
"""
Final 2M Target-Pool Stress Test
Amazon ML Challenge 2026 - Business Entity Resolution

Evaluates thresholds tau in {0.92, 0.93, 0.94} against a 2,000,000 target-record pool.
Uses the exact same 5,000 S1 validation queries and official per-S1 macro F0.5 metric.
"""

import os
import sys
import time
import json
import random
import pickle
import collections
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.phase7_improved_features import extract_pair_features_v2

def compute_strict_entity_f05(gt_ids: set, pred_ids: set):
    len_gt = len(gt_ids)
    len_pred = len(pred_ids)

    if len_gt == 0:
        if len_pred == 0:
            return 0, 0, 0, 1.0, 1.0, 1.0
        else:
            return 0, len_pred, 0, 0.0, 0.0, 0.0

    tp = len(gt_ids & pred_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)

    if len_pred == 0 or tp == 0:
        return tp, fp, fn, 0.0, 0.0, 0.0

    p = tp / len_pred
    r = tp / len_gt
    denom = 0.25 * p + r
    f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
    return tp, fp, fn, p, r, f05

def main():
    print("=" * 70)
    print("FINAL 2M TARGET-POOL STRESS TEST (tau in {0.92, 0.93, 0.94})")
    print("=" * 70)
    random.seed(42)
    np.random.seed(42)

    # 1. Load exact same 5,000 validation queries
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
    val_s1_ids = s1_all[:5000]
    gt_val = {s: gt_all[s] for s in val_s1_ids}

    needed_s1 = set(val_s1_ids)
    s1_records = {}
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

    val_targets = set()
    for mids in gt_val.values():
        val_targets |= mids

    # 2. Build 2M Target Pool
    print("\n[1] Loading 2,000,000 target records...")
    t0 = time.time()
    target_records = {}
    bg_target = 2000000
    bg_added = 0
    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0].strip()
                if eid in val_targets:
                    target_records[eid] = p
                elif bg_added < bg_target:
                    target_records[eid] = p
                    bg_added += 1

    print(f"  Loaded {len(target_records):,} targets ({len(val_targets):,} true + {bg_added:,} distractors) in {time.time()-t0:.2f}s.")

    # 3. Index into CountryBlockingIndexV2
    print("\n[2] Indexing 2M targets into CountryBlockingIndexV2 (max_posting_size=350)...")
    t0 = time.time()
    cbi = CountryBlockingIndexV2(max_posting_size=350)
    for eid, p in target_records.items():
        name = p[1] if len(p) > 1 else ""
        addr = p[2] if len(p) > 2 else ""
        ctry = p[3] if len(p) > 3 else "US"
        cbi.add_target_entity(eid, name, addr, ctry)
    print(f"  Indexing completed in {time.time()-t0:.2f}s.")

    # 4. Load trained V2 model
    with open("output_v2/best_model_v2.pkl", "rb") as f:
        m_obj = pickle.load(f)
    model = m_obj["model"]

    # 5. Candidate retrieval and feature extraction
    print("\n[3] Candidate retrieval and feature extraction for 5,000 validation entities...")
    t0 = time.time()
    candidate_scores = collections.defaultdict(dict)

    for idx, sid in enumerate(val_s1_ids):
        prep = s1_prepped[sid]
        ctry = prep[7]
        cands = cbi.retrieve_candidates(prep[:7], ctry, max_candidates=30)
        if cands:
            feats = []
            for tid in cands:
                tr_prep = cbi.target_data[tid][:7]
                f = extract_pair_features_v2(prep[:7], tid, tr_prep, country=ctry)
                feats.append(f)
            probs = model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
            for tid, prob in zip(cands, probs):
                candidate_scores[sid][tid] = float(prob)

    print(f"  Scoring completed in {time.time()-t0:.2f}s.")

    # 6. Evaluate thresholds: 0.92, 0.93, 0.94
    print("\n[4] Evaluating Thresholds on 2M Target Pool:")
    print("-" * 70)
    print(f"{'Threshold':<10} | {'Macro F0.5':<10} | {'Sing Acc':<10} | {'FP':<6} | {'FN':<6} | {'Avg Matches':<12}")
    print("-" * 70)

    n_sing = sum(len(m) == 0 for m in gt_val.values())
    results = []

    for th in [0.92, 0.93, 0.94]:
        scores = []
        tot_fp = 0
        tot_fn = 0
        tot_pred = 0
        sing_cor = 0

        for sid in val_s1_ids:
            gt_ids = gt_val[sid]
            c_scores = candidate_scores[sid]
            p_ids = {tid for tid, prob in c_scores.items() if prob >= th}

            tp, fp, fn, p, r, f05 = compute_strict_entity_f05(gt_ids, p_ids)
            scores.append(f05)
            tot_fp += fp
            tot_fn += fn
            tot_pred += len(p_ids)

            if len(gt_ids) == 0 and len(p_ids) == 0:
                sing_cor += 1

        macro_f = float(np.mean(scores))
        sing_acc = (sing_cor / n_sing) * 100.0
        avg_pred = tot_pred / len(val_s1_ids)

        row = {
            "threshold": th,
            "macro_f05": macro_f,
            "singleton_acc": sing_acc,
            "false_positives": tot_fp,
            "false_negatives": tot_fn,
            "avg_pred_matches": avg_pred
        }
        results.append(row)
        print(f"{th:<10.2f} | {macro_f:<10.4f} | {sing_acc:<9.2f}% | {tot_fp:<6,} | {tot_fn:<6,} | {avg_pred:<12.2f}")

    print("-" * 70)

    # Save results
    with open("output/stress_test_2m.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
