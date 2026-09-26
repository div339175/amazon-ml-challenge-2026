#!/usr/bin/env python3
"""
Phase 4: Comprehensive Blocking Audit & Failure Mode Analysis
Audits blocking recall across:
- US vs India
- S1 -> S2 vs S1 -> S3
- Analyzes why true matches were missed:
  1. Never generated in broad blocking (missing keys)
  2. Generated in broad blocking but pushed out by max_candidates cap
  3. Generated as candidate but rejected by the model threshold
- Categorizes exact failure cases
"""

import os
import sys
import collections
import pickle
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from business_entity_resolution.src.blocking import (
    CountryBlockingIndex,
    extract_blocking_keys
)
from business_entity_resolution.src.features import extract_pair_features
from business_entity_resolution.src.evaluate import evaluate_predictions

def main():
    print("Running Phase 4: Blocking Audit...")
    with open("output/best_model.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    optimal_thresh = saved.get("optimal_threshold", 0.90)

    # Load 5,000 S1 validation entities (stratified US and India)
    val_s1_us = []
    val_s1_in = []
    
    gt_map = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            gt_map[parts[0].strip()] = set(parts[1].strip().split(",")) if len(parts) > 1 and parts[1].strip() else set()
            if idx >= 150000: break

    s1_data = {}
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            if eid in gt_map:
                ctry = parts[3].strip() if len(parts) > 3 else ""
                if ctry == "US" and len(val_s1_us) < 2500:
                    val_s1_us.append(eid)
                    s1_data[eid] = parts
                elif ctry == "India" and len(val_s1_in) < 2500:
                    val_s1_in.append(eid)
                    s1_data[eid] = parts
            if len(val_s1_us) >= 2500 and len(val_s1_in) >= 2500:
                break

    val_s1_all = val_s1_us + val_s1_in
    val_s1_set = set(val_s1_all)
    val_gt = {eid: gt_map[eid] for eid in val_s1_all}

    # True match targets needed
    needed_targets = set()
    for mids in val_gt.values():
        needed_targets |= mids

    # Load target records (true matches + 100k distractors per country to test realistic density)
    print(f"Loading target records for {len(needed_targets):,} true targets + background distractors...")
    targets = {}
    us_distractors = 0
    in_distractors = 0
    MAX_DISTRACTORS = 100000

    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn) as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0].strip()
                ctry = parts[3].strip() if len(parts) > 3 else ""
                if eid in needed_targets:
                    targets[eid] = parts
                elif ctry == "US" and us_distractors < MAX_DISTRACTORS:
                    targets[eid] = parts
                    us_distractors += 1
                elif ctry == "India" and in_distractors < MAX_DISTRACTORS:
                    targets[eid] = parts
                    in_distractors += 1

    print(f"Total target pool indexed: {len(targets):,} records.")

    # Build blocking index with recording of raw un-truncated posting lists
    cbi = CountryBlockingIndex(max_posting_size=250)
    for tid, parts in targets.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        cbi.add_target_entity(tid, name, addr, ctry)

    print("Auditing candidates on validation S1 entities...")
    audit_results = {
        "overall": {"total_gt": 0, "broad_hit": 0, "top25_hit": 0, "top35_hit": 0, "model_hit": 0},
        "US": {"total_gt": 0, "broad_hit": 0, "top25_hit": 0, "top35_hit": 0, "model_hit": 0},
        "India": {"total_gt": 0, "broad_hit": 0, "top25_hit": 0, "top35_hit": 0, "model_hit": 0},
        "S1_S2": {"total_gt": 0, "broad_hit": 0, "top25_hit": 0, "top35_hit": 0, "model_hit": 0},
        "S1_S3": {"total_gt": 0, "broad_hit": 0, "top25_hit": 0, "top35_hit": 0, "model_hit": 0},
    }

    cand_counts = []
    zero_cands_cnt = 0
    missed_in_broad_examples = []
    pushed_out_examples = []
    rejected_by_model_examples = []
    false_positive_examples = []

    for s1_id in val_s1_all:
        s1_row = s1_data[s1_id]
        ctry = s1_row[3].strip() if len(s1_row) > 3 else ""
        true_mids = val_gt[s1_id]

        n_norm, n_stem, n_toks = normalize_business_name(s1_row[1])
        a_norm, a_toks, pin, nums = normalize_address(s1_row[2])
        s1_prep = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)

        # 1. Retrieve all broad matches (no cap)
        bkeys = extract_blocking_keys(n_stem, n_toks, a_norm, a_toks, pin, nums)
        c_index = cbi.indices.get(ctry, {})
        broad_cands = set()
        scores = collections.defaultdict(float)
        exact_hits = set()

        for k, weight in bkeys:
            pl = c_index.get(k, [])
            discount = 1.0 / (1.0 + 0.05 * len(pl))
            for tid in pl:
                broad_cands.add(tid)
                scores[tid] += weight * discount
                if k.startswith("nx:"): exact_hits.add(tid)

        for tid in exact_hits:
            scores[tid] += 20.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top25 = {t for t, _ in ranked[:25]}
        top35 = {t for t, _ in ranked[:35]}

        cand_counts.append(len(top35))
        if len(top35) == 0:
            zero_cands_cnt += 1

        # Model predictions
        pred_mids = set()
        cands_to_score = list(top35)
        if cands_to_score:
            pairs_feats = []
            for tid in cands_to_score:
                tr_prep = cbi.target_data[tid][:7]
                pairs_feats.append(extract_pair_features(s1_prep, tid, tr_prep))
            probs = model.predict_proba(np.array(pairs_feats, dtype=np.float32))[:, 1]
            for tid, p in zip(cands_to_score, probs):
                if p >= optimal_thresh:
                    pred_mids.add(tid)
                elif tid in true_mids and len(rejected_by_model_examples) < 10:
                    rejected_by_model_examples.append({
                        "s1_id": s1_id, "s1_name": s1_row[1], "s1_addr": s1_row[2],
                        "tr_id": tid, "tr_name": targets[tid][1], "tr_addr": targets[tid][2],
                        "prob": float(p)
                    })

        for p_id in pred_mids:
            if p_id not in true_mids and len(false_positive_examples) < 10:
                false_positive_examples.append({
                    "s1_id": s1_id, "s1_name": s1_row[1], "s1_addr": s1_row[2],
                    "tr_id": p_id, "tr_name": targets[p_id][1], "tr_addr": targets[p_id][2],
                })

        # Track stats
        for mid in true_mids:
            src_cat = "S1_S2" if mid.startswith("S2-") else "S1_S3"
            for bucket in ["overall", ctry, src_cat]:
                audit_results[bucket]["total_gt"] += 1
                if mid in broad_cands:
                    audit_results[bucket]["broad_hit"] += 1
                if mid in top25:
                    audit_results[bucket]["top25_hit"] += 1
                if mid in top35:
                    audit_results[bucket]["top35_hit"] += 1
                if mid in pred_mids:
                    audit_results[bucket]["model_hit"] += 1

            if mid not in broad_cands and len(missed_in_broad_examples) < 10:
                missed_in_broad_examples.append({
                    "s1_id": s1_id, "s1_name": s1_row[1], "s1_addr": s1_row[2], "s1_ctry": ctry,
                    "tr_id": mid, "tr_name": targets[mid][1], "tr_addr": targets[mid][2],
                })
            elif mid in broad_cands and mid not in top35 and len(pushed_out_examples) < 10:
                pushed_out_examples.append({
                    "s1_id": s1_id, "s1_name": s1_row[1], "s1_addr": s1_row[2],
                    "tr_id": mid, "tr_name": targets[mid][1], "tr_addr": targets[mid][2],
                    "rank": [t for t, _ in ranked].index(mid), "score": scores[mid]
                })

    # Save Markdown Audit Report
    report_path = "experiments/phase4_blocking_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4: Blocking Audit & Recall Bottleneck Report\n\n")
        f.write("## 1. Recall Breakdown by Group\n\n")
        f.write("| Group | Ground Truth Matches | Broad Recall | Top-25 Recall | Top-35 Recall | Model Recall (thresh=0.90) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for grp, res in audit_results.items():
            tot = res["total_gt"]
            b_r = (res["broad_hit"] / tot) * 100 if tot else 0
            t25_r = (res["top25_hit"] / tot) * 100 if tot else 0
            t35_r = (res["top35_hit"] / tot) * 100 if tot else 0
            m_r = (res["model_hit"] / tot) * 100 if tot else 0
            f.write(f"| **{grp}** | {tot:,} | {b_r:.2f}% | {t25_r:.2f}% | {t35_r:.2f}% | {m_r:.2f}% |\n")

        f.write(f"\n- **Average candidates per S1 (Top-35):** {np.mean(cand_counts):.2f}\n")
        f.write(f"- **Median candidates per S1:** {float(np.median(cand_counts)):.1f}\n")
        f.write(f"- **Percentage of S1 with 0 candidates:** {(zero_cands_cnt / len(val_s1_all))*100:.2f}%\n\n")

        f.write("## 2. Qualitative Error Examples\n\n")
        f.write("### A. Missed in Broad Blocking (Never Generated as Candidate)\n\n")
        for ex in missed_in_broad_examples[:5]:
            f.write(f"- **S1:** `{ex['s1_id']}` | Name: `{ex['s1_name']}` | Addr: `{ex['s1_addr']}`\n")
            f.write(f"  **TR:** `{ex['tr_id']}` | Name: `{ex['tr_name']}` | Addr: `{ex['tr_addr']}`\n\n")

        f.write("### B. Pushed Out by Top-K Rank Cutoff\n\n")
        for ex in pushed_out_examples[:5]:
            f.write(f"- **S1:** `{ex['s1_id']}` | Name: `{ex['s1_name']}` | Addr: `{ex['s1_addr']}`\n")
            f.write(f"  **TR:** `{ex['tr_id']}` (Rank {ex['rank']}) | Name: `{ex['tr_name']}` | Addr: `{ex['tr_addr']}` | Score: {ex['score']:.2f}\n\n")

        f.write("### C. Generated as Candidate but Rejected by Model (Threshold 0.90)\n\n")
        for ex in rejected_by_model_examples[:5]:
            f.write(f"- **S1:** `{ex['s1_id']}` | Name: `{ex['s1_name']}` | Addr: `{ex['s1_addr']}`\n")
            f.write(f"  **TR:** `{ex['tr_id']}` (Prob {ex['prob']:.4f}) | Name: `{ex['tr_name']}` | Addr: `{ex['tr_addr']}`\n\n")

    print(f"\nBlocking audit report saved to {report_path}")

if __name__ == "__main__":
    main()
