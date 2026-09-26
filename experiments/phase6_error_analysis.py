#!/usr/bin/env python3
"""
Phase 6: Error Analysis and Quantification
Quantifies the exact contribution of each error category on validation data.
"""

import os
import sys
import collections
import pickle
import re
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

def classify_error(s1_name, s1_addr, tr_name, tr_addr):
    """Classifies pair into primary error category."""
    n1_norm, n1_stem, n1_toks = normalize_business_name(s1_name)
    n2_norm, n2_stem, n2_toks = normalize_business_name(tr_name)
    a1_norm, a1_toks, pin1, num1 = normalize_address(s1_addr)
    a2_norm, a2_toks, pin2, num2 = normalize_address(tr_addr)

    if not tr_addr or len(a2_norm) == 0:
        return "missing_target_address"
    if n1_stem == n2_stem:
        if num1 and num2 and not (num1 & num2):
            return "house_number_difference"
        if pin1 and pin2 and pin1 != pin2:
            return "postal_pin_difference"
        return "address_variation_or_abbrev"
    
    # Check if transliteration issue (non-ascii characters in raw text)
    if any(ord(c) > 127 for c in s1_name) or any(ord(c) > 127 for c in tr_name):
        return "transliteration_or_script"
        
    # Check word order
    if set(n1_toks) == set(n2_toks) and n1_toks != n2_toks:
        return "word_order_transposition"
        
    # Check if DBA / alias (completely different tokens)
    common_toks = set(n1_toks) & set(n2_toks)
    if not common_toks:
        return "dba_or_different_brand_name"
        
    # Spelling variation / typo
    return "spelling_variation_or_typo"

def main():
    print("Running Phase 6: Error Analysis...")
    with open("output/best_model.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    optimal_thresh = saved.get("optimal_threshold", 0.90)

    # Load validation entities
    gt_map = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            gt_map[parts[0].strip()] = set(parts[1].strip().split(",")) if len(parts) > 1 and parts[1].strip() else set()
            if idx >= 60000: break

    val_s1 = {}
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            if eid in gt_map and len(val_s1) < 3000:
                val_s1[eid] = parts
            if len(val_s1) >= 3000: break

    needed_targets = set()
    for eid in val_s1:
        needed_targets |= gt_map[eid]

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
                elif bg_added < 50000:
                    targets[eid] = parts
                    bg_added += 1

    cbi = CountryBlockingIndex(max_posting_size=250)
    for tid, parts in targets.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        cbi.add_target_entity(tid, name, addr, ctry)

    # Classify errors
    error_counts = collections.Counter()
    fn_examples = collections.defaultdict(list)
    fp_examples = []
    singleton_errors = 0

    for s1_id, s1_row in val_s1.items():
        true_mids = gt_map[s1_id]
        ctry = s1_row[3].strip() if len(s1_row) > 3 else ""
        n_norm, n_stem, n_toks = normalize_business_name(s1_row[1])
        a_norm, a_toks, pin, nums = normalize_address(s1_row[2])
        s1_prep = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)

        cands = cbi.retrieve_candidates(s1_prep, ctry, max_candidates=35)
        pred_mids = set()

        if cands:
            pairs_feats = []
            for tid in cands:
                tr_prep = cbi.target_data[tid][:7]
                pairs_feats.append(extract_pair_features(s1_prep, tid, tr_prep))
            probs = model.predict_proba(np.array(pairs_feats, dtype=np.float32))[:, 1]
            for tid, p in zip(cands, probs):
                if p >= optimal_thresh:
                    pred_mids.add(tid)

        # Check singleton error
        if len(true_mids) == 0 and len(pred_mids) > 0:
            singleton_errors += 1

        # Check false negatives
        missed = true_mids - pred_mids
        for m in missed:
            tr_row = targets.get(m, [m, "", "", ""])
            cat = classify_error(s1_row[1], s1_row[2], tr_row[1], tr_row[2])
            error_counts[cat] += 1
            if len(fn_examples[cat]) < 3:
                fn_examples[cat].append({
                    "s1_id": s1_id, "s1_name": s1_row[1], "s1_addr": s1_row[2],
                    "tr_id": m, "tr_name": tr_row[1], "tr_addr": tr_row[2]
                })

    report_path = "experiments/phase6_error_analysis.md"
    total_fns = sum(error_counts.values())
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6: Error Analysis and Quantification\n\n")
        f.write(f"- Total False Negatives Analyzed: {total_fns:,}\n")
        f.write(f"- Singleton False Positive Merges: {singleton_errors}\n\n")
        f.write("## 1. False Negative Breakdown by Category\n\n")
        f.write("| Error Category | Count | Percentage | Primary Root Cause |\n")
        f.write("| --- | --- | --- | --- |\n")
        for cat, cnt in error_counts.most_common():
            pct = (cnt / total_fns) * 100 if total_fns > 0 else 0
            f.write(f"| **{cat}** | {cnt:,} | {pct:.1f}% | ")
            if cat == "missing_target_address":
                f.write("Target address empty; model penalizes address features even if name matches\n")
            elif cat == "spelling_variation_or_typo":
                f.write("Name typo or minor edit distance lowered prob below 0.90\n")
            elif cat == "address_variation_or_abbrev":
                f.write("Address formatting difference without house number overlap\n")
            elif cat == "dba_or_different_brand_name":
                f.write("Completely different trade name / DBA; requires strong address link\n")
            elif cat == "transliteration_or_script":
                f.write("Non-standard Indic or accented spelling divergence\n")
            elif cat == "house_number_difference":
                f.write("Different suite / apartment or slight municipal number mismatch\n")
            else:
                f.write("Other\n")

        f.write("\n## 2. Representative Examples by Error Category\n\n")
        for cat, examples in fn_examples.items():
            f.write(f"### {cat}\n")
            for ex in examples:
                f.write(f"- **S1:** `{ex['s1_id']}` | `{ex['s1_name']}` | `{ex['s1_addr']}`\n")
                f.write(f"  **TR:** `{ex['tr_id']}` | `{ex['tr_name']}` | `{ex['tr_addr']}`\n\n")

    print(f"Error analysis completed and saved to {report_path}")

if __name__ == "__main__":
    main()
