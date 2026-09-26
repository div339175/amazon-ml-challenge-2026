#!/usr/bin/env python3
"""
Test Prediction Pipeline V2
Amazon ML Challenge 2026 - Business Entity Resolution

Generates:
- output_v2/matching_results_v2.tsv
- output_v2/candidate_pairs_v2.tsv

Using:
- CountryBlockingIndexV2 (multi-pass blocking, consonant skeleton, compound spatial keys, posting limit 350)
- 36 Enhanced Features (Levenshtein, Jaro-Winkler, consonant skeleton match, token containment, n-gram cosines, missing-address interactions)
- LightGBM V2 Model (output_v2/best_model_v2.pkl)
- Selected Fixed Decision Threshold: tau = 0.84

Guarantees:
- Exactly 1,732,544 rows in exact test_source1.tsv order
- Matching IDs are strict subset of Candidate IDs
- Valid TAB separation with comma-separated IDs
- Zero self-matches, only valid test S2/S3 IDs
- Leaves baseline_v1/ and output/ untouched
"""

import os
import sys
import gc
import time
import pickle
import collections
from typing import Dict, List, Tuple, Set
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.phase7_improved_features import extract_pair_features_v2, FEATURE_NAMES_V2

def run_prediction_v2(
    test_dir: str = "dataset/test",
    model_path: str = "output_v2/best_model_v2.pkl",
    output_matching: str = "output_v2/matching_results_v2.tsv",
    output_candidates: str = "output_v2/candidate_pairs_v2.tsv",
    fixed_threshold: float = 0.84,
    max_cands: int = 30,
    batch_size: int = 10000
):
    print("=" * 70)
    print("V2 TEST INFERENCE PIPELINE (Amazon ML Challenge 2026)")
    print(f"Decision Threshold: Fixed tau = {fixed_threshold:.2f}")
    print(f"Target Outputs: {output_matching} & {output_candidates}")
    print("=" * 70)
    t_start = time.time()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading trained LightGBM V2 model from {model_path}...")
    with open(model_path, "rb") as f:
        m_obj = pickle.load(f)
    model = m_obj["model"]
    feature_names = m_obj.get("feature_names", FEATURE_NAMES_V2)
    print(f"Loaded {type(model).__name__} with {len(feature_names)} features.")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    # 1. Discover all S1 entities, their exact row order, and group by country
    print(f"\n[STEP 1] Scanning {s1_path} to index entities and maintain strict input order...")
    t0 = time.time()
    ordered_s1_ids = []
    # country -> list of (row_idx, s1_id, name, addr)
    country_s1_groups = collections.defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for row_idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            addr = parts[2].strip() if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else "US"

            ordered_s1_ids.append(s1_id)
            country_s1_groups[ctry].append((row_idx, s1_id, name, addr))

    total_s1 = len(ordered_s1_ids)
    print(f"  Scanned {total_s1:,} Source 1 entities in {time.time()-t0:.2f}s:")
    for ctry, grp in sorted(country_s1_groups.items(), key=lambda x: len(x[1])):
        print(f"    - Country '{ctry}': {len(grp):,} entities ({(len(grp)/total_s1)*100:.1f}%)")

    # Pre-allocate output arrays matching ordered_s1_ids
    # Using arrays of strings indexed by row_idx: minimal RAM overhead (~500 MB total)
    results_matches = ["" for _ in range(total_s1)]
    results_candidates = ["" for _ in range(total_s1)]

    # Process countries in ascending size: France (259k) -> US (663k) -> India (810k)
    sorted_countries = sorted(country_s1_groups.keys(), key=lambda c: len(country_s1_groups[c]))

    total_matches_predicted = 0
    total_singletons_predicted = 0
    total_candidates_generated = 0
    total_s2_matches = 0
    total_s3_matches = 0

    # 2. Process Country by Country
    for ctry_idx, ctry in enumerate(sorted_countries, 1):
        s1_list = country_s1_groups[ctry]
        print(f"\n==================================================")
        print(f"[{ctry_idx}/{len(sorted_countries)}] Processing Country: '{ctry}' ({len(s1_list):,} S1 queries)")
        print(f"==================================================")
        t_ctry_start = time.time()

        # Build Blocking Index V2 for this country
        cbi = CountryBlockingIndexV2(max_posting_size=350)
        target_count = 0

        for fn in [s2_path, s3_path]:
            t_load = time.time()
            base_fn = os.path.basename(fn)
            fn_cnt = 0
            with open(fn, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) > 3 and parts[3].strip() == ctry:
                        eid = parts[0].strip()
                        name = parts[1].strip() if len(parts) > 1 else ""
                        addr = parts[2].strip() if len(parts) > 2 else ""
                        cbi.add_target_entity(eid, name, addr, ctry)
                        fn_cnt += 1
            target_count += fn_cnt
            print(f"  Loaded {fn_cnt:,} records from {base_fn} in {time.time()-t_load:.2f}s.")

        print(f"  Indexed {target_count:,} target records for country '{ctry}'.")
        print(f"  Running candidate retrieval and inference...")

        # Process S1 queries in batches
        total_batches = (len(s1_list) + batch_size - 1) // batch_size
        ctry_matches = 0
        ctry_singletons = 0
        ctry_cands = 0
        t_infer_start = time.time()

        for b_idx in range(total_batches):
            batch_s1 = s1_list[b_idx * batch_size : (b_idx + 1) * batch_size]

            batch_pairs_feats = []
            # list of (row_idx, s1_id, tr_id)
            batch_pair_meta = []
            batch_cands_by_s1 = {}

            for row_idx, s1_id, name, addr in batch_s1:
                n_norm, n_stem, n_toks = normalize_business_name(name)
                a_norm, a_toks, pin, nums = normalize_address(addr)
                s1_prep = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)

                cands = cbi.retrieve_candidates(s1_prep, ctry, max_candidates=max_cands)
                results_candidates[row_idx] = ",".join(cands)
                batch_cands_by_s1[row_idx] = []
                ctry_cands += len(cands)

                for tid in cands:
                    tr_prep = cbi.target_data[tid][:7]
                    feats = extract_pair_features_v2(s1_prep, tid, tr_prep, country=ctry)
                    batch_pairs_feats.append(feats)
                    batch_pair_meta.append((row_idx, s1_id, tid))

            # Run batch inference if there are candidate pairs
            if batch_pairs_feats:
                X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)[:, 1]

                for (row_idx, s1_id, tid), prob in zip(batch_pair_meta, probs):
                    if prob >= fixed_threshold:
                        batch_cands_by_s1[row_idx].append(tid)

            # Assign matches to pre-allocated results
            for row_idx, _, _, _ in batch_s1:
                matched_ids = batch_cands_by_s1[row_idx]
                # Deduplicate while preserving rank order
                seen_m = set()
                dedup_m = []
                for m in matched_ids:
                    if m not in seen_m:
                        seen_m.add(m)
                        dedup_m.append(m)
                        if m.startswith("S2-"):
                            total_s2_matches += 1
                        elif m.startswith("S3-"):
                            total_s3_matches += 1

                results_matches[row_idx] = ",".join(dedup_m)
                if dedup_m:
                    ctry_matches += len(dedup_m)
                else:
                    ctry_singletons += 1

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                done_cnt = min((b_idx + 1) * batch_size, len(s1_list))
                pct = (done_cnt / len(s1_list)) * 100.0
                rate = done_cnt / max(1e-3, (time.time() - t_infer_start))
                print(f"    Batch {b_idx+1}/{total_batches}: {done_cnt:,}/{len(s1_list):,} S1 queries ({pct:.1f}%) at {rate:.0f} queries/s...")

        total_matches_predicted += ctry_matches
        total_singletons_predicted += ctry_singletons
        total_candidates_generated += ctry_cands

        t_ctry_elapsed = time.time() - t_ctry_start
        print(f"  Country '{ctry}' completed in {t_ctry_elapsed:.1f}s ({t_ctry_elapsed/60:.1f} min).")
        print(f"    Matches: {ctry_matches:,} | Singletons: {ctry_singletons:,} ({(ctry_singletons/len(s1_list))*100:.2f}%) | Candidates: {ctry_cands:,}")

        # Explicit cleanup of country index to free RAM immediately
        del cbi
        gc.collect()

    # 3. Write Output Files
    print("\n==================================================")
    print("Writing Final Output Files (output_v2/)")
    print("==================================================")
    os.makedirs(os.path.dirname(output_matching), exist_ok=True)
    os.makedirs(os.path.dirname(output_candidates), exist_ok=True)

    t_write = time.time()
    with open(output_matching, "w", encoding="utf-8") as f_match, \
         open(output_candidates, "w", encoding="utf-8") as f_cand:

        # Header rows
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, s1_id in enumerate(ordered_s1_ids):
            m_str = results_matches[idx]
            c_str = results_candidates[idx]
            f_match.write(f"{s1_id}\t{m_str}\n")
            f_cand.write(f"{s1_id}\t{c_str}\n")

    print(f"Files written successfully in {time.time()-t_write:.2f}s.")

    # 4. Immediate Self-Verification
    print("\n==================================================")
    print("Verifying Integrity & Challenge Submission Rules")
    print("==================================================")
    match_file_size = os.path.getsize(output_matching)
    cand_file_size = os.path.getsize(output_candidates)

    print(f"  - Matching file:   {output_matching} ({match_file_size / (1024*1024):.2f} MB)")
    print(f"  - Candidates file: {output_candidates} ({cand_file_size / (1024*1024):.2f} MB)")
    print(f"  - Total S1 Rows:   {total_s1:,} (Target: 1,732,544)")
    print(f"  - Total Candidates Generated: {total_candidates_generated:,} (Avg {total_candidates_generated/total_s1:.2f} per S1)")
    print(f"  - Total Matches Predicted:    {total_matches_predicted:,} (Avg {total_matches_predicted/total_s1:.2f} per S1)")
    print(f"  - Predicted Singletons:       {total_singletons_predicted:,} ({(total_singletons_predicted/total_s1)*100:.2f}%)")
    print(f"  - Non-Empty Matches:          {total_s1 - total_singletons_predicted:,} ({((total_s1 - total_singletons_predicted)/total_s1)*100:.2f}%)")
    print(f"  - Source 2 Matches:           {total_s2_matches:,}")
    print(f"  - Source 3 Matches:           {total_s3_matches:,}")

    # Integrity assertions
    assert total_s1 == 1732544, f"Row count mismatch: expected 1,732,544, got {total_s1}"
    print("  ✓ Row count assertion passed (exactly 1,732,544 S1 rows).")

    total_time = time.time() - t_start
    print(f"\nEnd-to-end inference completed in {total_time:.1f}s ({total_time/60:.2f} minutes).")

    return {
        "total_s1": total_s1,
        "total_matches": total_matches_predicted,
        "total_singletons": total_singletons_predicted,
        "non_empty_matches": total_s1 - total_singletons_predicted,
        "total_candidates": total_candidates_generated,
        "s2_matches": total_s2_matches,
        "s3_matches": total_s3_matches,
        "matching_file_size_mb": match_file_size / (1024*1024),
        "candidates_file_size_mb": cand_file_size / (1024*1024),
        "total_seconds": total_time
    }

if __name__ == "__main__":
    run_prediction_v2()
