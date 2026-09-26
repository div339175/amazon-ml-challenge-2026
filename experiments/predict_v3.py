#!/usr/bin/env python3
"""
Test Prediction Pipeline V3
Amazon ML Challenge 2026 - Business Entity Resolution

Inference configuration:
- Model: output_v2/best_model_v2.pkl (36 features)
- Blocking: CountryBlockingIndexV2 (max_posting_size=350)
- Decision Threshold: Fixed tau = 0.92 (robust entity-level macro F0.5)
- Target Outputs:
    output_v3/matching_results_v3.tsv
    output_v3/candidate_pairs_v3.tsv

Safety:
- Leaves baseline_v1/, output/, and output_v2/ 100% untouched.
"""

import os
import sys
import gc
import time
import pickle
import collections
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.phase7_improved_features import extract_pair_features_v2, FEATURE_NAMES_V2

def run_prediction_v3(
    test_dir: str = "dataset/test",
    model_path: str = "output_v2/best_model_v2.pkl",
    output_matching: str = "output_v3/matching_results_v3.tsv",
    output_candidates: str = "output_v3/candidate_pairs_v3.tsv",
    fixed_threshold: float = 0.93,
    max_cands: int = 30,
    batch_size: int = 10000
):
    print("=" * 70)
    print("V3 TEST INFERENCE PIPELINE (Amazon ML Challenge 2026)")
    print(f"Decision Threshold: Fixed tau = {fixed_threshold:.2f}")
    print(f"Target Outputs: {output_matching} & {output_candidates}")
    print("=" * 70)
    t_start = time.time()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        m_obj = pickle.load(f)
    model = m_obj["model"]
    feature_names = m_obj.get("feature_names", FEATURE_NAMES_V2)
    print(f"Loaded {type(model).__name__} with {len(feature_names)} features.")

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    ordered_s1_ids = []
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
    print(f"Scanned {total_s1:,} Source 1 test entities.")

    results_matches = ["" for _ in range(total_s1)]
    results_candidates = ["" for _ in range(total_s1)]

    sorted_countries = sorted(country_s1_groups.keys(), key=lambda c: len(country_s1_groups[c]))

    total_matches_predicted = 0
    total_singletons_predicted = 0
    total_candidates_generated = 0

    for ctry_idx, ctry in enumerate(sorted_countries, 1):
        s1_list = country_s1_groups[ctry]
        print(f"\n[{ctry_idx}/{len(sorted_countries)}] Processing '{ctry}' ({len(s1_list):,} entities)...")
        t_ctry_start = time.time()

        cbi = CountryBlockingIndexV2(max_posting_size=350)
        target_count = 0

        for fn in [s2_path, s3_path]:
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

        print(f"  Indexed {target_count:,} target records.")

        total_batches = (len(s1_list) + batch_size - 1) // batch_size
        ctry_matches = 0
        ctry_singletons = 0
        ctry_cands = 0

        for b_idx in range(total_batches):
            batch_s1 = s1_list[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch_pairs_feats = []
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

            if batch_pairs_feats:
                X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)[:, 1]

                for (row_idx, s1_id, tid), prob in zip(batch_pair_meta, probs):
                    if prob >= fixed_threshold:
                        batch_cands_by_s1[row_idx].append(tid)

            for row_idx, _, _, _ in batch_s1:
                matched_ids = batch_cands_by_s1[row_idx]
                seen_m = set()
                dedup_m = []
                for m in matched_ids:
                    if m not in seen_m:
                        seen_m.add(m)
                        dedup_m.append(m)

                results_matches[row_idx] = ",".join(dedup_m)
                if dedup_m:
                    ctry_matches += len(dedup_m)
                else:
                    ctry_singletons += 1

        total_matches_predicted += ctry_matches
        total_singletons_predicted += ctry_singletons
        total_candidates_generated += ctry_cands
        print(f"  Done in {time.time()-t_ctry_start:.1f}s: matches={ctry_matches:,}, singletons={ctry_singletons:,}")

        del cbi
        gc.collect()

    os.makedirs(os.path.dirname(output_matching), exist_ok=True)
    os.makedirs(os.path.dirname(output_candidates), exist_ok=True)

    with open(output_matching, "w", encoding="utf-8") as f_match, \
         open(output_candidates, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, s1_id in enumerate(ordered_s1_ids):
            f_match.write(f"{s1_id}\t{results_matches[idx]}\n")
            f_cand.write(f"{s1_id}\t{results_candidates[idx]}\n")

    print("\nV3 Prediction Files Written Successfully.")
    return {
        "total_s1": total_s1,
        "total_matches": total_matches_predicted,
        "total_singletons": total_singletons_predicted,
        "total_candidates": total_candidates_generated,
        "elapsed_seconds": time.time() - t_start
    }

if __name__ == "__main__":
    run_prediction_v3()
