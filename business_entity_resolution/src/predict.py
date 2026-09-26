#!/usr/bin/env python3
"""
Phase 8, 9 & 12: Entity-Level Prediction & Final Test Inference
Runs end-to-end inference over the complete test set:
- test_source1.tsv (1.73M entities)
- test_source2.tsv (4.89M entities)
- test_source3.tsv (5.08M entities)

Processes country-by-country (US, India, France, and any open-set country) to maintain
optimal memory efficiency (< 3GB RAM) and maximum throughput.
Generates:
  output/matching_results.tsv
  output/candidate_pairs.tsv
guaranteeing:
- Exactly 1 row per test Source 1 entity in exact source order
- Valid TAB separation with comma-separated ID lists
- Matched IDs are guaranteed to be a subset of candidate IDs
- No duplicate IDs, no self-matches, only valid test S2/S3 IDs
"""

import os
import sys
import time
import pickle
import collections
import numpy as np
from typing import Dict, List, Set, Tuple

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from business_entity_resolution.src.blocking import (
    CountryBlockingIndex
)
from business_entity_resolution.src.features import (
    extract_pair_features,
    FEATURE_NAMES
)

def run_test_inference(
    test_dir: str = "dataset/test",
    model_path: str = "output/best_model.pkl",
    output_matching: str = "output/matching_results.tsv",
    output_candidates: str = "output/candidate_pairs.tsv",
    max_cands: int = 30
):
    print("==================================================")
    print("PHASES 8, 9 & 12: FINAL TEST INFERENCE PIPELINE")
    print("==================================================")
    t_start = time.time()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")
        
    print(f"Loading trained model from {model_path}...")
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)
    model = model_obj["model"]
    optimal_threshold = model_obj.get("optimal_threshold", 0.85)
    print(f"Loaded {type(model).__name__} with optimal threshold = {optimal_threshold:.2f}")
    
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    # 1. Discover all S1 entities, their order, and group by country
    print(f"\nScanning {s1_path} to discover countries and entities...")
    ordered_s1_ids = []
    country_s1_records = collections.defaultdict(list)
    
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            s1_id = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            addr = parts[2].strip() if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else ""
            
            ordered_s1_ids.append(s1_id)
            country_s1_records[ctry].append((s1_id, name, addr))
            
    total_s1 = len(ordered_s1_ids)
    print(f"Total Source 1 test entities: {total_s1:,}")
    for ctry, recs in country_s1_records.items():
        print(f"  - Country '{ctry}': {len(recs):,} entities")
        
    # In-memory storage for final predictions
    # dict: s1_id -> list of matched_ids, dict: s1_id -> list of cand_ids
    results_matches: Dict[str, List[str]] = {}
    results_candidates: Dict[str, List[str]] = {}
    
    # 2. Process each country independently
    for ctry, s1_list in country_s1_records.items():
        print(f"\n==================================================")
        print(f"Processing Country: '{ctry}' ({len(s1_list):,} S1 entities)")
        print(f"==================================================")
        t_ctry_start = time.time()
        
        # Build blocking index for this country from S2 and S3
        cbi = CountryBlockingIndex(max_posting_size=250)
        target_count = 0
        
        for fn in [s2_path, s3_path]:
            print(f"Loading {os.path.basename(fn)} records for country '{ctry}'...")
            t0 = time.time()
            with open(fn, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) > 3 and parts[3].strip() == ctry:
                        eid = parts[0].strip()
                        name = parts[1].strip() if len(parts) > 1 else ""
                        addr = parts[2].strip() if len(parts) > 2 else ""
                        cbi.add_target_entity(eid, name, addr, ctry)
                        target_count += 1
            print(f"  Done in {time.time() - t0:.2f}s.")
            
        print(f"Indexed {target_count:,} target records for country '{ctry}'.")
        
        # Now run candidate generation, feature extraction, and prediction for this country's S1 entities
        print(f"Running candidate retrieval and inference for {len(s1_list):,} entities...")
        t0 = time.time()
        
        BATCH_SIZE = 10000
        total_batches = (len(s1_list) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for b_idx in range(total_batches):
            batch_s1 = s1_list[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            
            # Retrieve candidates and extract features for batch
            batch_pairs_feats = []
            batch_pair_ids = []  # list of (s1_id, tr_id)
            
            for s1_id, name, addr in batch_s1:
                n_norm, n_stem, n_toks = normalize_business_name(name)
                a_norm, a_toks, pin, nums = normalize_address(addr)
                s1_prep = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)
                
                cands = cbi.retrieve_candidates(s1_prep, ctry, max_candidates=max_cands)
                results_candidates[s1_id] = cands
                results_matches[s1_id] = []  # default empty (singleton)
                
                for tr_id in cands:
                    tr_prep = cbi.target_data[tr_id][:7]
                    feats = extract_pair_features(s1_prep, tr_id, tr_prep)
                    batch_pairs_feats.append(feats)
                    batch_pair_ids.append((s1_id, tr_id))
                    
            if batch_pairs_feats:
                X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                probs = model.predict_proba(X_batch)[:, 1]
                
                # Assign matches
                for (s1_id, tr_id), prob in zip(batch_pair_ids, probs):
                    if prob >= optimal_threshold:
                        results_matches[s1_id].append(tr_id)
                        
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                processed_cnt = min((b_idx + 1) * BATCH_SIZE, len(s1_list))
                print(f"  Processed {processed_cnt:,}/{len(s1_list):,} entities ({(processed_cnt/len(s1_list))*100:.1f}%) in {time.time() - t0:.1f}s...")
                
        print(f"Country '{ctry}' completed in {time.time() - t_ctry_start:.2f}s.")
        
        # Explicit memory cleanup
        del cbi
        
    # 3. Write final submission files strictly in original S1 order
    print("\n==================================================")
    print("Writing Final Submission Files")
    print("==================================================")
    os.makedirs(os.path.dirname(output_matching), exist_ok=True)
    os.makedirs(os.path.dirname(output_candidates), exist_ok=True)
    
    total_matches_written = 0
    total_singletons_written = 0
    total_cands_written = 0
    s2_matches = 0
    s3_matches = 0
    
    with open(output_matching, "w", encoding="utf-8") as f_match, \
         open(output_candidates, "w", encoding="utf-8") as f_cand:
         
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in ordered_s1_ids:
            # Candidates
            cands = results_candidates.get(s1_id, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            total_cands_written += len(cands)
            
            # Matches
            matches = results_matches.get(s1_id, [])
            # Deduplicate while preserving order
            seen_m = set()
            dedup_matches = []
            for m in matches:
                if m not in seen_m:
                    seen_m.add(m)
                    dedup_matches.append(m)
                    if m.startswith("S2-"): s2_matches += 1
                    elif m.startswith("S3-"): s3_matches += 1
                    
            match_str = ",".join(dedup_matches)
            f_match.write(f"{s1_id}\t{match_str}\n")
            
            if not dedup_matches:
                total_singletons_written += 1
            else:
                total_matches_written += len(dedup_matches)
                
    t_elapsed = time.time() - t_start
    print(f"\nInference completed in {t_elapsed:.2f} seconds ({t_elapsed/60:.2f} minutes).")
    print(f"Summary Statistics:")
    print(f"  - Total Source 1 Rows Written: {len(ordered_s1_ids):,}")
    print(f"  - Total Candidates Generated: {total_cands_written:,} (Avg {total_cands_written/total_s1:.2f} per S1)")
    print(f"  - Total Matches Predicted: {total_matches_written:,} (Avg {total_matches_written/total_s1:.2f} per S1)")
    print(f"  - Predicted Singletons (No Match): {total_singletons_written:,} ({(total_singletons_written/total_s1)*100:.2f}%)")
    print(f"  - Source 2 Matches: {s2_matches:,}")
    print(f"  - Source 3 Matches: {s3_matches:,}")
    print(f"  - Output matching file: {output_matching}")
    print(f"  - Output candidate file: {output_candidates}")
    
    return {
        "total_s1": len(ordered_s1_ids),
        "total_matches": total_matches_written,
        "total_singletons": total_singletons_written,
        "total_candidates": total_cands_written,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
        "elapsed_seconds": t_elapsed
    }

if __name__ == "__main__":
    run_test_inference()
