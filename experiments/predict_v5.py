#!/usr/bin/env python3
"""
V5 Fast Test Inference Pipeline (High Performance, Sparse CSR, Multi-Worker, Resumable)
Amazon ML Challenge 2026 - Business Entity Resolution

Architecture:
- Candidate Retrieval: CountrySparseBlockingIndexV5 (Word & Char-level TF-IDF sparse matrix + exact/spatial hashes)
- Features: 44-feature vector from V4 pipeline (verified 100% compatible)
- Model: output_v4/best_model_v4.pkl
- Threshold: tau = 0.86
- Capping: ~35 candidates per S1
- Multi-processing: Configurable workers (default: 3 workers for 4-CPU Camber machine)
- Memory Safety: Clamped BLAS threads (OMP_NUM_THREADS=1), streamed batch writes, country-by-country cleanup
- Outputs:
    output_v5/matching_results_v5.tsv
    output_v5/candidate_pairs_v5.tsv
"""

import os
import sys

# Ensure workspace root is in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Clamp BLAS/OpenMP threads before loading numpy / scipy / lightgbm
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import time
import math
import pickle
import argparse
import collections
from multiprocessing import get_context
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v5 import CountrySparseBlockingIndexV5
from experiments.v4_pipeline import extract_features_v4, FEATURE_NAMES_V4

# Global worker state for multi-process pool
_WORKER_INDEX: Optional[CountrySparseBlockingIndexV5] = None
_WORKER_MODEL = None
_WORKER_TRAIN_NAME_COUNTS = None
_WORKER_TRAIN_ADDR_COUNTS = None
_WORKER_TOTAL_TR_RECORDS = 1.0

def _init_worker(
    index: CountrySparseBlockingIndexV5,
    model_path: str,
    train_name_counts: collections.Counter,
    train_addr_counts: collections.Counter,
    total_tr_records: float
):
    global _WORKER_INDEX, _WORKER_MODEL, _WORKER_TRAIN_NAME_COUNTS, _WORKER_TRAIN_ADDR_COUNTS, _WORKER_TOTAL_TR_RECORDS
    _WORKER_INDEX = index
    with open(model_path, "rb") as f:
        _WORKER_MODEL = pickle.load(f)["model"]
    _WORKER_TRAIN_NAME_COUNTS = train_name_counts
    _WORKER_TRAIN_ADDR_COUNTS = train_addr_counts
    _WORKER_TOTAL_TR_RECORDS = total_tr_records

def _process_s1_batch_worker(
    batch_s1: List[Tuple[int, str, str, str]],
    ctry: str,
    fixed_threshold: float,
    top_k: int
) -> List[Tuple[int, str, str, str]]:
    """
    Worker task: takes a batch of S1 queries, retrieves candidates via CSR index,
    extracts 44 features, runs model inference, and returns (row_idx, s1_id, match_str, cand_str).
    """
    global _WORKER_INDEX, _WORKER_MODEL, _WORKER_TRAIN_NAME_COUNTS, _WORKER_TRAIN_ADDR_COUNTS, _WORKER_TOTAL_TR_RECORDS

    prepped_queries = []
    query_meta = []

    for row_idx, s1_id, name, addr in batch_s1:
        nn, sn, tn = normalize_business_name(name)
        na, ta, pin, nums = normalize_address(addr)
        s1_prep = (nn, sn, tn, na, ta, pin, nums, ctry, name, addr)
        prepped_queries.append(s1_prep)

        n_freq = _WORKER_TRAIN_NAME_COUNTS.get(nn, 1) / _WORKER_TOTAL_TR_RECORDS
        a_freq = _WORKER_TRAIN_ADDR_COUNTS.get(na, 1) / _WORKER_TOTAL_TR_RECORDS
        n_rar = float(-math.log10(max(1e-5, n_freq)))
        a_rar = float(-math.log10(max(1e-5, a_freq)))

        query_meta.append((row_idx, s1_id, s1_prep, n_rar, a_rar))

    # Fast batch candidate retrieval via SciPy sparse dot product + hashes
    batch_candidates = _WORKER_INDEX.retrieve_batch_candidates(prepped_queries, top_k=top_k)

    # Prepare feature matrix for candidate pairs
    batch_pairs_feats = []
    batch_pair_meta = []  # (meta_idx, tid)

    for meta_idx, (row_idx, s1_id, s1_prep, n_rar, a_rar) in enumerate(query_meta):
        cands = batch_candidates[meta_idx]
        for tid in cands:
            tr_prep = _WORKER_INDEX.target_prepped.get(tid)
            if tr_prep is None:
                continue
            feats = extract_features_v4(s1_prep, tid, tr_prep, country=ctry, name_rarity=n_rar, addr_rarity=a_rar)
            batch_pairs_feats.append(feats)
            batch_pair_meta.append((meta_idx, tid))

    matches_by_meta = collections.defaultdict(list)
    if batch_pairs_feats:
        X_batch = np.array(batch_pairs_feats, dtype=np.float32)
        probs = _WORKER_MODEL.predict_proba(X_batch)[:, 1]
        for (meta_idx, tid), prob in zip(batch_pair_meta, probs):
            if prob >= fixed_threshold:
                matches_by_meta[meta_idx].append(tid)

    # Format results
    batch_results = []
    for meta_idx, (row_idx, s1_id, _, _, _) in enumerate(query_meta):
        raw_m = matches_by_meta.get(meta_idx, [])
        seen_m = set()
        dedup_m = []
        for m in raw_m:
            if m not in seen_m:
                seen_m.add(m)
                dedup_m.append(m)

        cands = batch_candidates[meta_idx]
        match_str = ",".join(dedup_m)
        cand_str = ",".join(cands)
        batch_results.append((row_idx, s1_id, match_str, cand_str))

    return batch_results

def parse_args():
    parser = argparse.ArgumentParser(description="V5 Fast Entity Resolution Inference")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--train-s1-path", type=str, default="dataset/train/train_source1.tsv", help="Path to train source1 for rarity")
    parser.add_argument("--model-path", type=str, default="output_v4/best_model_v4.pkl", help="Path to trained model")
    parser.add_argument("--output-matching", type=str, default="output_v5/matching_results_v5.tsv", help="Final matching TSV")
    parser.add_argument("--output-candidates", type=str, default="output_v5/candidate_pairs_v5.tsv", help="Final candidates TSV")
    parser.add_argument("--temp-dir", type=str, default="output_v5/temp_infer", help="Temporary country cache directory")
    parser.add_argument("--fixed-threshold", type=float, default=0.86, help="Decision threshold tau")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2500, help="S1 batch size for inference")
    parser.add_argument("--top-k", type=int, default=35, help="Top-K candidates per S1")
    return parser.parse_args()

def run_v5_test_inference():
    args = parse_args()

    print("=" * 75)
    print("V5 HIGH-PERFORMANCE TEST INFERENCE PIPELINE (Amazon ML Challenge 2026)")
    print(f"Model: {args.model_path}")
    print(f"Decision Threshold: tau = {args.fixed_threshold:.2f}")
    print(f"Workers: {args.num_workers} | Batch Size: {args.batch_size} | Top-K: {args.top_k}")
    print(f"Outputs: {args.output_matching} & {args.output_candidates}")
    print("=" * 75)
    t_start = time.time()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Verify model compatibility
    with open(args.model_path, "rb") as f:
        m_obj = pickle.load(f)
    model = m_obj["model"]
    feature_names = m_obj.get("feature_names", FEATURE_NAMES_V4)
    print(f"Loaded {type(model).__name__} with {len(feature_names)} features.")
    if len(feature_names) != 44:
        raise ValueError(f"Expected 44 features for V4 model, found {len(feature_names)}")

    # 1. Precompute Train-Only Corpus Frequencies for Rarity Features
    print("\n[Step 1] Loading Train Corpus Frequencies (Train Only)...")
    train_name_counts = collections.Counter()
    train_addr_counts = collections.Counter()
    total_tr_records = 0

    if os.path.exists(args.train_s1_path):
        with open(args.train_s1_path, "r", encoding="utf-8") as f:
            f.readline()
            for idx, line in enumerate(f):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > 2:
                    nn, _, _ = normalize_business_name(parts[1])
                    na, _, _, _ = normalize_address(parts[2])
                    if nn: train_name_counts[nn] += 1
                    if na: train_addr_counts[na] += 1
                    total_tr_records += 1
                if idx >= 150000: break

    total_tr_records = float(max(1, total_tr_records))
    print(f"  Precomputed corpus frequencies on {int(total_tr_records):,} train records.")

    # 2. Scan Test Source 1 Records in Exact Source Order
    print("\n[Step 2] Scanning Test Source 1 Records...")
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")

    ordered_s1_ids = []
    country_s1_groups = collections.defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
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
    print(f"  Total Source 1 Queries: {total_s1:,}")
    for c, q in sorted(country_s1_groups.items()):
        print(f"    - {c}: {len(q):,} queries")

    # Temp files directory
    os.makedirs(args.temp_dir, exist_ok=True)
    sorted_countries = sorted(country_s1_groups.keys(), key=lambda c: len(country_s1_groups[c]))

    total_matches_predicted = 0
    total_singletons_predicted = 0
    total_candidates_generated = 0

    # 3. Country-by-Country Processing
    for ctry_idx, ctry in enumerate(sorted_countries, 1):
        s1_list = country_s1_groups[ctry]
        expected_count = len(s1_list)
        temp_match_file = os.path.join(args.temp_dir, f"match_{ctry}.tsv")
        temp_cand_file = os.path.join(args.temp_dir, f"cand_{ctry}.tsv")

        # Check if already completed
        if os.path.exists(temp_match_file) and os.path.exists(temp_cand_file):
            with open(temp_match_file) as f_chk:
                m_cnt = sum(1 for _ in f_chk)
            with open(temp_cand_file) as f_chk:
                c_cnt = sum(1 for _ in f_chk)
            if m_cnt == expected_count and c_cnt == expected_count:
                print(f"\n[{ctry_idx}/{len(sorted_countries)}] Country '{ctry}' already 100% completed ({expected_count:,} records). Skipping.")
                continue
            else:
                print(f"\n[{ctry_idx}/{len(sorted_countries)}] Country '{ctry}' incomplete ({m_cnt}/{expected_count}). Re-processing.")

        print(f"\n[{ctry_idx}/{len(sorted_countries)}] Processing Country '{ctry}' ({expected_count:,} entities)...")
        t_ctry_start = time.time()

        # Load target records for this country
        print(f"  Loading target records for {ctry} from Source 2 & Source 3...")
        t0 = time.time()
        targets_list = []
        for fn in [s2_path, s3_path]:
            if not os.path.exists(fn):
                continue
            with open(fn, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) > 3 and parts[3].strip() == ctry:
                        eid = parts[0].strip()
                        name = parts[1].strip() if len(parts) > 1 else ""
                        addr = parts[2].strip() if len(parts) > 2 else ""
                        targets_list.append((eid, name, addr))

        print(f"  Loaded {len(targets_list):,} targets in {time.time()-t0:.2f}s.")

        # Build V5 Sparse Blocking Index
        print(f"  Building V5 Sparse Blocking Index for {ctry}...")
        t0 = time.time()
        cbi_v5 = CountrySparseBlockingIndexV5(country=ctry)
        cbi_v5.build_index(targets_list)
        print(f"  V5 Index built in {time.time()-t0:.2f}s (Word Vocab: {cbi_v5.T_word.shape[1]:,}, Char Vocab: {cbi_v5.T_char.shape[1]:,}).")

        # Partition queries into batches
        batches = [
            s1_list[i : i + args.batch_size]
            for i in range(0, expected_count, args.batch_size)
        ]
        total_batches = len(batches)

        ctry_matches = 0
        ctry_singletons = 0
        ctry_cands = 0

        # Execute inference across worker processes
        with open(temp_match_file, "w", encoding="utf-8") as f_tm, \
             open(temp_cand_file, "w", encoding="utf-8") as f_tc:

            if args.num_workers > 1:
                ctx = get_context("fork" if sys.platform != "win32" else "spawn")
                init_args = (cbi_v5, args.model_path, train_name_counts, train_addr_counts, total_tr_records)
                with ctx.Pool(processes=args.num_workers, initializer=_init_worker, initargs=init_args) as pool:
                    task_args = [(b, ctry, args.fixed_threshold, args.top_k) for b in batches]
                    for b_idx, batch_res in enumerate(pool.starmap(_process_s1_batch_worker, task_args)):
                        for row_idx, s1_id, match_str, cand_str in batch_res:
                            f_tm.write(f"{row_idx}\t{s1_id}\t{match_str}\n")
                            f_tc.write(f"{row_idx}\t{s1_id}\t{cand_str}\n")
                            if match_str:
                                ctry_matches += len(match_str.split(","))
                            else:
                                ctry_singletons += 1
                            if cand_str:
                                ctry_cands += len(cand_str.split(","))

                        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                            elapsed_b = time.time() - t_ctry_start
                            q_done = min(expected_count, (b_idx + 1) * args.batch_size)
                            q_rate = q_done / max(1.0, elapsed_b)
                            print(f"    Batch {b_idx+1}/{total_batches} ({q_done:,}/{expected_count:,} queries, {q_rate:.1f} q/s) | Matches: {ctry_matches:,}")
            else:
                # Single-worker mode
                _init_worker(cbi_v5, args.model_path, train_name_counts, train_addr_counts, total_tr_records)
                for b_idx, b in enumerate(batches):
                    batch_res = _process_s1_batch_worker(b, ctry, args.fixed_threshold, args.top_k)
                    for row_idx, s1_id, match_str, cand_str in batch_res:
                        f_tm.write(f"{row_idx}\t{s1_id}\t{match_str}\n")
                        f_tc.write(f"{row_idx}\t{s1_id}\t{cand_str}\n")
                        if match_str:
                            ctry_matches += len(match_str.split(","))
                        else:
                            ctry_singletons += 1
                        if cand_str:
                            ctry_cands += len(cand_str.split(","))

                    if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                        elapsed_b = time.time() - t_ctry_start
                        q_done = min(expected_count, (b_idx + 1) * args.batch_size)
                        q_rate = q_done / max(1.0, elapsed_b)
                        print(f"    Batch {b_idx+1}/{total_batches} ({q_done:,}/{expected_count:,} queries, {q_rate:.1f} q/s) | Matches: {ctry_matches:,}")

        total_matches_predicted += ctry_matches
        total_singletons_predicted += ctry_singletons
        total_candidates_generated += ctry_cands
        print(f"  {ctry} Completed in {time.time()-t_ctry_start:.1f}s: matches={ctry_matches:,}, singletons={ctry_singletons:,}")

        # Clean memory before next country
        del cbi_v5
        del targets_list
        gc.collect()

    # 4. Stream Merge All Country Temp Files in Exact Source Order
    print("\n[Step 4] Merging Country Predictions into Exact Source Order...")
    t0 = time.time()

    ordered_matches = [None] * total_s1
    ordered_cands = [None] * total_s1

    final_tot_matches = 0
    final_tot_singletons = 0
    final_tot_cands = 0

    for ctry in sorted_countries:
        temp_match_file = os.path.join(args.temp_dir, f"match_{ctry}.tsv")
        temp_cand_file = os.path.join(args.temp_dir, f"cand_{ctry}.tsv")

        with open(temp_match_file, "r", encoding="utf-8") as f_tm:
            for line in f_tm:
                parts = line.rstrip("\r\n").split("\t")
                row_idx = int(parts[0])
                m_str = parts[2] if len(parts) > 2 else ""
                ordered_matches[row_idx] = m_str
                if m_str:
                    final_tot_matches += len(m_str.split(","))
                else:
                    final_tot_singletons += 1

        with open(temp_cand_file, "r", encoding="utf-8") as f_tc:
            for line in f_tc:
                parts = line.rstrip("\r\n").split("\t")
                row_idx = int(parts[0])
                c_str = parts[2] if len(parts) > 2 else ""
                ordered_cands[row_idx] = c_str
                if c_str:
                    final_tot_cands += len(c_str.split(","))

    # Write Final Files
    os.makedirs(os.path.dirname(args.output_matching), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_candidates), exist_ok=True)

    with open(args.output_matching, "w", encoding="utf-8") as f_match, \
         open(args.output_candidates, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, s1_id in enumerate(ordered_s1_ids):
            m_val = ordered_matches[idx] if ordered_matches[idx] is not None else ""
            c_val = ordered_cands[idx] if ordered_cands[idx] is not None else ""
            f_match.write(f"{s1_id}\t{m_val}\n")
            f_cand.write(f"{s1_id}\t{c_val}\n")

    print(f"  Final submission files written in {time.time()-t0:.2f}s.")
    print("\n" + "=" * 75)
    print("V5 TEST INFERENCE COMPLETED SUCCESSFULLY")
    print(f"  Total S1 Entities:           {total_s1:,}")
    print(f"  Total Singletons Predicted:  {final_tot_singletons:,} ({(final_tot_singletons/total_s1)*100:.2f}%)")
    print(f"  Total Non-Empty Predictions: {total_s1 - final_tot_singletons:,} ({((total_s1 - final_tot_singletons)/total_s1)*100:.2f}%)")
    print(f"  Total Matches Predicted:     {final_tot_matches:,}")
    print(f"  Average Matches per S1:      {final_tot_matches / total_s1:.4f}")
    print(f"  Total Candidates Generated:  {final_tot_cands:,}")
    print(f"  Average Candidates per S1:   {final_tot_cands / total_s1:.2f}")
    print(f"  Total Wall Time:             {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} minutes)")
    print("=" * 75)

if __name__ == "__main__":
    run_v5_test_inference()
