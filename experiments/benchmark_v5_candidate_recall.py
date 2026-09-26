#!/usr/bin/env python3
"""
Benchmark Candidate Recall and Retrieval Throughput for V5 Sparse Blocking Index
Amazon ML Challenge 2026 - Business Entity Resolution

Features:
- Word-level & Char-level TF-IDF via SciPy CSR sparse matrix dot product
- Exact hash tables (Exact Name, Exact Addr, PIN+HouseNum)
- Configurable worker count (default: 3 workers for 4-CPU Camber machine)
- Strict thread clamping (OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, etc.)
- Benchmarks candidate recall against train ground truth with large distractor pool
- Measures throughput (q/s), candidate counts, P95, and memory usage
"""

import os
import sys

# Ensure workspace root is in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Clamp BLAS/OpenMP threads before importing numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import math
import random
import argparse
import resource
import collections
from multiprocessing import get_context
from typing import Dict, List, Tuple, Set

import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v5 import CountrySparseBlockingIndexV5

_WORKER_INDEX = None

def _init_worker(index: CountrySparseBlockingIndexV5):
    global _WORKER_INDEX
    _WORKER_INDEX = index

def _process_query_batch(batch_tuples: List[Tuple], top_k: int) -> List[List[str]]:
    global _WORKER_INDEX
    return _WORKER_INDEX.retrieve_batch_candidates(batch_tuples, top_k=top_k)

def parse_args():
    parser = argparse.ArgumentParser(description="V5 Candidate Recall & Latency Benchmark")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of S1 validation queries (default: 1000)")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker processes (default: 3)")
    parser.add_argument("--country", type=str, default="all", choices=["all", "US", "India"], help="Country to benchmark")
    parser.add_argument("--target-pool-size", type=int, default=1500000, help="Distractor target pool size (default: 1.5M)")
    parser.add_argument("--top-k", type=int, default=35, help="Top-K candidates per S1 (default: 35)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for sparse matrix queries (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    else:
        return usage / 1024.0

def load_validation_data(num_queries: int, target_pool_size: int, country_filter: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)

    print("\n[Step 1] Loading Ground Truth...")
    gt_all = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            if not p or not p[0].strip():
                continue
            s1_id = p[0].strip()
            mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
            gt_all[s1_id] = mids
            if idx >= 200000:
                break

    # Load S1 source records
    print("[Step 2] Scanning Train Source 1 Records...")
    s1_metadata = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if not p or not p[0].strip():
                continue
            sid = p[0].strip()
            if sid in gt_all:
                name = p[1].strip() if len(p) > 1 else ""
                addr = p[2].strip() if len(p) > 2 else ""
                ctry = p[3].strip() if len(p) > 3 else "US"
                s1_metadata[sid] = (name, addr, ctry)

    # Filter candidate queries by country
    eligible_s1 = [
        sid for sid in s1_metadata
        if country_filter == "all" or s1_metadata[sid][2] == country_filter
    ]
    random.shuffle(eligible_s1)
    val_s1_ids = eligible_s1[:num_queries]

    gt_val = {s: gt_all[s] for s in val_s1_ids}
    needed_true_targets = set()
    for mids in gt_val.values():
        needed_true_targets |= mids

    # Preprocess S1 queries
    s1_prepped_by_country = collections.defaultdict(list)
    for sid in val_s1_ids:
        name, addr, ctry = s1_metadata[sid]
        nn, sn, tn = normalize_business_name(name)
        na, ta, pin, nums = normalize_address(addr)
        prep = (nn, sn, tn, na, ta, pin, nums, ctry, name, addr, sid)
        s1_prepped_by_country[ctry].append(prep)

    print(f"  Selected {len(val_s1_ids):,} validation queries across countries:")
    for ctry, q_list in sorted(s1_prepped_by_country.items()):
        print(f"    - {ctry}: {len(q_list):,} queries")

    # Load Target Records: needed true targets + distractors
    print(f"\n[Step 3] Loading Target Pool (True targets + up to {target_pool_size:,} distractors)...")
    t0 = time.time()
    target_by_country = collections.defaultdict(list)
    loaded_true_targets = set()
    distractor_count = 0

    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        if not os.path.exists(fn):
            continue
        with open(fn, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if not p or not p[0].strip():
                    continue
                tid = p[0].strip()
                name = p[1].strip() if len(p) > 1 else ""
                addr = p[2].strip() if len(p) > 2 else ""
                ctry = p[3].strip() if len(p) > 3 else "US"

                if country_filter != "all" and ctry != country_filter:
                    continue

                if tid in needed_true_targets:
                    target_by_country[ctry].append((tid, name, addr))
                    loaded_true_targets.add(tid)
                elif distractor_count < target_pool_size:
                    target_by_country[ctry].append((tid, name, addr))
                    distractor_count += 1

    tot_targets = sum(len(v) for v in target_by_country.values())
    print(f"  Loaded {tot_targets:,} target records in {time.time()-t0:.2f}s.")
    print(f"  True target coverage in pool: {len(loaded_true_targets):,}/{len(needed_true_targets):,}")

    return s1_prepped_by_country, target_by_country, gt_val

def run_benchmark():
    args = parse_args()
    print("=" * 75)
    print("V5 CANDIDATE RETRIEVAL BENCHMARK")
    print(f"Queries: {args.num_queries:,} | Top-K: {args.top_k} | Workers: {args.num_workers}")
    print(f"Target Pool Size: {args.target_pool_size:,} | Country Filter: {args.country}")
    print("=" * 75)

    s1_prepped_by_country, target_by_country, gt_val = load_validation_data(
        num_queries=args.num_queries,
        target_pool_size=args.target_pool_size,
        country_filter=args.country,
        seed=args.seed
    )

    overall_hits = 0
    overall_true_targets = 0
    overall_cands_count = 0
    all_cand_counts = []
    country_metrics = {}

    total_query_count = sum(len(q) for q in s1_prepped_by_country.values())
    t_bench_start = time.time()

    for ctry, queries in sorted(s1_prepped_by_country.items()):
        targets = target_by_country.get(ctry, [])
        if not queries or not targets:
            print(f"\n[Warning] No queries or targets for country {ctry}. Skipping.")
            continue

        print(f"\n[{ctry}] Indexing {len(targets):,} targets into CountrySparseBlockingIndexV5...")
        t_idx_start = time.time()
        cbi_v5 = CountrySparseBlockingIndexV5(country=ctry)
        cbi_v5.build_index(targets)
        t_idx = time.time() - t_idx_start
        print(f"[{ctry}] Index built in {t_idx:.2f}s | Word Vocab: {cbi_v5.T_word.shape[1]:,} | Char Vocab: {cbi_v5.T_char.shape[1]:,}")

        # Chunk queries into batches
        batch_size = args.batch_size
        query_batches = [
            queries[i : i + batch_size]
            for i in range(0, len(queries), batch_size)
        ]

        print(f"[{ctry}] Retrieving candidates for {len(queries):,} queries ({len(query_batches)} batches, workers={args.num_workers})...")
        t_ret_start = time.time()
        retrieved_cand_lists = []

        if args.num_workers > 1:
            # Multi-process execution
            ctx = get_context("fork" if sys.platform != "win32" else "spawn")
            with ctx.Pool(processes=args.num_workers, initializer=_init_worker, initargs=(cbi_v5,)) as pool:
                # Each task processes one batch
                batch_args = [(b, args.top_k) for b in query_batches]
                results = pool.starmap(_process_query_batch, batch_args)
                for res in results:
                    retrieved_cand_lists.extend(res)
        else:
            # Single-process execution
            for b in query_batches:
                res = cbi_v5.retrieve_batch_candidates(b, top_k=args.top_k)
                retrieved_cand_lists.extend(res)

        t_ret = time.time() - t_ret_start
        q_rate = len(queries) / max(0.001, t_ret)

        # Evaluate candidate recall
        c_hits = 0
        c_total_gt = 0
        c_cand_counts = []

        for q_tuple, cands in zip(queries, retrieved_cand_lists):
            sid = q_tuple[10]
            true_mids = gt_val.get(sid, set())
            cand_set = set(cands)
            c_cand_counts.append(len(cand_set))
            all_cand_counts.append(len(cand_set))

            for tid in true_mids:
                c_total_gt += 1
                if tid in cand_set:
                    c_hits += 1

        rec = (c_hits / c_total_gt * 100.0) if c_total_gt > 0 else 0.0
        avg_c = np.mean(c_cand_counts) if c_cand_counts else 0.0
        p95_c = np.percentile(c_cand_counts, 95) if c_cand_counts else 0.0

        overall_hits += c_hits
        overall_true_targets += c_total_gt
        overall_cands_count += sum(c_cand_counts)

        country_metrics[ctry] = {
            "queries": len(queries),
            "targets": len(targets),
            "indexing_sec": t_idx,
            "retrieval_sec": t_ret,
            "queries_per_sec": q_rate,
            "recall": rec,
            "hits": c_hits,
            "true_targets": c_total_gt,
            "avg_cands": avg_c,
            "p95_cands": p95_c,
        }

        print(f"[{ctry}] Recall: {rec:.2f}% ({c_hits:,}/{c_total_gt:,}) | Throughput: {q_rate:.1f} q/s | Avg Cands: {avg_c:.1f} (P95: {p95_c:.0f})")

    # Overall Summary
    total_time = time.time() - t_bench_start
    overall_recall = (overall_hits / overall_true_targets * 100.0) if overall_true_targets > 0 else 0.0
    overall_avg_cands = (overall_cands_count / total_query_count) if total_query_count > 0 else 0.0
    overall_p95_cands = np.percentile(all_cand_counts, 95) if all_cand_counts else 0.0
    peak_rss = get_peak_rss_mb()

    print("\n" + "=" * 75)
    print("V5 BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"Total Validation Queries:      {total_query_count:,}")
    print(f"Overall Candidate Recall:      {overall_recall:.2f}% ({overall_hits:,}/{overall_true_targets:,} true targets)")
    print(f"Average Candidates per Query:  {overall_avg_cands:.2f}")
    print(f"P95 Candidates per Query:      {overall_p95_cands:.1f}")
    print(f"Total Benchmark Wall Time:     {total_time:.2f}s")
    print(f"Peak RSS Memory:               {peak_rss:.1f} MB")
    print("-" * 75)
    print(f"{'Country':<10} | {'Queries':<8} | {'Recall':<10} | {'Throughput (q/s)':<18} | {'Avg Cands':<10}")
    print("-" * 75)
    for ctry, m in country_metrics.items():
        print(f"{ctry:<10} | {m['queries']:<8} | {m['recall']:>6.2f}%    | {m['queries_per_sec']:>14.1f}   | {m['avg_cands']:>8.1f}")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmark()
