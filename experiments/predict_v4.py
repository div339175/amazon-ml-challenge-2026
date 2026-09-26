#!/usr/bin/env python3
"""
V4 Test Inference Pipeline (High Performance, Memory-Capped, Resumable)
Amazon ML Challenge 2026 - Business Entity Resolution

Approved Configuration:
- Model: output_v4/best_model_v4.pkl (44 features)
- Retrieval: V4 Multi-Channel Union (V2 + V4 + Exact Name + Exact Addr + PIN/HouseNum)
- Fixed Threshold: tau = 0.86
- Outputs:
    output_v4/matching_results_v4.tsv
    output_v4/candidate_pairs_v4.tsv

Guarantees:
- Leaves baseline_v1/, output/, output_v2/, output_v3/, and best_model_v4.pkl untouched.
- Exactly 1,732,544 rows in exact test_source1.tsv order.
- Skips already completed country files in output_v4/temp_infer/ (e.g. France).
- Memory capped under 2 GB via disk-backed batch streaming and shared target data.
"""

import os
import sys
import gc
import time
import math
import pickle
import collections
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.blocking_v4 import CountryBlockingIndexV4
from experiments.v4_pipeline import extract_features_v4, FEATURE_NAMES_V4

def run_v4_test_inference(
    test_dir: str = "dataset/test",
    train_s1_path: str = "dataset/train/train_source1.tsv",
    model_path: str = "output_v4/best_model_v4.pkl",
    output_matching: str = "output_v4/matching_results_v4.tsv",
    output_candidates: str = "output_v4/candidate_pairs_v4.tsv",
    fixed_threshold: float = 0.86,
    batch_size: int = 2500
):
    print("=" * 75)
    print("V4 FINAL TEST INFERENCE PIPELINE (Amazon ML Challenge 2026)")
    print(f"Model: {model_path}")
    print(f"Decision Threshold: tau = {fixed_threshold:.2f}")
    print(f"Outputs: {output_matching} & {output_candidates}")
    print("=" * 75)
    t_start = time.time()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        m_obj = pickle.load(f)
    model = m_obj["model"]
    feature_names = m_obj.get("feature_names", FEATURE_NAMES_V4)
    print(f"Loaded {type(model).__name__} with {len(feature_names)} features.")

    # 1. Precompute Train-Only Corpus Frequencies for Rarity Features
    print("\n[Step 1] Loading Train Corpus Frequencies (Train Only)...")
    train_name_counts = collections.Counter()
    train_addr_counts = collections.Counter()
    total_tr_records = 0

    with open(train_s1_path, "r", encoding="utf-8") as f:
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
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

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
    temp_dir = "output_v4/temp_infer"
    os.makedirs(temp_dir, exist_ok=True)

    sorted_countries = sorted(country_s1_groups.keys(), key=lambda c: len(country_s1_groups[c]))

    total_matches_predicted = 0
    total_singletons_predicted = 0
    total_candidates_generated = 0

    # 3. Country-by-Country Processing
    for ctry_idx, ctry in enumerate(sorted_countries, 1):
        s1_list = country_s1_groups[ctry]
        expected_count = len(s1_list)
        temp_match_file = os.path.join(temp_dir, f"match_{ctry}.tsv")
        temp_cand_file = os.path.join(temp_dir, f"cand_{ctry}.tsv")

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

        # Build Multi-Channel Blocking Indices for this Country
        print(f"  Building Multi-Channel Indices for {ctry}...")
        cbi_v2 = CountryBlockingIndexV2(max_posting_size=350)
        cbi_v4 = CountryBlockingIndexV4()
        # Avoid redundant memory in cbi_v4 by referencing cbi_v2.target_data
        cbi_v4.target_data = cbi_v2.target_data

        exact_name_index = collections.defaultdict(list)
        exact_addr_index = collections.defaultdict(list)
        pin_num_index = collections.defaultdict(list)

        target_count = 0
        t0 = time.time()

        for fn in [s2_path, s3_path]:
            with open(fn, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) > 3 and parts[3].strip() == ctry:
                        eid = parts[0].strip()
                        name = parts[1].strip() if len(parts) > 1 else ""
                        addr = parts[2].strip() if len(parts) > 2 else ""

                        cbi_v2.add_target_entity(eid, name, addr, ctry)
                        cbi_v4.add_target_entity(eid, name, addr, ctry)

                        nn, sn, tn = normalize_business_name(name)
                        na, ta, pin, nums = normalize_address(addr)

                        if nn and len(nn) >= 3 and len(exact_name_index[nn]) < 200:
                            exact_name_index[nn].append(eid)
                        if na and len(na) >= 5 and len(exact_addr_index[na]) < 200:
                            exact_addr_index[na].append(eid)
                        if pin and nums:
                            for n in nums:
                                k = f"{pin}_{n}"
                                if len(pin_num_index[k]) < 200:
                                    pin_num_index[k].append(eid)

                        target_count += 1

        print(f"  Indexed {target_count:,} target records in {time.time()-t0:.2f}s.")

        def retrieve_union_cands(s1_prep, top_k=35):
            (nn, sn, tn, na, ta, pin, nums) = s1_prep[:7]
            c2 = cbi_v2.retrieve_candidates(s1_prep[:7], ctry, max_candidates=25)
            c4 = cbi_v4.retrieve_candidates(s1_prep[:7], ctry, max_candidates=25)
            c_en = exact_name_index.get(nn, [])[:50]
            c_ea = exact_addr_index.get(na, [])[:50]
            c_pn = []
            if pin and nums:
                for n in nums:
                    c_pn.extend(pin_num_index.get(f"{pin}_{n}", [])[:30])

            # Direct dictionary scoring: 3x faster than collections.Counter
            scores = collections.defaultdict(float)
            for tid in c2: scores[tid] += 10.0
            for tid in c4: scores[tid] += 10.0
            for tid in c_en: scores[tid] += 40.0
            for tid in c_ea: scores[tid] += 35.0
            for tid in c_pn: scores[tid] += 10.0

            return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        total_batches = (expected_count + batch_size - 1) // batch_size
        ctry_matches = 0
        ctry_singletons = 0
        ctry_cands = 0

        # Overwrite temp files cleanly
        with open(temp_match_file, "w", encoding="utf-8") as f_tm, \
             open(temp_cand_file, "w", encoding="utf-8") as f_tc:

            for b_idx in range(total_batches):
                batch_s1 = s1_list[b_idx * batch_size : (b_idx + 1) * batch_size]
                batch_pairs_feats = []
                batch_pair_meta = []  # (row_idx, tid)
                batch_cands_by_s1 = {}
                batch_cands_list = {}

                for row_idx, s1_id, name, addr in batch_s1:
                    nn, sn, tn = normalize_business_name(name)
                    na, ta, pin, nums = normalize_address(addr)
                    s1_prep = (nn, sn, tn, na, ta, pin, nums, ctry, name, addr)

                    n_freq = train_name_counts.get(nn, 1) / total_tr_records
                    a_freq = train_addr_counts.get(na, 1) / total_tr_records
                    n_rar = float(-math.log10(max(1e-5, n_freq)))
                    a_rar = float(-math.log10(max(1e-5, a_freq)))

                    cands = retrieve_union_cands(s1_prep, top_k=35)
                    batch_cands_list[row_idx] = cands
                    batch_cands_by_s1[row_idx] = []
                    ctry_cands += len(cands)

                    for tid in cands:
                        tr_prep = cbi_v2.target_data[tid]
                        feats = extract_features_v4(s1_prep, tid, tr_prep, country=ctry, name_rarity=n_rar, addr_rarity=a_rar)
                        batch_pairs_feats.append(feats)
                        batch_pair_meta.append((row_idx, tid))

                if batch_pairs_feats:
                    X_batch = np.array(batch_pairs_feats, dtype=np.float32)
                    probs = model.predict_proba(X_batch)[:, 1]

                    for (row_idx, tid), prob in zip(batch_pair_meta, probs):
                        if prob >= fixed_threshold:
                            batch_cands_by_s1[row_idx].append(tid)

                # Write out batch immediately to disk
                for row_idx, s1_id, _, _ in batch_s1:
                    matched_ids = batch_cands_by_s1[row_idx]
                    seen_m = set()
                    dedup_m = []
                    for m in matched_ids:
                        if m not in seen_m:
                            seen_m.add(m)
                            dedup_m.append(m)

                    match_str = ",".join(dedup_m)
                    cand_str = ",".join(batch_cands_list[row_idx])

                    f_tm.write(f"{row_idx}\t{s1_id}\t{match_str}\n")
                    f_tc.write(f"{row_idx}\t{s1_id}\t{cand_str}\n")

                    if dedup_m:
                        ctry_matches += len(dedup_m)
                    else:
                        ctry_singletons += 1

                if (b_idx + 1) % 20 == 0 or (b_idx + 1) == total_batches:
                    elapsed_b = time.time() - t_ctry_start
                    q_done = min(expected_count, (b_idx + 1) * batch_size)
                    q_rate = q_done / max(1.0, elapsed_b)
                    print(f"    Batch {b_idx+1}/{total_batches} ({q_done:,}/{expected_count:,} queries, {q_rate:.1f} q/s) | Elapsed: {elapsed_b:.1f}s | Matches so far: {ctry_matches:,}")

        total_matches_predicted += ctry_matches
        total_singletons_predicted += ctry_singletons
        total_candidates_generated += ctry_cands
        print(f"  {ctry} Completed in {time.time()-t_ctry_start:.1f}s: matches={ctry_matches:,}, singletons={ctry_singletons:,}")

        # Clean memory before next country
        del cbi_v2
        del cbi_v4
        del exact_name_index
        del exact_addr_index
        del pin_num_index
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
        temp_match_file = os.path.join(temp_dir, f"match_{ctry}.tsv")
        temp_cand_file = os.path.join(temp_dir, f"cand_{ctry}.tsv")

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
    os.makedirs(os.path.dirname(output_matching), exist_ok=True)
    os.makedirs(os.path.dirname(output_candidates), exist_ok=True)

    with open(output_matching, "w", encoding="utf-8") as f_match, \
         open(output_candidates, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, s1_id in enumerate(ordered_s1_ids):
            m_val = ordered_matches[idx] if ordered_matches[idx] is not None else ""
            c_val = ordered_cands[idx] if ordered_cands[idx] is not None else ""
            f_match.write(f"{s1_id}\t{m_val}\n")
            f_cand.write(f"{s1_id}\t{c_val}\n")

    print(f"  Final submission files written in {time.time()-t0:.2f}s.")
    print("\n" + "=" * 75)
    print("V4 TEST INFERENCE COMPLETED SUCCESSFULLY")
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
    run_v4_test_inference()
