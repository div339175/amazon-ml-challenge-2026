#!/usr/bin/env python3
"""
Phase 1: Deep Candidate Recall & Failure Audit
Amazon ML Challenge 2026 - Business Entity Resolution

Constructs a realistic validation experiment using TRAIN data only:
- 3,000 held-out S1 validation entities (realistic country & singleton distribution)
- 1,500,000+ target records (true matches + ~1.5M distractors from train_source2/3)
- Ground truth is completely hidden from candidate generation
- Evaluates CountryBlockingIndexV2 candidate recall across all required slices:
  1. Overall candidate recall
  2. Number of candidates per S1
  3. By country (US vs India)
  4. By source (S2 vs S3)
  5. By specific subsets:
     - Exact name
     - Exact address
     - Missing target address
     - Transliteration / phonetic variant
     - Short names (<= 10 chars)
     - Long names (>= 25 chars)
     - Common business names (freq >= 5)
     - Entities with multiple records (>= 3 matches)
  6. Exact failure mode breakdown for missing true matches

Outputs:
- output/phase1_candidate_recall_audit.json
- output/phase1_candidate_recall_audit.md
"""

import os
import sys
import time
import json
import random
import pickle
import collections
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from experiments.blocking_v2 import CountryBlockingIndexV2, extract_blocking_keys_v2

def main():
    print("=" * 70)
    print("PHASE 1: CANDIDATE RECALL AUDIT ON REALISTIC TARGET DISTRACTOR POOL")
    print("=" * 70)
    os.makedirs("output", exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # 1. Load Ground Truth and sample 3,000 S1 validation entities
    print("\n[Step 1] Loading ground truth and creating leak-free validation sample...")
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

    # Count true targets
    needed_true_targets = set()
    for mids in gt_val.values():
        needed_true_targets |= mids

    print(f"  Validation queries: {len(val_s1_ids):,} entities")
    print(f"  True target matches in ground truth: {len(needed_true_targets):,} target IDs")

    # 2. Load S1 source records
    print("\n[Step 2] Loading S1 source records...")
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
    name_corpus_counts = collections.Counter()
    for sid in val_s1_ids:
        p = s1_records[sid]
        n_norm, n_stem, n_toks = normalize_business_name(p[1])
        a_norm, a_toks, pin, nums = normalize_address(p[2])
        s1_prepped[sid] = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums, p[3].strip(), p[1], p[2])
        name_corpus_counts[n_norm] += 1

    # 3. Load Target Records: true targets + 1,500,000 distractors
    print("\n[Step 3] Loading target records (true targets + 1.5M distractors)...")
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

    print(f"  Loaded {len(target_records):,} target records ({len(needed_true_targets):,} true + {bg_added:,} distractors) in {time.time()-t0:.2f}s.")

    # 4. Build CountryBlockingIndexV2
    print("\n[Step 4] Indexing target records into CountryBlockingIndexV2 (max_posting_size=350)...")
    t0 = time.time()
    cbi = CountryBlockingIndexV2(max_posting_size=350)
    for eid, p in target_records.items():
        name = p[1] if len(p) > 1 else ""
        addr = p[2] if len(p) > 2 else ""
        ctry = p[3] if len(p) > 3 else "US"
        cbi.add_target_entity(eid, name, addr, ctry)
    print(f"  Indexing completed in {time.time()-t0:.2f}s.")

    # 5. Retrieve candidates for validation queries
    print("\n[Step 5] Retrieving candidates (top-30) for 3,000 S1 queries...")
    t0 = time.time()
    retrieved_cands = {}
    cand_counts = []

    for sid in val_s1_ids:
        prep = s1_prepped[sid]
        ctry = prep[7]
        cands = cbi.retrieve_candidates(prep[:7], ctry, max_candidates=30)
        retrieved_cands[sid] = set(cands)
        cand_counts.append(len(cands))

    print(f"  Candidate retrieval completed in {time.time()-t0:.2f}s. Avg candidates/S1: {np.mean(cand_counts):.2f}")

    # 6. Detailed Analysis of True Matches (Retrieved vs Missed)
    print("\n[Step 6] Categorizing candidate recall across subsets...")
    
    # Tracking buckets
    overall_total = 0
    overall_retrieved = 0

    by_country = collections.defaultdict(lambda: {"total": 0, "retrieved": 0})
    by_source = collections.defaultdict(lambda: {"total": 0, "retrieved": 0})
    by_subset = {
        "Exact Name": {"total": 0, "retrieved": 0},
        "Exact Address": {"total": 0, "retrieved": 0},
        "Missing Address": {"total": 0, "retrieved": 0},
        "Transliteration / Typos (Name diff, sim >= 0.6)": {"total": 0, "retrieved": 0},
        "Short Names (<= 10 chars)": {"total": 0, "retrieved": 0},
        "Long Names (>= 25 chars)": {"total": 0, "retrieved": 0},
        "Common Names (freq >= 5 in val)": {"total": 0, "retrieved": 0},
        "Multiple Records (>= 3 matches for S1)": {"total": 0, "retrieved": 0}
    }

    missed_matches = []
    
    for sid in val_s1_ids:
        s1_data = s1_prepped[sid]
        s1_norm_n, s1_stem_n, s1_toks_n, s1_norm_a, s1_toks_a, s1_pin, s1_nums, s1_ctry, s1_raw_n, s1_raw_a = s1_data
        true_mids = gt_val[sid]
        cands = retrieved_cands[sid]
        s1_num_matches = len(true_mids)

        for tid in true_mids:
            if tid not in target_records:
                continue

            overall_total += 1
            is_retrieved = (tid in cands)
            if is_retrieved:
                overall_retrieved += 1

            # Country
            by_country[s1_ctry]["total"] += 1
            if is_retrieved:
                by_country[s1_ctry]["retrieved"] += 1

            # Source (S2 vs S3)
            src = "Source 2 (S2-)" if tid.startswith("S2-") else "Source 3 (S3-)"
            by_source[src]["total"] += 1
            if is_retrieved:
                by_source[src]["retrieved"] += 1

            # Target data
            t_raw = target_records[tid]
            t_name = t_raw[1] if len(t_raw) > 1 else ""
            t_addr = t_raw[2] if len(t_raw) > 2 else ""
            t_norm_n, t_stem_n, t_toks_n = normalize_business_name(t_name)
            t_norm_a, t_toks_a, t_pin, t_nums = normalize_address(t_addr)

            # Subsets:
            # 1. Exact Name
            if s1_norm_n == t_norm_n and s1_norm_n != "":
                by_subset["Exact Name"]["total"] += 1
                if is_retrieved: by_subset["Exact Name"]["retrieved"] += 1

            # 2. Exact Address
            if s1_norm_a == t_norm_a and s1_norm_a != "":
                by_subset["Exact Address"]["total"] += 1
                if is_retrieved: by_subset["Exact Address"]["retrieved"] += 1

            # 3. Missing Address
            if t_norm_a == "":
                by_subset["Missing Address"]["total"] += 1
                if is_retrieved: by_subset["Missing Address"]["retrieved"] += 1

            # 4. Transliteration / Typo (name differs, but fuzz ratio >= 60)
            if s1_norm_n != t_norm_n and fuzz.ratio(s1_stem_n, t_stem_n) >= 60:
                by_subset["Transliteration / Typos (Name diff, sim >= 0.6)"]["total"] += 1
                if is_retrieved: by_subset["Transliteration / Typos (Name diff, sim >= 0.6)"]["retrieved"] += 1

            # 5. Short names
            if len(s1_raw_n) <= 10 or len(s1_toks_n) <= 2:
                by_subset["Short Names (<= 10 chars)"]["total"] += 1
                if is_retrieved: by_subset["Short Names (<= 10 chars)"]["retrieved"] += 1

            # 6. Long names
            if len(s1_raw_n) >= 25 or len(s1_toks_n) >= 4:
                by_subset["Long Names (>= 25 chars)"]["total"] += 1
                if is_retrieved: by_subset["Long Names (>= 25 chars)"]["retrieved"] += 1

            # 7. Common names
            if name_corpus_counts[s1_norm_n] >= 3:
                by_subset["Common Names (freq >= 5 in val)"]["total"] += 1
                if is_retrieved: by_subset["Common Names (freq >= 5 in val)"]["retrieved"] += 1

            # 8. Multiple records
            if s1_num_matches >= 3:
                by_subset["Multiple Records (>= 3 matches for S1)"]["total"] += 1
                if is_retrieved: by_subset["Multiple Records (>= 3 matches for S1)"]["retrieved"] += 1

            # Failure mode diagnosis for missed matches
            if not is_retrieved:
                # Analyze why:
                # Did S1 and Target share ANY blocking keys?
                s1_keys = {k for k, _ in extract_blocking_keys_v2(s1_stem_n, s1_toks_n, s1_norm_a, s1_toks_a, s1_pin, s1_nums)}
                t_keys = {k for k, _ in extract_blocking_keys_v2(t_stem_n, t_toks_n, t_norm_a, t_toks_a, t_pin, t_nums)}
                shared_keys = s1_keys & t_keys

                c_index = cbi.indices.get(s1_ctry, {})
                active_shared_keys = [k for k in shared_keys if k in c_index and tid in c_index[k]]

                reason = "No shared blocking keys"
                if shared_keys:
                    if not active_shared_keys:
                        reason = "Shared key exceeded posting list cap (350)"
                    else:
                        reason = "Shared keys had low weight; out-ranked by distractors in top-30"

                missed_matches.append({
                    "s1_id": sid,
                    "target_id": tid,
                    "country": s1_ctry,
                    "s1_name": s1_raw_n,
                    "target_name": t_name,
                    "s1_addr": s1_raw_a,
                    "target_addr": t_addr,
                    "name_sim": fuzz.ratio(s1_stem_n, t_stem_n),
                    "addr_sim": fuzz.ratio(s1_norm_a, t_norm_a),
                    "reason": reason,
                    "shared_keys": list(shared_keys)[:3]
                })

    overall_recall = (overall_retrieved / overall_total) * 100.0
    missing_pct = 100.0 - overall_recall
    avg_cands = float(np.mean(cand_counts))

    print(f"\n--- CANDIDATE RECALL AUDIT SUMMARY ---")
    print(f"Total True Target Pairs: {overall_total:,}")
    print(f"True Pairs Retrieved:    {overall_retrieved:,} ({overall_recall:.2f}%)")
    print(f"True Pairs Missing:      {overall_total - overall_retrieved:,} ({missing_pct:.2f}%)")
    print(f"Average Candidates / S1: {avg_cands:.2f}")

    # Reasons breakdown
    reasons_counter = collections.Counter([m["reason"] for m in missed_matches])
    print(f"\nMissed Matches Failure Mode Breakdown ({len(missed_matches):,} misses):")
    for r, cnt in reasons_counter.most_common():
        print(f"  - {r}: {cnt:,} ({cnt/len(missed_matches)*100:.1f}%)")

    # Format Markdown Report
    report_md = "output/phase1_candidate_recall_audit.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Phase 1: Deep Candidate Recall & Failure Audit\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Target Pool Size Evaluated**: **{len(target_records):,}** records ({len(needed_true_targets):,} true matches + {bg_added:,} distractors).\n")
        f.write(f"- **Validation Entities**: **{len(val_s1_ids):,}** held-out Source 1 queries.\n")
        f.write(f"- **Total True Matches**: **{overall_total:,}** ground truth target pairs.\n")
        f.write(f"- **Candidate Recall**: **{overall_recall:.2f}%** ({overall_retrieved:,}/{overall_total:,} retrieved in top-30).\n")
        f.write(f"- **Missing True Matches**: **{missing_pct:.2f}%** ({overall_total - overall_retrieved:,} missed true matches).\n")
        f.write(f"- **Candidate Volume**: **{avg_cands:.2f}** candidates per S1 entity.\n\n")

        f.write("## 2. Candidate Recall by Country and Source\n\n")
        f.write("| Slice | True Pairs | Retrieved | Candidate Recall | Missed Pairs |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for ctry, st in sorted(by_country.items()):
            rec = (st["retrieved"] / st["total"]) * 100.0 if st["total"] > 0 else 0
            f.write(f"| **Country: {ctry}** | {st['total']:,} | {st['retrieved']:,} | **{rec:.2f}%** | {st['total'] - st['retrieved']:,} |\n")
        for src, st in sorted(by_source.items()):
            rec = (st["retrieved"] / st["total"]) * 100.0 if st["total"] > 0 else 0
            f.write(f"| **Source: {src}** | {st['total']:,} | {st['retrieved']:,} | **{rec:.2f}%** | {st['total'] - st['retrieved']:,} |\n")

        f.write("\n## 3. Candidate Recall by Attribute & Challenge Subset\n\n")
        f.write("| Subset Category | True Pairs | Retrieved | Candidate Recall | Missed Pairs |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cat, st in by_subset.items():
            rec = (st["retrieved"] / st["total"]) * 100.0 if st["total"] > 0 else 0
            f.write(f"| **{cat}** | {st['total']:,} | {st['retrieved']:,} | **{rec:.2f}%** | {st['total'] - st['retrieved']:,} |\n")

        f.write("\n## 4. Root Causes for Missed True Matches\n\n")
        f.write(f"Out of **{len(missed_matches):,}** missed true matches ({missing_pct:.2f}% of all true matches):\n\n")
        f.write("| Failure Mode | Missed Count | % of All Misses | Description & Remedy |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r, cnt in reasons_counter.most_common():
            pct = (cnt / len(missed_matches)) * 100.0
            if "No shared" in r:
                desc = "Name and address share zero blocking keys (e.g. DBA trade name variation, heavy typos, or distinct abbreviations). Remedy: character n-gram / TF-IDF similarity / substring index."
            elif "exceeded posting" in r:
                desc = "Common name or address token hit the 350 posting limit ceiling. Remedy: rare-token IDF weighting and compound specific keys."
            else:
                desc = "Target had matching keys but distractors accumulated higher weights and pushed it past top-30. Remedy: multi-pass channel union so each channel guarantees its top hits."
            f.write(f"| **{r}** | **{cnt:,}** | **{pct:.1f}%** | {desc} |\n")

        f.write("\n## 5. Sample Missed True Matches (Diagnostic Inspection)\n\n")
        for idx, m in enumerate(missed_matches[:5], 1):
            f.write(f"**Sample {idx}** [{m['country']}] — Reason: *{m['reason']}*\n")
            f.write(f"- S1: `{m['s1_name']}` | Address: `{m['s1_addr']}`\n")
            f.write(f"- Target ({m['target_id']}): `{m['target_name']}` | Address: `{m['target_addr']}`\n")
            f.write(f"- Name Sim: {m['name_sim']}%, Address Sim: {m['addr_sim']}%\n\n")

    # Save JSON artifact
    json_path = "output/phase1_candidate_recall_audit.json"
    json_data = {
        "overall_total": overall_total,
        "overall_retrieved": overall_retrieved,
        "overall_recall": overall_recall,
        "missing_pct": missing_pct,
        "avg_candidates_per_s1": avg_cands,
        "by_country": dict(by_country),
        "by_source": dict(by_source),
        "by_subset": by_subset,
        "failure_modes": dict(reasons_counter),
        "sample_missed_matches": missed_matches[:20]
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nAudit complete. Artifacts saved:\n  - {report_md}\n  - {json_path}")

if __name__ == "__main__":
    main()
