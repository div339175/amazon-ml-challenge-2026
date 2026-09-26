#!/usr/bin/env python3
"""
Phase 5: High-Recall Multi-Pass Blocking Strategies Evaluation
Measures candidate recall and candidate volume for each blocking strategy independently
and in combination.
"""

import os
import sys
import collections
import re
import numpy as np

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
    LEGAL_TOKENS_SET,
)

ADDR_STOPWORDS = {
    'st', 'rd', 'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'ter', 'hwy',
    'apt', 'ste', 'fl', 'bldg', 'rm', 'n', 's', 'e', 'w', 'opp', 'near',
    'pobox', 'rue', 'chemin', 'allee', 'india', 'us', 'usa', 'france',
    'state', 'city', 'dist', 'district', 'region', 'floor', 'suite', 'block',
    'phase', 'sector', 'nagar', 'colony', 'road', 'street', 'main', 'cross',
    'layout', 'delhi', 'mumbai', 'bangalore', 'chennai', 'paris', 'new'
}

def get_consonant_skeleton(word: str) -> str:
    """Extracts first letter + consonants for phonetic-like signature."""
    if not word or len(word) < 3: return word
    first = word[0]
    cons = "".join([c for c in word[1:] if c in "bcdfghjklmnpqrstvwxyz"])
    return (first + cons)[:5]

def get_multi_pass_keys(stem_name, name_tokens, norm_addr, addr_tokens, pin, nums):
    """
    Extracts blocking keys tagged by strategy.
    Returns: dict of strategy_name -> list of keys
    """
    strat_keys = collections.defaultdict(list)

    # 1. Exact normalized stem name
    if stem_name and len(stem_name) >= 3:
        strat_keys["1_exact_name"].append("nx:" + stem_name)

    # 2. Concatenated name string (handles merged words, domains)
    if len(name_tokens) >= 2:
        concat = "".join(name_tokens)
        if len(concat) >= 5:
            strat_keys["2_concat_name"].append("nc:" + concat[:16])
    elif stem_name and len(stem_name) >= 5:
        concat = "".join(stem_name.split())
        strat_keys["2_concat_name"].append("nc:" + concat[:16])

    # 3. Rare/distinctive name tokens
    for tok in name_tokens:
        if len(tok) >= 3 and tok not in LEGAL_TOKENS_SET:
            strat_keys["3_name_tokens"].append("nt:" + tok)

    # 4. Name prefix (first token 3-char prefix)
    if name_tokens:
        first = name_tokens[0]
        if len(first) >= 4 and first not in LEGAL_TOKENS_SET:
            strat_keys["4_name_prefix"].append("np:" + first[:3])

    # 5. Phonetic / Consonant skeleton
    if name_tokens:
        skel = "_".join([get_consonant_skeleton(t) for t in name_tokens[:2]])
        if len(skel) >= 4:
            strat_keys["5_phonetic_skel"].append("psk:" + skel)

    # 6. House number + Locality/Word (strong spatial signal)
    sig_addr = [w for w in addr_tokens if len(w) >= 4 and w not in ADDR_STOPWORDS][:4]
    num_list = sorted(list(nums))[:2]
    for n in num_list:
        for w in sig_addr[:2]:
            strat_keys["6_num_locality"].append(f"anw:{n}_{w}")

    # 7. Postal/PIN + name prefix
    if pin and name_tokens:
        first = name_tokens[0]
        if len(first) >= 3:
            strat_keys["7_pin_name_pref"].append(f"pnp:{pin}_{first[:3]}")

    # 8. House number + Postal/PIN
    if pin and num_list:
        for n in num_list:
            strat_keys["8_num_pin"].append(f"anp:{n}_{pin}")

    # 9. Multi-word address pair (for DBA / different brand name)
    if len(sig_addr) >= 2:
        strat_keys["9_addr_pair"].append(f"amp:{sig_addr[0]}_{sig_addr[1]}")

    return strat_keys

def main():
    print("Testing Phase 5 Multi-Pass Blocking Strategies on validation slice...")
    # Load 3000 validation S1 entities and ground truth
    gt_map = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            gt_map[parts[0].strip()] = set(parts[1].strip().split(",")) if len(parts) > 1 and parts[1].strip() else set()
            if idx >= 100000: break

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

    # Load targets (needed true matches + 100k background records)
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
                elif bg_added < 100000:
                    targets[eid] = parts
                    bg_added += 1

    print(f"Validation: {len(val_s1)} S1 entities, {len(needed_targets)} true matches, {len(targets):,} indexed targets.")

    # Build inverted index by strategy
    # strat_indices[country][strat_name][key] = [tids...]
    strat_indices = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    target_prepped = {}
    MAX_POST = 300

    for tid, parts in targets.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        n_norm, n_stem, n_toks = normalize_business_name(name)
        a_norm, a_toks, pin, nums = normalize_address(addr)
        target_prepped[tid] = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums)

        s_keys = get_multi_pass_keys(n_stem, n_toks, a_norm, a_toks, pin, nums)
        c_dict = strat_indices[ctry]
        for s_name, k_list in s_keys.items():
            k_dict = c_dict[s_name]
            for k in k_list:
                pl = k_dict[k]
                if len(pl) < MAX_POST:
                    pl.append(tid)

    total_gt_matches = sum(len(gt_map[s]) for s in val_s1)
    print(f"Total Ground Truth Matches to retrieve: {total_gt_matches:,}")

    # Evaluate each strategy individually and cumulatively
    all_strat_names = sorted(list({sn for cd in strat_indices.values() for sn in cd.keys()}))
    strat_hits = {sn: 0 for sn in all_strat_names}
    cumulative_cands = {s: set() for s in val_s1}
    cumulative_recall_progression = []

    # Pre-extract S1 queries
    s1_queries = {}
    for s1_id, parts in val_s1.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        n_norm, n_stem, n_toks = normalize_business_name(name)
        a_norm, a_toks, pin, nums = normalize_address(addr)
        s_keys = get_multi_pass_keys(n_stem, n_toks, a_norm, a_toks, pin, nums)
        s1_queries[s1_id] = (ctry, s_keys)

    # 1. Individual performance
    for s_name in all_strat_names:
        hits = 0
        cand_counts = []
        for s1_id, (ctry, s_keys) in s1_queries.items():
            true_mids = gt_map[s1_id]
            k_list = s_keys.get(s_name, [])
            c_index = strat_indices[ctry][s_name]
            retrieved = set()
            for k in k_list:
                retrieved.update(c_index.get(k, []))
            hits += len(true_mids & retrieved)
            cand_counts.append(len(retrieved))
        rec = (hits / total_gt_matches) * 100
        avg_c = np.mean(cand_counts)
        strat_hits[s_name] = (rec, avg_c)
        print(f"Strategy {s_name:20s}: Recall = {rec:6.2f}% | Avg candidates = {avg_c:6.1f}")

    # 2. Cumulative Union
    print("\n--- Cumulative Multi-Pass Union Progression ---")
    cum_hits = 0
    cum_set_by_s1 = {s: set() for s in val_s1}
    for s_name in all_strat_names:
        for s1_id, (ctry, s_keys) in s1_queries.items():
            k_list = s_keys.get(s_name, [])
            c_index = strat_indices[ctry][s_name]
            for k in k_list:
                cum_set_by_s1[s1_id].update(c_index.get(k, []))
        total_hits = sum(len(gt_map[s] & cum_set_by_s1[s]) for s in val_s1)
        avg_cands = np.mean([len(cum_set_by_s1[s]) for s in val_s1])
        rec = (total_hits / total_gt_matches) * 100
        print(f"+ {s_name:20s} -> Cumulative Recall = {rec:6.2f}% | Avg candidates = {avg_cands:6.1f}")

    # Save to report
    rep_path = "experiments/phase5_blocking_strategies.md"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Multi-Pass Blocking Strategies Evaluation\n\n")
        f.write("| Strategy | Individual Recall | Avg Cands/S1 | Description |\n")
        f.write("| --- | --- | --- | --- |\n")
        for sn in all_strat_names:
            rec, avg_c = strat_hits[sn]
            f.write(f"| **{sn}** | {rec:.2f}% | {avg_c:.1f} | Key: `{sn}` |\n")
        f.write(f"\n### Final Cumulative Multi-Pass Recall: **{rec:.2f}%** (Avg {avg_cands:.1f} candidates per S1)\n")
    print(f"Report saved to {rep_path}")

if __name__ == "__main__":
    main()
