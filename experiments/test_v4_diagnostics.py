#!/usr/bin/env python3
"""
Diagnostic test script to verify V4 features, negative mining, calibration,
and threshold evaluation on a 1.2M target pool.
"""

import os
import sys
import time
import json
import random
import collections
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from business_entity_resolution.src.features import char_ngram_cosine, jaccard_similarity
from experiments.blocking_v2 import CountryBlockingIndexV2
from experiments.blocking_v4 import CountryBlockingIndexV4, clean_ocr_leetspeak, normalize_indic_phonetics, get_consonants
from experiments.phase7_improved_features import token_containment_ratio

FEATURE_NAMES_V4_CLEAN = [
    # Baseline 36 features from V2
    "name_exact_match",
    "name_stem_exact_match",
    "name_fuzz_ratio",
    "name_fuzz_partial_ratio",
    "name_fuzz_token_sort",
    "name_fuzz_token_set",
    "name_levenshtein_sim",
    "name_char3gram_cosine",
    "name_token_jaccard",
    "name_len_diff",
    "name_tok_cnt_diff",
    "addr_exact_match",
    "addr_fuzz_ratio",
    "addr_fuzz_token_set",
    "addr_levenshtein_sim",
    "addr_char3gram_cosine",
    "addr_token_jaccard",
    "pin_match_score",
    "numeric_house_jaccard",
    "numeric_house_overlap_cnt",
    "addr_len_diff",
    "addr_missing",
    "source_is_s2",
    "combined_name_addr_sim",
    "strong_exact_name_addr",
    "strong_name_and_num",
    "strong_name_and_pin",
    "name_jaro_winkler",
    "name_consonant_skeleton_match",
    "name_token_containment",
    "name_char2gram_cosine",
    "addr_jaro_winkler",
    "addr_token_containment",
    "exact_name_with_empty_addr",
    "exact_addr_with_diff_name",
    "country_is_india",
    # Additional high-signal V4 features (strictly pairwise)
    "name_leetspeak_exact_match",
    "name_first_token_match",
    "name_last_token_match",
    "name_phonetic_sim",
    "pin_and_house_both_match",
    "name_len_ratio",
]

def extract_features_clean(s1_prep: tuple, target_id: str, tr_prep: tuple, country: str = "US") -> list:
    (s1_norm_n, s1_stem_n, s1_toks_n, s1_norm_a, s1_toks_a, s1_pin, s1_nums) = s1_prep[:7]
    (tr_norm_n, tr_stem_n, tr_toks_n, tr_norm_a, tr_toks_a, tr_pin, tr_nums) = tr_prep[:7]

    # Baseline Name Features
    name_exact = 1.0 if s1_norm_n == tr_norm_n and s1_norm_n else 0.0
    name_stem_exact = 1.0 if s1_stem_n == tr_stem_n and s1_stem_n else 0.0
    name_ratio = fuzz.ratio(s1_stem_n, tr_stem_n) / 100.0
    name_partial = fuzz.partial_ratio(s1_stem_n, tr_stem_n) / 100.0
    name_sort = fuzz.token_sort_ratio(s1_norm_n, tr_norm_n) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm_n, tr_norm_n) / 100.0
    name_lev = Levenshtein.normalized_similarity(s1_stem_n, tr_stem_n)
    name_c3 = char_ngram_cosine(s1_stem_n, tr_stem_n, n=3)
    name_jacc = jaccard_similarity(set(s1_toks_n), set(tr_toks_n))
    name_len_diff = float(abs(len(s1_norm_n) - len(tr_norm_n)))
    name_tok_diff = float(abs(len(s1_toks_n) - len(tr_toks_n)))

    # Baseline Address Features
    addr_missing = 1.0 if not tr_norm_a else 0.0
    if not tr_norm_a:
        addr_exact = 0.0; addr_ratio = 0.0; addr_set = 0.0; addr_lev = 0.0
        addr_c3 = 0.0; addr_jacc = 0.0; pin_score = 0.0; num_jacc = 0.0; num_cnt = 0.0
        addr_len_diff = float(len(s1_norm_a))
    else:
        addr_exact = 1.0 if s1_norm_a == tr_norm_a else 0.0
        addr_ratio = fuzz.ratio(s1_norm_a, tr_norm_a) / 100.0
        addr_set = fuzz.token_set_ratio(s1_norm_a, tr_norm_a) / 100.0
        addr_lev = Levenshtein.normalized_similarity(s1_norm_a, tr_norm_a)
        addr_c3 = char_ngram_cosine(s1_norm_a, tr_norm_a, n=3)
        addr_jacc = jaccard_similarity(set(s1_toks_a), set(tr_toks_a))

        if s1_pin and tr_pin:
            pin_score = 1.0 if s1_pin == tr_pin else -1.0
        else:
            pin_score = 0.0

        if s1_nums and tr_nums:
            num_jacc = jaccard_similarity(s1_nums, tr_nums)
            num_cnt = float(len(s1_nums & tr_nums))
        else:
            num_jacc = 0.0
            num_cnt = 0.0
        addr_len_diff = float(abs(len(s1_norm_a) - len(tr_norm_a)))

    source_is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    combined_sim = (0.6 * name_set) + (0.4 * addr_set)
    strong_exact = 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0
    strong_name_num = 1.0 if (name_set > 0.85 and num_cnt >= 1.0) else 0.0
    strong_name_pin = 1.0 if (name_set > 0.85 and pin_score == 1.0) else 0.0

    name_jw = float(JaroWinkler.similarity(s1_stem_n, tr_stem_n))
    skel1 = get_consonants(s1_stem_n)
    skel2 = get_consonants(tr_stem_n)
    skel_match = 1.0 if (skel1 == skel2 and len(skel1) >= 3) else 0.0

    name_contain = token_containment_ratio(s1_toks_n, tr_toks_n)
    name_c2 = char_ngram_cosine(s1_stem_n, tr_stem_n, n=2)

    if not tr_norm_a:
        addr_jw = 0.0
        addr_contain = 0.0
    else:
        addr_jw = float(JaroWinkler.similarity(s1_norm_a, tr_norm_a))
        addr_contain = token_containment_ratio(s1_toks_a, tr_toks_a)

    exact_name_empty_addr = 1.0 if (name_stem_exact == 1.0 and addr_missing == 1.0) else 0.0
    exact_addr_diff_name = 1.0 if (addr_set > 0.85 and num_cnt >= 1.0 and name_set < 0.5) else 0.0
    country_is_in = 1.0 if country == "India" else 0.0

    # Additional V4 features
    s1_clean_n = clean_ocr_leetspeak(s1_norm_n)
    tr_clean_n = clean_ocr_leetspeak(tr_norm_n)
    name_leet = 1.0 if s1_clean_n == tr_clean_n and s1_clean_n else 0.0

    first_tok_m = 1.0 if s1_toks_n and tr_toks_n and s1_toks_n[0] == tr_toks_n[0] else 0.0
    last_tok_m = 1.0 if s1_toks_n and tr_toks_n and s1_toks_n[-1] == tr_toks_n[-1] else 0.0

    s1_phon = normalize_indic_phonetics(s1_norm_n)
    tr_phon = normalize_indic_phonetics(tr_norm_n)
    name_phon = fuzz.token_sort_ratio(s1_phon, tr_phon) / 100.0

    pin_house_both = 1.0 if (pin_score == 1.0 and num_cnt >= 1.0) else 0.0

    len_s1 = max(1, len(s1_stem_n))
    len_tr = max(1, len(tr_stem_n))
    name_len_ratio = min(len_s1, len_tr) / max(len_s1, len_tr)

    return [
        name_exact, name_stem_exact, name_ratio, name_partial, name_sort,
        name_set, name_lev, name_c3, name_jacc, name_len_diff, name_tok_diff,
        addr_exact, addr_ratio, addr_set, addr_lev, addr_c3, addr_jacc,
        pin_score, num_jacc, num_cnt, addr_len_diff, addr_missing,
        source_is_s2, combined_sim, strong_exact, strong_name_num, strong_name_pin,
        name_jw, skel_match, name_contain, name_c2, addr_jw, addr_contain,
        exact_name_empty_addr, exact_addr_diff_name, country_is_in,
        # V4
        name_leet, first_tok_m, last_tok_m, name_phon, pin_house_both, name_len_ratio
    ]

def compute_strict_entity_f05(gt_ids: set, pred_ids: set):
    len_gt = len(gt_ids)
    len_pred = len(pred_ids)
    if len_gt == 0:
        return (0, 0, 0, 1.0, 1.0, 1.0) if len_pred == 0 else (0, len_pred, 0, 0.0, 0.0, 0.0)
    tp = len(gt_ids & pred_ids)
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)
    if len_pred == 0 or tp == 0:
        return tp, fp, fn, 0.0, 0.0, 0.0
    p = tp / len_pred
    r = tp / len_gt
    f05 = (1.25 * p * r) / (0.25 * p + r)
    return tp, fp, fn, p, r, f05

def main():
    print("Running Clean V4 Diagnostics...")
    random.seed(42)
    np.random.seed(42)

    # Load 15,000 train S1, 2,000 val S1
    gt_all = {}
    with open("dataset/train/train_ground_truth.tsv") as f:
        f.readline()
        for idx, line in enumerate(f):
            p = line.rstrip("\r\n").split("\t")
            mids = set(p[1].strip().split(",")) if len(p) > 1 and p[1].strip() else set()
            gt_all[p[0].strip()] = mids
            if idx >= 100000: break

    s1_all = list(gt_all.keys())
    random.shuffle(s1_all)
    val_s1_ids = s1_all[:2000]
    train_s1_ids = s1_all[2000:17000]

    gt_val = {s: gt_all[s] for s in val_s1_ids}
    gt_train = {s: gt_all[s] for s in train_s1_ids}

    needed_s1 = set(val_s1_ids) | set(train_s1_ids)
    s1_records = {}
    with open("dataset/train/train_source1.tsv") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0].strip()
            if eid in needed_s1:
                s1_records[eid] = p
                if len(s1_records) == len(needed_s1): break

    s1_prepped = {}
    for sid in needed_s1:
        p = s1_records[sid]
        nn, sn, tn = normalize_business_name(p[1])
        na, ta, pin, nums = normalize_address(p[2])
        s1_prepped[sid] = (nn, sn, tn, na, ta, pin, nums, p[3].strip(), p[1], p[2])

    val_true_targets = set()
    for mids in gt_val.values(): val_true_targets |= mids
    train_true_targets = set()
    for mids in gt_train.values(): train_true_targets |= mids

    # Targets: Train targets + Val targets + 400k distractors (fast diagnostic)
    print("Loading target records...")
    target_records = {}
    needed_true = val_true_targets | train_true_targets
    bg_limit = 400000
    bg_added = 0

    for fn in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        with open(fn) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0].strip()
                if eid in needed_true:
                    target_records[eid] = p
                elif bg_added < bg_limit:
                    target_records[eid] = p
                    bg_added += 1

    print(f"Loaded {len(target_records):,} target records.")

    # Multi-Channel Indices
    cbi_v2 = CountryBlockingIndexV2(max_posting_size=350)
    cbi_v4 = CountryBlockingIndexV4()
    exact_name_index = collections.defaultdict(lambda: collections.defaultdict(list))
    exact_addr_index = collections.defaultdict(lambda: collections.defaultdict(list))
    pin_num_index = collections.defaultdict(lambda: collections.defaultdict(list))

    for eid, p in target_records.items():
        name = p[1] if len(p) > 1 else ""
        addr = p[2] if len(p) > 2 else ""
        ctry = p[3].strip() if len(p) > 3 else "US"
        cbi_v2.add_target_entity(eid, name, addr, ctry)
        cbi_v4.add_target_entity(eid, name, addr, ctry)
        nn, sn, tn = normalize_business_name(name)
        na, ta, pin, nums = normalize_address(addr)
        if nn and len(nn) >= 3:
            exact_name_index[ctry][nn].append(eid)
        if na and len(na) >= 5:
            exact_addr_index[ctry][na].append(eid)
        if pin and nums:
            for n in nums:
                pin_num_index[ctry][f"{pin}_{n}"].append(eid)

    def retrieve_union(s1_prep, country, top_k=35):
        (nn, sn, tn, na, ta, pin, nums) = s1_prep[:7]
        c2 = cbi_v2.retrieve_candidates(s1_prep[:7], country, max_candidates=25)
        c4 = cbi_v4.retrieve_candidates(s1_prep[:7], country, max_candidates=25)
        c_exact_n = exact_name_index[country].get(nn, [])[:50]
        c_exact_a = exact_addr_index[country].get(na, [])[:50]
        c_pin_num = []
        if pin and nums:
            for n in nums:
                c_pin_num.extend(pin_num_index[country].get(f"{pin}_{n}", [])[:30])

        counts = collections.Counter(c2 + c4 + c_exact_n + c_exact_a + c_pin_num)
        scored = []
        for tid, cnt in counts.items():
            sc = cnt * 10.0
            if tid in c_exact_n: sc += 30.0
            if tid in c_exact_a: sc += 25.0
            scored.append((tid, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in scored[:top_k]]

    # Training Data
    print("Extracting train features...")
    X_train_list = []
    y_train_list = []

    for sid in train_s1_ids:
        prep = s1_prepped[sid]
        ctry = prep[7]
        true_m = gt_train[sid]

        for tid in true_m:
            if tid in cbi_v2.target_data:
                tr_prep = cbi_v2.target_data[tid]
                f = extract_features_clean(prep, tid, tr_prep, country=ctry)
                X_train_list.append(f)
                y_train_list.append(1)

        cands = retrieve_union(prep, ctry, top_k=20)
        neg_added = 0
        for tid in cands:
            if tid not in true_m and tid in cbi_v2.target_data:
                tr_prep = cbi_v2.target_data[tid]
                f = extract_features_clean(prep, tid, tr_prep, country=ctry)
                X_train_list.append(f)
                y_train_list.append(0)
                neg_added += 1
                if neg_added >= max(2, len(true_m) * 2):
                    break

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"Train size: {len(X_train):,} (Pos: {np.sum(y_train==1):,}, Neg: {np.sum(y_train==0):,})")

    # Validation Candidates
    print("Extracting val features...")
    val_cand_hits = 0
    val_total_gt = sum(len(gt_val[s]) for s in val_s1_ids)
    val_pairs = []
    val_meta = []

    for sid in val_s1_ids:
        prep = s1_prepped[sid]
        ctry = prep[7]
        true_m = gt_val[sid]
        cands = retrieve_union(prep, ctry, top_k=35)
        val_cand_hits += len(true_m & set(cands))

        for tid in cands:
            tr_prep = cbi_v2.target_data[tid]
            f = extract_features_clean(prep, tid, tr_prep, country=ctry)
            val_pairs.append(f)
            val_meta.append((sid, tid))

    X_val = np.array(val_pairs, dtype=np.float32)
    print(f"Val candidate recall: {(val_cand_hits/val_total_gt)*100:.2f}% | Total pairs: {len(X_val):,}")

    # Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'learning_rate': 0.06,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]

    # Group by S1
    s1_scores = collections.defaultdict(dict)
    for (sid, tid), p in zip(val_meta, val_probs):
        s1_scores[sid][tid] = float(p)

    # Evaluate Thresholds
    print("\n--- Raw Probability Evaluation (Standard Thresholding) ---")
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]
    n_sing = sum(len(m) == 0 for m in gt_val.values())

    for th in thresholds:
        scores = []
        tot_fp = 0
        tot_fn = 0
        tot_pred = 0
        sing_cor = 0

        for sid in val_s1_ids:
            gt_ids = gt_val[sid]
            c_dict = s1_scores[sid]
            pred_set = {tid for tid, p in c_dict.items() if p >= th}

            tp, fp, fn, p, r, f05 = compute_strict_entity_f05(gt_ids, pred_set)
            scores.append(f05)
            tot_fp += fp
            tot_fn += fn
            tot_pred += len(pred_set)
            if len(gt_ids) == 0 and len(pred_set) == 0:
                sing_cor += 1

        macro_f = float(np.mean(scores))
        sing_acc = (sing_cor / n_sing) * 100.0 if n_sing > 0 else 0
        avg_pred = tot_pred / len(val_s1_ids)
        print(f"tau={th:<5.2f} | Strict Macro F0.5: {macro_f:.4f} | SingAcc: {sing_acc:.2f}% | FP: {tot_fp:>4,} | FN: {tot_fn:>4,} | AvgPred: {avg_pred:.2f}")

if __name__ == "__main__":
    main()
