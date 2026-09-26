#!/usr/bin/env python3
"""
Phase 5, 6, 7 & 11: Training, Hard Negative Mining, Model Selection, Threshold Optimization & Experiments
Implements:
- Leak-free train / validation split by Source 1 entity ID
- Hard negative mining directly from the blocking stage
- Comprehensive model training (Logistic Regression, HistGradientBoosting, LightGBM/XGBoost)
- Systematic experiment comparison across Experiments A, B, C, D, E
- F_0.5 threshold optimization
- Automatic saving of experiment results to Markdown and CSV reports
"""

import os
import sys
import time
import math
import random
import pickle
import collections
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)
from business_entity_resolution.src.blocking import (
    CountryBlockingIndex,
    extract_blocking_keys
)
from business_entity_resolution.src.features import (
    extract_pair_features,
    FEATURE_NAMES
)
from business_entity_resolution.src.evaluate import (
    evaluate_predictions
)

EXPERIMENT_REPORT_MD = "output/experiment_results.md"
EXPERIMENT_REPORT_CSV = "output/experiment_results.csv"
MODEL_SAVE_PATH = "output/best_model.pkl"

def load_data_split(
    train_dir: str = "dataset/train",
    n_train_s1: int = 20000,
    n_val_s1: int = 5000,
    seed: int = 42
):
    """
    Loads a leak-free split of Source 1 entities and corresponding target records.
    Splits by Source 1 entity ID so no entity appears in both train and val.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading ground truth from {train_dir}...")
    gt_all = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            mids = parts[1].strip().split(",") if len(parts) > 1 and parts[1].strip() else []
            gt_all[s1_id] = set(mids)
            if idx >= 150000:  # Sample from the first 150k S1 records
                break
                
    s1_all_ids = list(gt_all.keys())
    random.shuffle(s1_all_ids)
    
    val_s1_ids = set(s1_all_ids[:n_val_s1])
    train_s1_ids = set(s1_all_ids[n_val_s1 : n_val_s1 + n_train_s1])
    
    gt_train = {s: gt_all[s] for s in train_s1_ids}
    gt_val = {s: gt_all[s] for s in val_s1_ids}
    
    print(f"Split: {len(train_s1_ids):,} train S1 entities, {len(val_s1_ids):,} val S1 entities.")
    
    needed_s1 = train_s1_ids | val_s1_ids
    needed_targets = set()
    for mids in gt_all.values():
        needed_targets |= mids
        
    # Read S1 records
    print("Loading Source 1 records...")
    s1_records = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            if eid in needed_s1:
                s1_records[eid] = parts
                if len(s1_records) == len(needed_s1):
                    break
                    
    # Read S2 and S3 target records (include needed true matches + background records for hard negatives)
    print("Loading Source 2 and Source 3 target records...")
    target_records = {}
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        bg_added = 0
        with open(os.path.join(train_dir, fn), "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[0].strip()
                if eid in needed_targets:
                    target_records[eid] = parts
                elif bg_added < 50000:
                    target_records[eid] = parts
                    bg_added += 1
                    
    print(f"Loaded {len(s1_records):,} S1 records and {len(target_records):,} target records.")
    return s1_records, target_records, gt_train, gt_val, train_s1_ids, val_s1_ids

def build_dataset_and_candidates(
    s1_records: Dict[str, List[str]],
    target_records: Dict[str, List[str]],
    gt_train: Dict[str, Set[str]],
    gt_val: Dict[str, Set[str]],
    train_s1_ids: Set[str],
    val_s1_ids: Set[str],
    max_cands_per_s1: int = 25
):
    """
    Builds the blocking index, extracts preprocessed tuples, retrieves candidates,
    mines hard negatives, and extracts feature matrices.
    """
    print("\n--- Phase 3: Building Blocking Index ---")
    t0 = time.time()
    cbi = CountryBlockingIndex(max_posting_size=250)
    for tid, parts in target_records.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        cbi.add_target_entity(tid, name, addr, ctry)
    print(f"Blocking index built in {time.time() - t0:.2f}s.")
    
    # Preprocess S1 entities
    s1_prepped = {}
    for s1_id, parts in s1_records.items():
        name = parts[1] if len(parts) > 1 else ""
        addr = parts[2] if len(parts) > 2 else ""
        ctry = parts[3] if len(parts) > 3 else ""
        n_norm, n_stem, n_toks = normalize_business_name(name)
        a_norm, a_toks, pin, nums = normalize_address(addr)
        s1_prepped[s1_id] = (n_norm, n_stem, n_toks, a_norm, a_toks, pin, nums, ctry)
        
    print("\n--- Phase 5: Generating Candidates & Mining Hard Negatives ---")
    t0 = time.time()
    
    # Validation candidates
    val_candidates = {}
    for s1_id in val_s1_ids:
        prep = s1_prepped[s1_id]
        ctry = prep[7]
        cands = cbi.retrieve_candidates(prep[:7], ctry, max_candidates=max_cands_per_s1)
        val_candidates[s1_id] = set(cands)
        
    # Training pairs: positives + hard negatives from blocking
    X_train_list = []
    y_train_list = []
    
    for s1_id in train_s1_ids:
        prep = s1_prepped[s1_id]
        ctry = prep[7]
        gt_set = gt_train.get(s1_id, set())
        
        # 1. Positives
        for tr_id in gt_set:
            if tr_id in cbi.target_data:
                tr_prep = cbi.target_data[tr_id][:7]
                feat = extract_pair_features(prep[:7], tr_id, tr_prep)
                X_train_list.append(feat)
                y_train_list.append(1)
                
        # 2. Hard Negatives from blocking
        cands = cbi.retrieve_candidates(prep[:7], ctry, max_candidates=10)
        neg_added = 0
        for tr_id in cands:
            if tr_id not in gt_set and tr_id in cbi.target_data:
                tr_prep = cbi.target_data[tr_id][:7]
                feat = extract_pair_features(prep[:7], tr_id, tr_prep)
                X_train_list.append(feat)
                y_train_list.append(0)
                neg_added += 1
                if neg_added >= max(2, len(gt_set) * 2):
                    break
                    
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    
    print(f"Training dataset: {len(X_train):,} pairs (Positives: {np.sum(y_train==1):,}, Negatives: {np.sum(y_train==0):,})")
    print(f"Pair creation took {time.time() - t0:.2f}s.")
    
    return cbi, s1_prepped, val_candidates, X_train, y_train

def run_experiments_and_optimize(
    cbi: CountryBlockingIndex,
    s1_prepped: Dict[str, Tuple],
    val_candidates: Dict[str, Set[str]],
    gt_val: Dict[str, Set[str]],
    X_train: np.ndarray,
    y_train: np.ndarray
):
    """
    Executes Experiments A, B, C, D, E, evaluates macro F0.5 on held-out validation set,
    and returns the best model and optimal threshold.
    """
    os.makedirs("output", exist_ok=True)
    results = []
    
    # Pre-extract validation pair features for fast evaluation across models
    print("\nExtracting validation candidate features...")
    t0 = time.time()
    val_pairs_list = []
    val_pair_keys = []
    
    for s1_id, cands in val_candidates.items():
        s1_p = s1_prepped[s1_id][:7]
        for tr_id in cands:
            if tr_id in cbi.target_data:
                tr_p = cbi.target_data[tr_id][:7]
                feats = extract_pair_features(s1_p, tr_id, tr_p)
                val_pairs_list.append(feats)
                val_pair_keys.append((s1_id, tr_id))
                
    X_val = np.array(val_pairs_list, dtype=np.float32) if val_pairs_list else np.empty((0, len(FEATURE_NAMES)))
    print(f"Extracted features for {len(X_val):,} validation candidate pairs in {time.time() - t0:.2f}s.")
    
    # ----------------------------------------------------
    # EXPERIMENT A: Basic Normalization + Fuzzy Sim + Logistic Regression
    # ----------------------------------------------------
    print("\n--- Running Experiment A: Basic Fuzzy + Logistic Regression ---")
    # Feature indices for basic fuzzy similarity:
    # 0: name_exact, 2: name_fuzz_ratio, 4: name_fuzz_token_sort, 5: name_fuzz_token_set
    # 11: addr_exact, 12: addr_fuzz_ratio, 13: addr_fuzz_token_set
    feat_idx_A = [0, 2, 4, 5, 11, 12, 13]
    scaler_A = StandardScaler()
    X_tr_A = scaler_A.fit_transform(X_train[:, feat_idx_A])
    X_val_A = scaler_A.transform(X_val[:, feat_idx_A]) if len(X_val) > 0 else X_val
    
    model_A = LogisticRegression(max_iter=1000, random_state=42)
    model_A.fit(X_tr_A, y_train)
    val_probs_A = model_A.predict_proba(X_val_A)[:, 1] if len(X_val) > 0 else []
    
    # Evaluate at default threshold 0.50
    preds_A = collections.defaultdict(set)
    for (s1_id, tr_id), p in zip(val_pair_keys, val_probs_A):
        if p >= 0.50:
            preds_A[s1_id].add(tr_id)
    eval_A = evaluate_predictions(gt_val, preds_A, val_candidates)
    eval_A["experiment"] = "Exp A: Basic Fuzzy + Logistic Regression (thresh=0.50)"
    results.append(eval_A)
    print(f"Exp A -> Macro F0.5: {eval_A['macro_f05']:.4f} | Prec: {eval_A['macro_precision']:.4f} | Rec: {eval_A['macro_recall']:.4f}")
    
    # ----------------------------------------------------
    # EXPERIMENT B: Exp A + TF-IDF n-gram / Jaccard / Numeric Features
    # ----------------------------------------------------
    print("\n--- Running Experiment B: Exp A + TF-IDF n-gram & Numeric Features + Logistic Regression ---")
    feat_idx_B = [0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19]
    scaler_B = StandardScaler()
    X_tr_B = scaler_B.fit_transform(X_train[:, feat_idx_B])
    X_val_B = scaler_B.transform(X_val[:, feat_idx_B]) if len(X_val) > 0 else X_val
    
    model_B = LogisticRegression(max_iter=1000, random_state=42)
    model_B.fit(X_tr_B, y_train)
    val_probs_B = model_B.predict_proba(X_val_B)[:, 1] if len(X_val) > 0 else []
    
    preds_B = collections.defaultdict(set)
    for (s1_id, tr_id), p in zip(val_pair_keys, val_probs_B):
        if p >= 0.50:
            preds_B[s1_id].add(tr_id)
    eval_B = evaluate_predictions(gt_val, preds_B, val_candidates)
    eval_B["experiment"] = "Exp B: + TF-IDF/Jaccard/Numeric (thresh=0.50)"
    results.append(eval_B)
    print(f"Exp B -> Macro F0.5: {eval_B['macro_f05']:.4f} | Prec: {eval_B['macro_precision']:.4f} | Rec: {eval_B['macro_recall']:.4f}")
    
    # ----------------------------------------------------
    # EXPERIMENT C: Exp B + Improved Blocking & Full Feature Set (HistGradientBoosting)
    # ----------------------------------------------------
    print("\n--- Running Experiment C: Full Feature Set + HistGradientBoosting ---")
    model_C = HistGradientBoostingClassifier(max_iter=150, min_samples_leaf=20, random_state=42)
    model_C.fit(X_train, y_train)
    val_probs_C = model_C.predict_proba(X_val)[:, 1] if len(X_val) > 0 else []
    
    preds_C = collections.defaultdict(set)
    for (s1_id, tr_id), p in zip(val_pair_keys, val_probs_C):
        if p >= 0.50:
            preds_C[s1_id].add(tr_id)
    eval_C = evaluate_predictions(gt_val, preds_C, val_candidates)
    eval_C["experiment"] = "Exp C: Full Features + HistGradientBoosting (thresh=0.50)"
    results.append(eval_C)
    print(f"Exp C -> Macro F0.5: {eval_C['macro_f05']:.4f} | Prec: {eval_C['macro_precision']:.4f} | Rec: {eval_C['macro_recall']:.4f}")
    
    # ----------------------------------------------------
    # EXPERIMENT D: Hard Negative Mining + LightGBM GBDT
    # ----------------------------------------------------
    print("\n--- Running Experiment D: LightGBM GBDT with Hard Negative Mining ---")
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 200,
        'learning_rate': 0.08,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
    model_D = lgb.LGBMClassifier(**lgb_params)
    model_D.fit(X_train, y_train)
    val_probs_D = model_D.predict_proba(X_val)[:, 1] if len(X_val) > 0 else []
    
    preds_D = collections.defaultdict(set)
    for (s1_id, tr_id), p in zip(val_pair_keys, val_probs_D):
        if p >= 0.50:
            preds_D[s1_id].add(tr_id)
    eval_D = evaluate_predictions(gt_val, preds_D, val_candidates)
    eval_D["experiment"] = "Exp D: LightGBM + Hard Negatives (thresh=0.50)"
    results.append(eval_D)
    print(f"Exp D -> Macro F0.5: {eval_D['macro_f05']:.4f} | Prec: {eval_D['macro_precision']:.4f} | Rec: {eval_D['macro_recall']:.4f}")
    
    # ----------------------------------------------------
    # EXPERIMENT E: Precision-heavy Threshold Optimization for F0.5
    # ----------------------------------------------------
    print("\n--- Running Experiment E: Systematic F0.5 Threshold Optimization ---")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    best_f05 = -1.0
    best_thresh = 0.50
    best_eval = None
    
    for th in thresholds:
        preds_th = collections.defaultdict(set)
        for (s1_id, tr_id), p in zip(val_pair_keys, val_probs_D):
            if p >= th:
                preds_th[s1_id].add(tr_id)
        eval_th = evaluate_predictions(gt_val, preds_th, val_candidates)
        print(f"  Thresh {th:.2f} -> Macro F0.5: {eval_th['macro_f05']:.4f} | Prec: {eval_th['macro_precision']:.4f} | Rec: {eval_th['macro_recall']:.4f}")
        if eval_th['macro_f05'] > best_f05:
            best_f05 = eval_th['macro_f05']
            best_thresh = th
            best_eval = eval_th
            
    best_eval["experiment"] = f"Exp E: LightGBM + Optimal Threshold ({best_thresh:.2f})"
    results.append(best_eval)
    print(f"\nOptimal Threshold: {best_thresh:.2f} (Macro F0.5: {best_f05:.4f})")
    
    # Save best model artifact
    save_obj = {
        "model": model_D,
        "optimal_threshold": best_thresh,
        "feature_names": FEATURE_NAMES,
        "validation_metrics": best_eval
    }
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    
    # Save experiment report to Markdown and CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(EXPERIMENT_REPORT_CSV, index=False)
    
    with open(EXPERIMENT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Systematic Experiment Results & Validation Comparison\n\n")
        f.write("| Experiment | Macro F0.5 | Macro Prec | Macro Rec | Singleton Acc | Candidate Recall | Pred Matches |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in results:
            f.write(f"| **{r['experiment']}** | **{r['macro_f05']:.4f}** | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} | {r['singleton_accuracy']:.4f} | {r['candidate_recall']:.4f} | {r['total_predicted_matches']:,} |\n")
        f.write("\n### Best Configuration Summary\n\n")
        f.write(f"- **Selected Model:** LightGBM GBDT (MIT Licensed)\n")
        f.write(f"- **Optimal Decision Threshold:** `{best_thresh:.2f}`\n")
        f.write(f"- **Validation Macro F0.5:** `{best_f05:.4f}`\n")
        f.write(f"- **Singleton Accuracy:** `{best_eval['singleton_accuracy']*100:.2f}%`\n")
        f.write(f"- **Candidate Recall:** `{best_eval['candidate_recall']*100:.2f}%`\n")
        
    print(f"Experiment reports saved to:\n  - {EXPERIMENT_REPORT_MD}\n  - {EXPERIMENT_REPORT_CSV}")
    return model_D, best_thresh, best_eval

def main():
    print("==================================================")
    print("PHASES 5, 6, 7 & 11: TRAINING & EXPERIMENTS PIPELINE")
    print("==================================================")
    s1_records, target_records, gt_train, gt_val, train_s1_ids, val_s1_ids = load_data_split(
        train_dir="dataset/train",
        n_train_s1=50000,
        n_val_s1=5000,
        seed=42
    )
    
    cbi, s1_prepped, val_candidates, X_train, y_train = build_dataset_and_candidates(
        s1_records, target_records, gt_train, gt_val, train_s1_ids, val_s1_ids, max_cands_per_s1=35
    )
    
    model, best_thresh, best_eval = run_experiments_and_optimize(
        cbi, s1_prepped, val_candidates, gt_val, X_train, y_train
    )
    print("\nTraining and experiment suite finished successfully!")

if __name__ == "__main__":
    main()
