#!/usr/bin/env python3
"""
Phase 2: Reproduce the exact previous validation metrics.
Loads output/best_model.pkl and runs on the exact 5,000 validation slice.
"""
import os
import sys
import pickle
import random
import numpy as np

from business_entity_resolution.src.train import (
    load_data_split,
    build_dataset_and_candidates
)
from business_entity_resolution.src.features import extract_pair_features, FEATURE_NAMES
from business_entity_resolution.src.evaluate import evaluate_predictions
import collections

def main():
    print("Reproducing previous validation results...")
    with open("output/best_model.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    optimal_thresh = saved["optimal_threshold"]
    print(f"Loaded model with threshold = {optimal_thresh}")

    s1_records, target_records, gt_train, gt_val, train_s1_ids, val_s1_ids = load_data_split(
        train_dir="dataset/train",
        n_train_s1=50000,
        n_val_s1=5000,
        seed=42
    )

    cbi, s1_prepped, val_candidates, X_train, y_train = build_dataset_and_candidates(
        s1_records, target_records, gt_train, gt_val, train_s1_ids, val_s1_ids, max_cands_per_s1=35
    )

    print("Scoring validation candidates...")
    preds = collections.defaultdict(set)
    total_val_pairs = 0
    for s1_id in val_s1_ids:
        prep = s1_prepped[s1_id][:7]
        cands = val_candidates[s1_id]
        pairs_feats = []
        cands_list = list(cands)
        for tid in cands_list:
            if tid in cbi.target_data:
                tr_prep = cbi.target_data[tid][:7]
                pairs_feats.append(extract_pair_features(prep, tid, tr_prep))
        if pairs_feats:
            total_val_pairs += len(pairs_feats)
            probs = model.predict_proba(np.array(pairs_feats, dtype=np.float32))[:, 1]
            for tid, p in zip(cands_list, probs):
                if p >= optimal_thresh:
                    preds[s1_id].add(tid)

    res = evaluate_predictions(gt_val, preds, val_candidates)
    print("\n--- REPRODUCED VALIDATION RESULTS ---")
    print(f"Macro F0.5:         {res['macro_f05']:.4f}  (Baseline target: 0.9348)")
    print(f"Macro Precision:    {res['macro_precision']:.4f}  (Baseline target: 0.9945)")
    print(f"Macro Recall:       {res['macro_recall']:.4f}  (Baseline target: 0.8570)")
    print(f"Candidate Recall:   {res['candidate_recall']:.4f}  (Baseline target: 0.8980)")
    print(f"Singleton Accuracy: {res['singleton_accuracy']:.4f}  (Baseline target: 0.9892)")
    print(f"Total Pred Matches: {res['total_predicted_matches']:,}")
    print(f"Total Pred Singletons: {res['total_predicted_singletons']:,}")

if __name__ == "__main__":
    main()
