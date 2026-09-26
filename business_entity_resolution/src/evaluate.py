#!/usr/bin/env python3
"""
Phase 10: Validation and Metric Evaluation
Implements the exact macro-averaged F_0.5 evaluation metric as defined by the Amazon ML Challenge 2026.
Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Singletons (S1 entities with 0 matches) receive:
    1.0 if correctly predicted as empty (0 matches)
    0.0 if any match is predicted.
"""

from typing import Dict, Set, List, Any
import numpy as np

def compute_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (f05, precision, recall) for a single Source 1 entity.
    """
    len_gt = len(gt_ids)
    len_pred = len(pred_ids)
    
    # Singleton handling
    if len_gt == 0:
        if len_pred == 0:
            return 1.0, 1.0, 1.0  # Correct singleton prediction
        else:
            return 0.0, 0.0, 1.0  # False merge on singleton
            
    # Non-singleton entity
    if len_pred == 0:
        return 0.0, 1.0, 0.0  # Missed all matches
        
    tp = len(gt_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0
        
    precision = tp / len_pred
    recall = tp / len_gt
    
    denom = (0.25 * precision + recall)
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return f05, precision, recall

def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    candidate_pairs: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth across all Source 1 entities.
    Returns comprehensive metrics dictionary.
    """
    entity_scores = []
    entity_precisions = []
    entity_recalls = []
    
    singleton_total = 0
    singleton_correct = 0
    
    total_tp = 0
    total_pred = 0
    total_gt = 0
    
    s2_match_cnt = 0
    s3_match_cnt = 0
    pred_singletons = 0
    
    # Candidate recall tracking if candidate_pairs is provided
    cand_hit = 0
    cand_total = 0
    cand_counts = []
    
    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        
        f05, prec, rec = compute_entity_f05(gt_set, pred_set)
        entity_scores.append(f05)
        entity_precisions.append(prec)
        entity_recalls.append(rec)
        
        if len(gt_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            total_gt += len(gt_set)
            
        if len(pred_set) == 0:
            pred_singletons += 1
        else:
            total_pred += len(pred_set)
            tp = len(gt_set & pred_set)
            total_tp += tp
            for mid in pred_set:
                if mid.startswith("S2-"):
                    s2_match_cnt += 1
                elif mid.startswith("S3-"):
                    s3_match_cnt += 1
                    
        if candidate_pairs is not None:
            cands = candidate_pairs.get(s1_id, set())
            cand_counts.append(len(cands))
            if len(gt_set) > 0:
                cand_total += len(gt_set)
                cand_hit += len(gt_set & cands)
                
    macro_f05 = float(np.mean(entity_scores)) if entity_scores else 0.0
    macro_precision = float(np.mean(entity_precisions)) if entity_precisions else 0.0
    macro_recall = float(np.mean(entity_recalls)) if entity_recalls else 0.0
    
    micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
    micro_recall = total_tp / total_gt if total_gt > 0 else 0.0
    micro_denom = (0.25 * micro_precision + micro_recall)
    micro_f05 = (1.25 * micro_precision * micro_recall / micro_denom) if micro_denom > 0 else 0.0
    
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    cand_recall = cand_hit / cand_total if cand_total > 0 else 1.0
    avg_cands = float(np.mean(cand_counts)) if cand_counts else 0.0
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f05": micro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "singleton_accuracy": singleton_acc,
        "total_s1_evaluated": len(ground_truth),
        "total_predicted_matches": total_pred,
        "total_predicted_singletons": pred_singletons,
        "s2_match_count": s2_match_cnt,
        "s3_match_count": s3_match_cnt,
        "candidate_recall": cand_recall,
        "average_candidates_per_s1": avg_cands,
    }

if __name__ == "__main__":
    # Test with official README example:
    # S1-00001: pred = [S2-00047, S2-00193, S3-00812], gt = [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F0.5 = 0.714
    gt = {"S1-00001": {"S2-00047", "S3-00812"}, "S1-00002": set()}
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}, "S1-00002": set()}
    
    res = evaluate_predictions(gt, pred)
    print("Test Validation Metric Results:")
    for k, v in res.items():
        print(f"  {k:28s}: {v}")
