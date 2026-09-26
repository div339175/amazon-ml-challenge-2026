#!/usr/bin/env python3
"""
Phase 1: Exploratory Data Analysis (EDA)
Inspects all training and test files, checks data integrity, missing values,
country distributions, match statistics, and saves a comprehensive report.
"""

import os
import sys
import collections
import json
from typing import Dict, Any, List

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
REPORT_PATH = "eda_reports/eda_summary.json"
REPORT_MD_PATH = "eda_reports/eda_report.md"

def inspect_tsv_file(file_path: str, is_ground_truth: bool = False) -> Dict[str, Any]:
    print(f"Inspecting {file_path}...")
    stats = {
        "file_path": file_path,
        "row_count": 0,
        "columns": [],
        "missing_per_col": {},
        "unique_countries": collections.Counter(),
        "duplicate_ids": 0,
        "sample_names": [],
        "sample_addresses": [],
        "entity_prefix_counts": collections.Counter(),
    }
    
    seen_ids = set()
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        cols = header_line.split("\t")
        stats["columns"] = cols
        stats["missing_per_col"] = {c: 0 for c in cols}
        
        for idx, line in enumerate(f):
            stats["row_count"] += 1
            parts = line.rstrip("\r\n").split("\t")
            # Pad parts if line was truncated
            if len(parts) < len(cols):
                parts += [""] * (len(cols) - len(parts))
            elif len(parts) > len(cols):
                # Extra tabs? Join excess into last column
                parts = parts[:len(cols)-1] + ["\t".join(parts[len(cols)-1:])]
            
            # Entity ID
            ent_id = parts[0].strip()
            if ent_id in seen_ids:
                stats["duplicate_ids"] += 1
            else:
                seen_ids.add(ent_id)
                
            prefix = ent_id.split("-")[0] if "-" in ent_id else "UNKNOWN"
            stats["entity_prefix_counts"][prefix] += 1
            
            for c_idx, c_name in enumerate(cols):
                val = parts[c_idx].strip()
                if not val:
                    stats["missing_per_col"][c_name] += 1
            
            if not is_ground_truth:
                # Expecting entity_id, business_name, business_address, country
                if "country" in cols:
                    c_idx = cols.index("country")
                    c_val = parts[c_idx].strip()
                    stats["unique_countries"][c_val] += 1
                    
                if len(stats["sample_names"]) < 10 and "business_name" in cols:
                    name_idx = cols.index("business_name")
                    if parts[name_idx].strip():
                        stats["sample_names"].append(parts[name_idx].strip())
                        
                if len(stats["sample_addresses"]) < 10 and "business_address" in cols:
                    addr_idx = cols.index("business_address")
                    if parts[addr_idx].strip():
                        stats["sample_addresses"].append(parts[addr_idx].strip())
                        
            if (idx + 1) % 1_000_000 == 0:
                print(f"  Processed {idx + 1:,} rows...")
                
    return stats

def inspect_ground_truth(gt_path: str) -> Dict[str, Any]:
    print(f"Inspecting Ground Truth {gt_path}...")
    gt_stats = {
        "total_source1_entities": 0,
        "zero_matches": 0,
        "one_match": 0,
        "multiple_matches": 0,
        "match_count_distribution": collections.Counter(),
        "s2_matches_total": 0,
        "s3_matches_total": 0,
        "duplicate_s1_ids": 0,
        "max_matches": 0,
        "sample_ground_truth": []
    }
    
    seen_s1 = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for idx, line in enumerate(f):
            gt_stats["total_source1_entities"] += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id in seen_s1:
                gt_stats["duplicate_s1_ids"] += 1
            seen_s1.add(s1_id)
            
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            if not matched_str:
                matched_ids = []
            else:
                matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                
            num_matches = len(matched_ids)
            gt_stats["match_count_distribution"][num_matches] += 1
            if num_matches > gt_stats["max_matches"]:
                gt_stats["max_matches"] = num_matches
                
            if num_matches == 0:
                gt_stats["zero_matches"] += 1
            elif num_matches == 1:
                gt_stats["one_match"] += 1
            else:
                gt_stats["multiple_matches"] += 1
                
            for m in matched_ids:
                if m.startswith("S2-"):
                    gt_stats["s2_matches_total"] += 1
                elif m.startswith("S3-"):
                    gt_stats["s3_matches_total"] += 1
                    
            if len(gt_stats["sample_ground_truth"]) < 5 and matched_ids:
                gt_stats["sample_ground_truth"].append({
                    "s1_id": s1_id,
                    "matched": matched_ids
                })
                
            if (idx + 1) % 1_000_000 == 0:
                print(f"  Processed {idx + 1:,} GT rows...")
                
    return gt_stats

def main():
    os.makedirs("eda_reports", exist_ok=True)
    all_reports = {}
    
    source_files = [
        ("train_source1", os.path.join(TRAIN_DIR, "train_source1.tsv"), False),
        ("train_source2", os.path.join(TRAIN_DIR, "train_source2.tsv"), False),
        ("train_source3", os.path.join(TRAIN_DIR, "train_source3.tsv"), False),
        ("train_ground_truth", os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), True),
        ("test_source1", os.path.join(TEST_DIR, "test_source1.tsv"), False),
        ("test_source2", os.path.join(TEST_DIR, "test_source2.tsv"), False),
        ("test_source3", os.path.join(TEST_DIR, "test_source3.tsv"), False),
    ]
    
    for key, path, is_gt in source_files:
        if not os.path.exists(path):
            print(f"Warning: {path} not found!")
            continue
        if is_gt:
            res = inspect_ground_truth(path)
        else:
            res = inspect_tsv_file(path)
        all_reports[key] = res

    # Write JSON summary
    # Convert counters to dicts for serialization
    serializable = {}
    for k, v in all_reports.items():
        s_v = dict(v)
        if "unique_countries" in s_v:
            s_v["unique_countries"] = dict(s_v["unique_countries"])
        if "entity_prefix_counts" in s_v:
            s_v["entity_prefix_counts"] = dict(s_v["entity_prefix_counts"])
        if "match_count_distribution" in s_v:
            s_v["match_count_distribution"] = {str(m): c for m, c in sorted(s_v["match_count_distribution"].items())}
        serializable[k] = s_v

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
        
    # Generate Markdown Report
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis (EDA) Report\n\n")
        f.write("## 1. Dataset Overview\n\n")
        f.write("| Dataset File | Total Rows | Columns | Duplicate IDs | Countries |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for key, info in serializable.items():
            if key == "train_ground_truth":
                continue
            c_str = ", ".join([f"{c}: {n:,}" for c, n in info.get("unique_countries", {}).items()])
            f.write(f"| `{key}` | {info['row_count']:,} | `{info['columns']}` | {info['duplicate_ids']} | {c_str} |\n")
            
        f.write("\n## 2. Missing Values Analysis\n\n")
        for key, info in serializable.items():
            f.write(f"### `{key}`\n")
            f.write(f"- Total rows: {info.get('row_count', info.get('total_source1_entities', 0)):,}\n")
            f.write("- Missing counts:\n")
            for c, n in info.get("missing_per_col", {}).items():
                pct = (n / info['row_count']) * 100 if info['row_count'] > 0 else 0
                f.write(f"  - `{c}`: {n:,} ({pct:.2f}%)\n")
            f.write("\n")
            
        if "train_ground_truth" in serializable:
            gt = serializable["train_ground_truth"]
            f.write("## 3. Ground Truth Matching Statistics\n\n")
            f.write(f"- Total Source 1 Entities: {gt['total_source1_entities']:,}\n")
            f.write(f"- Zero Matches (Singletons): {gt['zero_matches']:,} ({(gt['zero_matches']/gt['total_source1_entities'])*100:.2f}%)\n")
            f.write(f"- Exactly One Match: {gt['one_match']:,} ({(gt['one_match']/gt['total_source1_entities'])*100:.2f}%)\n")
            f.write(f"- Multiple Matches: {gt['multiple_matches']:,} ({(gt['multiple_matches']/gt['total_source1_entities'])*100:.2f}%)\n")
            f.write(f"- Total S2 Matches: {gt['s2_matches_total']:,}\n")
            f.write(f"- Total S3 Matches: {gt['s3_matches_total']:,}\n")
            f.write(f"- Max Matches for single S1: {gt['max_matches']}\n\n")
            f.write("### Match Count Distribution\n\n")
            f.write("| Matches Count | Number of S1 Entities | Percentage |\n")
            f.write("| --- | --- | --- |\n")
            for cnt_str, freq in sorted(gt["match_count_distribution"].items(), key=lambda x: int(x[0])):
                pct = (freq / gt['total_source1_entities']) * 100
                f.write(f"| {cnt_str} | {freq:,} | {pct:.2f}% |\n")
                
    print(f"\nEDA completed successfully. Reports saved to:\n  - {REPORT_PATH}\n  - {REPORT_MD_PATH}")

if __name__ == "__main__":
    main()
