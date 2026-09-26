#!/usr/bin/env python3
"""
Phase 3: Deep Gap Analysis & Distribution Comparison
Compares distributions between Train and Test across:
- US, India, France
- S1, S2, S3
- Missing rates, lengths, token counts, postal codes, house numbers
- S1 -> S2 vs S1 -> S3 ground truth properties
"""

import os
import sys
import collections
import numpy as np
import pandas as pd
from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_house_info
)

def analyze_source_file(file_path: str, sample_limit: int = 100000):
    print(f"Analyzing {file_path} (sampling up to {sample_limit:,} rows per country)...")
    stats_by_country = collections.defaultdict(lambda: {
        "count": 0,
        "name_missing": 0,
        "addr_missing": 0,
        "name_lens": [],
        "addr_lens": [],
        "name_tok_counts": [],
        "addr_tok_counts": [],
        "has_pin": 0,
        "has_num": 0,
        "norm_names": set(),
        "norm_addrs": set(),
        "total_names_seen": 0,
        "total_addrs_seen": 0,
    })

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0].strip():
                continue
            eid = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            addr = parts[2].strip() if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else "UNKNOWN"

            st = stats_by_country[ctry]
            if st["count"] >= sample_limit:
                continue

            st["count"] += 1
            if not name:
                st["name_missing"] += 1
            else:
                st["total_names_seen"] += 1
                n_norm, n_stem, n_toks = normalize_business_name(name)
                st["name_lens"].append(len(name))
                st["name_tok_counts"].append(len(n_toks))
                st["norm_names"].add(n_stem)

            if not addr:
                st["addr_missing"] += 1
            else:
                st["total_addrs_seen"] += 1
                a_norm, a_toks, pin, nums = normalize_address(addr)
                st["addr_lens"].append(len(addr))
                st["addr_tok_counts"].append(len(a_toks))
                st["norm_addrs"].add(a_norm)
                if pin:
                    st["has_pin"] += 1
                if nums:
                    st["has_num"] += 1

            if (idx + 1) % 1000000 == 0:
                print(f"  Scanned {idx + 1:,} rows...")

    summary = {}
    for ctry, st in stats_by_country.items():
        n = st["count"]
        if n == 0: continue
        summary[ctry] = {
            "sample_size": n,
            "name_missing_pct": (st["name_missing"] / n) * 100,
            "addr_missing_pct": (st["addr_missing"] / n) * 100,
            "avg_name_len": float(np.mean(st["name_lens"])) if st["name_lens"] else 0.0,
            "avg_addr_len": float(np.mean(st["addr_lens"])) if st["addr_lens"] else 0.0,
            "avg_name_toks": float(np.mean(st["name_tok_counts"])) if st["name_tok_counts"] else 0.0,
            "avg_addr_toks": float(np.mean(st["addr_tok_counts"])) if st["addr_tok_counts"] else 0.0,
            "pin_availability_pct": (st["has_pin"] / n) * 100,
            "house_num_availability_pct": (st["has_num"] / n) * 100,
            "unique_name_ratio": len(st["norm_names"]) / st["total_names_seen"] if st["total_names_seen"] > 0 else 0.0,
            "unique_addr_ratio": len(st["norm_addrs"]) / st["total_addrs_seen"] if st["total_addrs_seen"] > 0 else 0.0,
        }
    return summary

def analyze_ground_truth_by_source(gt_path: str = "dataset/train/train_ground_truth.tsv"):
    print("\nAnalyzing S1 -> S2 vs S1 -> S3 in Ground Truth...")
    s1_countries = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            s1_countries[parts[0].strip()] = parts[3].strip() if len(parts) > 3 else ""
            if idx >= 200000: break

    stats = {
        "total_s1": 0,
        "s2_only": 0,
        "s3_only": 0,
        "both_s2_s3": 0,
        "neither": 0,
        "total_s2_matches": 0,
        "total_s3_matches": 0,
        "s2_match_counts": [],
        "s3_match_counts": [],
        "by_country": collections.defaultdict(lambda: {
            "total_s1": 0, "s2_matches": 0, "s3_matches": 0, "singletons": 0
        })
    }

    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            if s1_id not in s1_countries:
                continue
            ctry = s1_countries[s1_id]
            stats["total_s1"] += 1
            c_st = stats["by_country"][ctry]
            c_st["total_s1"] += 1

            mids = parts[1].strip().split(",") if len(parts) > 1 and parts[1].strip() else []
            s2_m = [m for m in mids if m.startswith("S2-")]
            s3_m = [m for m in mids if m.startswith("S3-")]

            stats["total_s2_matches"] += len(s2_m)
            stats["total_s3_matches"] += len(s3_m)
            c_st["s2_matches"] += len(s2_m)
            c_st["s3_matches"] += len(s3_m)

            if len(s2_m) > 0 and len(s3_m) > 0:
                stats["both_s2_s3"] += 1
            elif len(s2_m) > 0:
                stats["s2_only"] += 1
            elif len(s3_m) > 0:
                stats["s3_only"] += 1
            else:
                stats["neither"] += 1
                c_st["singletons"] += 1

            stats["s2_match_counts"].append(len(s2_m))
            stats["s3_match_counts"].append(len(s3_m))
            if stats["total_s1"] >= 100000:
                break

    return stats

def main():
    os.makedirs("experiments", exist_ok=True)
    report_file = "experiments/phase3_gap_analysis.md"

    files_to_analyze = [
        ("Train S1", "dataset/train/train_source1.tsv"),
        ("Test S1", "dataset/test/test_source1.tsv"),
        ("Train S2", "dataset/train/train_source2.tsv"),
        ("Test S2", "dataset/test/test_source2.tsv"),
        ("Train S3", "dataset/train/train_source3.tsv"),
        ("Test S3", "dataset/test/test_source3.tsv"),
    ]

    all_stats = {}
    for name, path in files_to_analyze:
        all_stats[name] = analyze_source_file(path)

    gt_stats = analyze_ground_truth_by_source()

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Phase 3: Comprehensive Distribution Shift & Gap Analysis\n\n")
        f.write("## 1. Feature Distribution Comparison: Train vs Test by Country\n\n")
        
        for ctry in ["US", "India", "France"]:
            f.write(f"### Country: {ctry}\n\n")
            f.write("| Source | Missing Name % | Missing Addr % | Avg Name Len | Avg Addr Len | PIN Avail % | Num Avail % | Unique Name Ratio | Unique Addr Ratio |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for name, _ in files_to_analyze:
                st = all_stats[name].get(ctry)
                if not st:
                    f.write(f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")
                else:
                    f.write(f"| {name} | {st['name_missing_pct']:.2f}% | {st['addr_missing_pct']:.2f}% | {st['avg_name_len']:.1f} | {st['avg_addr_len']:.1f} | {st['pin_availability_pct']:.1f}% | {st['house_num_availability_pct']:.1f}% | {st['unique_name_ratio']:.3f} | {st['unique_addr_ratio']:.3f} |\n")
            f.write("\n")

        f.write("## 2. Ground Truth Match Profile by Source (S1 -> S2 vs S1 -> S3)\n\n")
        f.write(f"- Sample S1 Entities: {gt_stats['total_s1']:,}\n")
        f.write(f"- Matched to Both S2 and S3: {gt_stats['both_s2_s3']:,} ({(gt_stats['both_s2_s3']/gt_stats['total_s1'])*100:.2f}%)\n")
        f.write(f"- Matched to S2 Only: {gt_stats['s2_only']:,} ({(gt_stats['s2_only']/gt_stats['total_s1'])*100:.2f}%)\n")
        f.write(f"- Matched to S3 Only: {gt_stats['s3_only']:,} ({(gt_stats['s3_only']/gt_stats['total_s1'])*100:.2f}%)\n")
        f.write(f"- Singletons (Neither): {gt_stats['neither']:,} ({(gt_stats['neither']/gt_stats['total_s1'])*100:.2f}%)\n")
        f.write(f"- Avg S2 Matches per S1: {gt_stats['total_s2_matches']/gt_stats['total_s1']:.2f}\n")
        f.write(f"- Avg S3 Matches per S1: {gt_stats['total_s3_matches']/gt_stats['total_s1']:.2f}\n\n")

        f.write("### Breakdown by Country in Ground Truth\n\n")
        f.write("| Country | S1 Count | Avg S2 Matches | Avg S3 Matches | Singleton % |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for ctry, c_st in gt_stats["by_country"].items():
            s1_c = c_st["total_s1"]
            if s1_c == 0: continue
            f.write(f"| {ctry} | {s1_c:,} | {c_st['s2_matches']/s1_c:.2f} | {c_st['s3_matches']/s1_c:.2f} | {(c_st['singletons']/s1_c)*100:.2f}% |\n")

    print(f"\nGap analysis report saved to {report_file}")

if __name__ == "__main__":
    main()
