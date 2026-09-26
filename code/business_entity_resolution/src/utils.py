#!/usr/bin/env python3
"""
Utility functions for ML Challenge 2026: Business Entity Resolution.
Provides streaming TSV reader, file validation helpers, and formatting checks.
"""

import os
import sys
import time
from typing import Generator, List, Set, Dict

def stream_tsv(
    file_path: str,
    skip_header: bool = True
) -> Generator[List[str], None, None]:
    """
    Streams a TSV file line-by-line without loading entire file into memory.
    Yields list of column values.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        if skip_header:
            f.readline()
        for line in f:
            line_clean = line.rstrip("\r\n")
            if line_clean:
                yield line_clean.split("\t")

def count_tsv_rows(file_path: str) -> int:
    """Fast line count for TSV files."""
    cnt = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # skip header
        for _ in f:
            cnt += 1
    return cnt

def load_entity_ids(file_path: str) -> Set[str]:
    """Loads set of entity IDs from the first column of a TSV file."""
    ids = set()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            p = line.split("\t", 1)
            if p and p[0].strip():
                ids.add(p[0].strip())
    return ids

class Timer:
    """Context manager for timing code execution blocks."""
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.description}] Starting...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        print(f"[{self.description}] Completed in {self.elapsed:.2f}s.")
