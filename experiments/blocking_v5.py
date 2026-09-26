#!/usr/bin/env python3
"""
V5 High-Performance Sparse Vector & Spatial Blocking Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Combines:
1. Lexical Word-Level TF-IDF (unigrams + bigrams, sublinear TF)
2. Subword Character n-gram TF-IDF (char_wb 3-4 grams, sublinear TF)
3. Deterministic Spatial & Exact Hash Tables (PIN+HouseNum, Exact Name, Exact Addr)
4. Vectorized SciPy CSR sparse matrix multiplication with BLAS thread control
5. Multi-channel union & deduplication capped at ~35 candidates per S1
"""

import os
import sys

# Ensure workspace root is in sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Prevent BLAS/OpenMP oversubscription across worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import math
import collections
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from business_entity_resolution.src.preprocessing import (
    normalize_business_name,
    normalize_address,
)

class CountrySparseBlockingIndexV5:
    """
    Country-isolated, memory-efficient sparse TF-IDF and spatial retrieval engine.
    """
    def __init__(
        self,
        country: str,
        word_max_features: int = 120000,
        char_max_features: int = 120000,
        max_posting_size: int = 150
    ):
        self.country = country
        self.max_posting_size = max_posting_size

        # Vectorizers
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,
            max_features=word_max_features,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            min_df=3,
            max_features=char_max_features,
            sublinear_tf=True,
            dtype=np.float32
        )

        # Target ID registry
        self.target_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        # Prepped target tuples: (norm_n, stem_n, toks_n, norm_a, toks_a, pin, nums)
        self.target_prepped: Dict[str, Tuple] = {}

        # Spatial and exact hash tables
        self.exact_name_index: Dict[str, List[int]] = collections.defaultdict(list)
        self.exact_addr_index: Dict[str, List[int]] = collections.defaultdict(list)
        self.pin_num_index: Dict[str, List[int]] = collections.defaultdict(list)

        # Sparse target matrices
        self.T_word: Optional[sp.csr_matrix] = None
        self.T_char: Optional[sp.csr_matrix] = None
        self.is_indexed: bool = False

    def build_index(self, target_records_list: List[Tuple[str, str, str]]):
        """
        Builds sparse matrices and hash tables from target records.
        target_records_list: list of (entity_id, business_name, business_address)
        """
        t0 = time.time()
        self.target_ids = []
        self.id_to_idx = {}
        self.target_prepped = {}
        self.exact_name_index.clear()
        self.exact_addr_index.clear()
        self.pin_num_index.clear()

        corpus_word_texts = []
        corpus_char_texts = []

        for idx, (eid, raw_name, raw_addr) in enumerate(target_records_list):
            self.target_ids.append(eid)
            self.id_to_idx[eid] = idx

            nn, sn, tn = normalize_business_name(raw_name)
            na, ta, pin, nums = normalize_address(raw_addr)
            prepped = (nn, sn, tn, na, ta, pin, nums)
            self.target_prepped[eid] = prepped

            # Corpus text strings
            combined_text = f"{nn} {na}" if na else nn
            corpus_word_texts.append(combined_text)
            corpus_char_texts.append(nn)

            # Spatial and exact hash keys
            if nn and len(nn) >= 3 and len(self.exact_name_index[nn]) < self.max_posting_size:
                self.exact_name_index[nn].append(idx)
            if na and len(na) >= 5 and len(self.exact_addr_index[na]) < self.max_posting_size:
                self.exact_addr_index[na].append(idx)
            if pin and nums:
                for n in nums:
                    k = f"{pin}_{n}"
                    if len(self.pin_num_index[k]) < self.max_posting_size:
                        self.pin_num_index[k].append(idx)

        # Fit and transform sparse matrices
        if corpus_word_texts:
            self.T_word = self.word_vectorizer.fit_transform(corpus_word_texts).astype(np.float32)
            self.T_char = self.char_vectorizer.fit_transform(corpus_char_texts).astype(np.float32)
        else:
            self.T_word = sp.csr_matrix((0, 0), dtype=np.float32)
            self.T_char = sp.csr_matrix((0, 0), dtype=np.float32)

        self.is_indexed = True

    def retrieve_batch_candidates(
        self,
        prepped_queries: List[Tuple],
        top_k: int = 35,
        word_cands_k: int = 25,
        char_cands_k: int = 20
    ) -> List[List[str]]:
        """
        Vectorized batch retrieval using sparse matrix dot-product and hash tables.
        prepped_queries: list of (nn, sn, tn, na, ta, pin, nums, ...)
        Returns: list of candidate ID lists, one per query.
        """
        if not self.is_indexed or len(self.target_ids) == 0:
            return [[] for _ in prepped_queries]

        num_queries = len(prepped_queries)
        q_word_texts = []
        q_char_texts = []

        for q in prepped_queries:
            nn = q[0]
            na = q[3]
            combined = f"{nn} {na}" if na else nn
            q_word_texts.append(combined)
            q_char_texts.append(nn)

        # Transform query batch to sparse CSR matrices
        Q_word = self.word_vectorizer.transform(q_word_texts).astype(np.float32)
        Q_char = self.char_vectorizer.transform(q_char_texts).astype(np.float32)

        # Matrix dot product (cosine similarity)
        S_word = Q_word.dot(self.T_word.T)
        S_char = Q_char.dot(self.T_char.T)

        batch_results = []
        n_targets = len(self.target_ids)

        for i in range(num_queries):
            q = prepped_queries[i]
            nn = q[0]
            na = q[3]
            pin = q[5]
            nums = q[6]

            candidate_scores = collections.defaultdict(float)

            # 1. Word TF-IDF top candidates
            row_w = S_word.getrow(i)
            if row_w.nnz > 0:
                data = row_w.data
                indices = row_w.indices
                if len(data) > word_cands_k:
                    part_idx = np.argpartition(data, -word_cands_k)[-word_cands_k:]
                    for p in part_idx:
                        candidate_scores[indices[p]] += float(data[p]) * 10.0
                else:
                    for d, idx in zip(data, indices):
                        candidate_scores[idx] += float(d) * 10.0

            # 2. Char TF-IDF top candidates
            row_c = S_char.getrow(i)
            if row_c.nnz > 0:
                data = row_c.data
                indices = row_c.indices
                if len(data) > char_cands_k:
                    part_idx = np.argpartition(data, -char_cands_k)[-char_cands_k:]
                    for p in part_idx:
                        candidate_scores[indices[p]] += float(data[p]) * 8.0
                else:
                    for d, idx in zip(data, indices):
                        candidate_scores[idx] += float(d) * 8.0

            # 3. Exact Name hash match (bonus 30.0)
            if nn and nn in self.exact_name_index:
                for idx in self.exact_name_index[nn]:
                    candidate_scores[idx] += 30.0

            # 4. Exact Address hash match (bonus 25.0)
            if na and na in self.exact_addr_index:
                for idx in self.exact_addr_index[na]:
                    candidate_scores[idx] += 25.0

            # 5. PIN + House Number hash match (bonus 20.0)
            if pin and nums:
                for n in nums:
                    k = f"{pin}_{n}"
                    if k in self.pin_num_index:
                        for idx in self.pin_num_index[k]:
                            candidate_scores[idx] += 20.0

            # Rank and cap at top_k
            if candidate_scores:
                sorted_indices = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)[:top_k]
                cands_eids = [self.target_ids[idx] for idx in sorted_indices]
            else:
                cands_eids = []

            batch_results.append(cands_eids)

        return batch_results
