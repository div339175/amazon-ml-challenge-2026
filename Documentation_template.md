# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Antigravity AI  
**Team Members:** Divakar Maurya  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We developed an end-to-end, high-recall, precision-optimized entity resolution pipeline designed to resolve noisy business records across three disparate data sources (`Source 1`, `Source 2`, `Source 3`). Our solution combines country-partitioned multi-key inverted index blocking with frequency-capped posting lists, phonetic script transliteration (`unidecode`), a 27-dimensional feature engineering pipeline (lexical, phonetic, edit-distance, n-gram TF-IDF, and postal/spatial signals), and a LightGBM Gradient Boosted Decision Tree optimized specifically for the precision-heavy macro $F_{0.5}$ metric with hard negative mining. On our held-out validation benchmark across diverse geographical regions, the system achieves a macro $F_{0.5}$ score of **0.9348** with **99.45% precision**, **98.92% singleton accuracy**, and **89.80% candidate recall** while reducing the pairwise comparison space by over **99.9998%**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) of the 26.4 million records revealed several critical data characteristics and noise patterns:
1. **Source Disparity:** Source 1 is a clean, deduplicated reference set (2.21M train records, 1.73M test records), whereas Source 2 (~5M records) and Source 3 (~5.3M records) contain noisy, unlinked representations of businesses.
2. **Cardinality & Match Distribution:** 89.02% of Source 1 entities match multiple entities in Source 2 and Source 3 (up to 11 matches, average 3.46 matches per entity), 5.40% match exactly one entity, and 5.58% are true singletons with zero matches. Accurate prediction of singletons is critical because a correct singleton prediction receives 1.0 on entity $F_{0.5}$, whereas a false merge receives 0.0.
3. **Open-Set Geography & Country Separation:** The training data contains records from `US` and `India`, whereas the test set introduces `France` as an unobserved country. Crucially, empirical auditing of all 7,638,365 ground-truth matching pairs in the training data confirmed that **100.000% of matches occur within the exact same country** (zero country cross-matches).
4. **Multilingual and Multi-Script Noise:** In India, business names and addresses frequently appear in Indian scripts (Devanagari: e.g. `रेड वेंचर्स`, Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`, Kannada: `ಕರ್ನಾಟಕ`) or phonetic Roman transliterations. In France, French corporate forms (`SARL`, `SAS`, `SA`) and accented characters (`École`, `Bordeaux`) are prevalent. In the US, extreme abbreviations (`Corp` vs `Corporation`, `Ave` vs `Avenue`) and domain-name conversions (`maurewilliamscolombier.com` vs `Maure Williams Colombier Inc`) are ubiquitous.
5. **Address Truncation & Numbering:** Approximately 3.3% of Source 2/3 records have missing addresses. House/building numbers often have leading zeros (`01604` vs `1604`) or alphanumeric suffixes (`A702` vs `702`).

### 2.2 Solution Strategy
**Approach Type:** Multi-Stage Blocking + Feature Engineering + Gradient Boosted Decision Tree Classifier with Hard Negative Mining and Precision-Heavy Threshold Optimization.

**Core Technical Innovations:**
1. **Dynamic Open-Set Country Partitioning:** By partitioning candidate generation strictly by country without hardcoding valid countries, the pairwise comparison space is reduced by ~70% immediately, guaranteeing 100% geographic consistency and seamlessly handling France and any open-set country.
2. **Phonetic Script Normalization:** Applying Unicode NFKD normalization and phonetic transliteration via `unidecode` directly bridges the gap between Indic/Devanagari/Tamil text, accented French text, and standard ASCII business names.
3. **Compound Spatial & Concatenated Blocking:** Inverted indexing uses both lexical stems, concatenated names (handling web domains like `.com` and space drops), and compound spatial keys (building number + locality token / postal code) to capture entities even under severe typos or alias variations.
4. **Hard Negative Mining:** Generating training negatives from the actual candidate retrieval stage (such as businesses sharing common brand names or locations but belonging to different entities) trains the model to distinguish true entity links from realistic distractors.
5. **Precision-Heavy Decision Thresholding:** Because $F_{0.5}$ weights precision twice as heavily as recall, optimizing the classification threshold on validation data ($\tau = 0.90$) dramatically reduces false positive merges, elevating precision to 99.45% and singleton accuracy to 98.92%.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To reduce the $1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ theoretical comparison space down to a manageable, highly relevant candidate set, we constructed a country-partitioned multi-key inverted index:
- **`nx:<stem_name>`:** Normalized core business name with standardized legal suffixes stripped (e.g. `nx:orelee barbershop`). Weight: 12.0.
- **`nc:<concat_name>`:** First 16 characters of concatenated name tokens, bridging domain names (`burgersolution.com`), hyphenations, and spacing discrepancies. Weight: 10.0.
- **`nt:<token>`:** Distinctive non-generic business tokens ($\text{length} \ge 3$, excluding stopwords and legal suffixes). Weight: 3.0 to 7.0 based on length.
- **`np:<prefix>`:** 4-character prefix of the primary token for typo resilience. Weight: 3.0.
- **`anw:<number>_<locality>`:** Compound building number + significant locality/city token (e.g. `anw:1795_westchester`). Weight: 7.0.
- **`anp:<number>_<postal_code>`:** Compound building number + extracted postal/PIN code (e.g. `anp:1795_27262`). Weight: 8.0.
- **`amw:<word1>_<word2>`:** Compound multi-word address key for records with missing or short business names. Weight: 6.0.

### 3.2 Scaling & Noise Reduction
To prevent high-frequency generic terms (e.g. `consulting`, `enterprises`, `hospital`, `delhi`, `paris`) from blowing up candidate pools, posting lists are capped at a maximum of 250 records. Furthermore, an inverse document frequency discount factor $\frac{1}{1 + 0.05 \times \text{len}(\text{posting\_list})}$ dynamically downweights common tokens in favor of rare, distinctive keys.

### 3.3 Candidate Yield & Recall
- **Candidate pairs generated per Source 1 entity:** Average ~15 to 25 candidates per entity (maximum 30 candidates).
- **Candidate Recall ceiling:** Reaches **89.8% to 99.7%** on true positive matches across validation tests.
- **Reduction Ratio:** Over **99.9998%** reduction in total pair evaluations.

---

## 4. Matching Model

### 4.1 Feature Engineering (27 Features)
For every candidate pair $(S_1, S_{2/3})$, a 27-dimensional feature vector is computed:

1. **Name Matching Features (11 features):**
   - Exact normalized name equality (binary)
   - Exact stem name equality (binary)
   - RapidFuzz ratio ($\in [0, 1]$)
   - RapidFuzz partial ratio ($\in [0, 1]$)
   - RapidFuzz token sort ratio ($\in [0, 1]$)
   - RapidFuzz token set ratio ($\in [0, 1]$)
   - Normalized Levenshtein similarity ($\in [0, 1]$)
   - Character 3-gram cosine similarity ($\in [0, 1]$)
   - Token Jaccard similarity ($\in [0, 1]$)
   - Absolute character length difference
   - Absolute token count difference

2. **Address Matching Features (11 features):**
   - Exact normalized address equality (binary)
   - RapidFuzz address ratio ($\in [0, 1]$)
   - RapidFuzz address token set ratio ($\in [0, 1]$)
   - Normalized Levenshtein address similarity ($\in [0, 1]$)
   - Character 3-gram address cosine similarity ($\in [0, 1]$)
   - Address token Jaccard similarity ($\in [0, 1]$)
   - Postal/PIN code match score ($+1.0$ if matching, $-1.0$ if both present and differing, $0.0$ if missing)
   - House/building numeric token Jaccard overlap ($\in [0, 1]$)
   - House/building numeric token count overlap
   - Absolute address character length difference
   - Target address missing indicator (binary)

3. **Interaction & Metadata Features (5 features):**
   - Source indicator ($1.0$ if Source 2, $0.0$ if Source 3)
   - Combined weighted similarity: $0.6 \times \text{name\_set\_ratio} + 0.4 \times \text{addr\_set\_ratio}$
   - Strong exact name & address match indicator (binary)
   - Strong name & building number match indicator (binary)
   - Strong name & postal/PIN code match indicator (binary)

### 4.2 Model Type & Architecture
- **Model:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`)
- **License:** MIT License (strictly complies with the challenge requirement)
- **Parameters:** 200 boosting trees, maximum depth 31 leaves, learning rate 0.08, binary cross-entropy loss. Total parameter count is $\approx 150,000$ parameters, far below the 8 Billion parameter ceiling.
- **Inference Speed:** $> 900,000$ pair predictions per second on CPU.

### 4.3 Training & Validation Methodology
- **Leak-Free Entity Splitting:** The dataset was split strictly by `source1_entity_id`. All records associated with a given Source 1 entity are confined exclusively to either the training set (50,000 S1 entities, ~380k pairs) or validation set (5,000 S1 entities, ~92k candidate pairs).
- **Hard Negative Mining:** For each training entity, positive pairs from ground truth were paired with hard negatives retrieved by the blocking index (same surname/brand with differing address; same building with differing business).
- **Threshold Selection:** Grid search was performed across thresholds from 0.50 to 0.95 with step 0.05 using the exact macro $F_{0.5}$ metric. The optimal threshold was identified at $\tau = 0.90$.

---

## 5. Results & Error Analysis

### 5.1 Validation Results Comparison

| Experiment | Pipeline Configuration | Macro F0.5 | Macro Precision | Macro Recall | Singleton Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp A** | Basic Normalization + Fuzzy Sim + Logistic Regression ($\tau=0.50$) | 0.9108 | 0.9635 | 0.8527 | 92.09% |
| **Exp B** | Exp A + TF-IDF n-gram & Numeric Features + Logistic Regression ($\tau=0.50$) | 0.9127 | 0.9527 | 0.8828 | 86.33% |
| **Exp C** | Full Features + HistGradientBoosting ($\tau=0.50$) | 0.9214 | 0.9590 | 0.8950 | 86.33% |
| **Exp D** | LightGBM GBDT + Hard Negative Mining ($\tau=0.50$) | 0.9231 | 0.9608 | 0.8952 | 86.69% |
| **Exp E** | **LightGBM GBDT + Optimal Threshold ($\tau=0.90$)** | **0.9348** | **0.9945** | **0.8570** | **98.92%** |

### 5.2 Error Analysis
1. **False Positives (Wrong Merges):**
   - Prior to threshold optimization ($\tau=0.50$), false positives primarily stemmed from retail chains or franchises sharing identical corporate brand names located on the same commercial boulevard or district without explicit unit numbers.
   - At $\tau=0.90$, false positives dropped precipitously, pushing macro precision to **99.45%** and singleton accuracy to **98.92%**.
2. **False Negatives (Missed Matches):**
   - Rare edge cases where an entity had both a drastically different transliterated brand abbreviation and a completely blank address field in Source 2 or Source 3.
   - Businesses where the street address underwent a complete administrative renaming between the reference data collection and the source data ingestion.

---

## 6. Conclusion
By pairing country-partitioned multi-key blocking with phonetic normalization, rich pairwise feature engineering, and a precision-tuned LightGBM classifier with hard negative mining, our solution achieves state-of-the-art entity resolution performance. Operating under strict constraints (no external lookups, open-set countries, memory $\le 3$ GB RAM, MIT licensed model $\ll 8\text{B}$ parameters), the pipeline achieves a validation macro $F_{0.5}$ score of **0.9348** while strictly adhering to all formatting rules.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multi-country unicode, phonetic transliteration, and address extractors.
- `src/blocking.py`: Country-aware inverted index and candidate generation system.
- `src/features.py`: 27-dimensional pairwise feature extraction pipeline.
- `src/train.py`: Leak-free training, hard negative mining, model training, and experiment comparison.
- `src/evaluate.py`: Official macro $F_{0.5}$ evaluation engine.
- `src/predict.py`: Scalable test inference producing `matching_results.tsv` and `candidate_pairs.tsv`.
- `src/utils.py`: Streaming TSV utilities and timer context managers.
- `README.md`: Detailed reproduction guide and environment documentation.
- `requirements.txt`: Pinned dependencies.

**Reproduction Command:**
```bash
python3 business_entity_resolution/src/predict.py
```

### B. Additional Results
Threshold tuning progression on held-out validation data ($N=5,000$ entities, 92,406 candidate pairs):

| Decision Threshold $\tau$ | Macro F0.5 | Macro Precision | Macro Recall | Singleton Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 0.50 | 0.9231 | 0.9608 | 0.8952 | 86.69% |
| 0.60 | 0.9271 | 0.9680 | 0.8907 | 88.49% |
| 0.70 | 0.9304 | 0.9739 | 0.8868 | 91.01% |
| 0.80 | 0.9333 | 0.9845 | 0.8732 | 94.96% |
| 0.85 | 0.9341 | 0.9885 | 0.8673 | 96.76% |
| **0.90** | **0.9348** | **0.9945** | **0.8570** | **98.92%** |
| 0.95 | 0.9316 | 0.9971 | 0.8458 | 99.64% |
