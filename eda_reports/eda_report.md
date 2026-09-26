# Exploratory Data Analysis (EDA) Report

## 1. Dataset Overview

| Dataset File | Total Rows | Columns | Duplicate IDs | Countries |
| --- | --- | --- | --- | --- |
| `train_source1` | 2,206,821 | `['entity_id', 'business_name', 'business_address', 'country']` | 0 | US: 1,323,633, India: 883,188 |
| `train_source2` | 5,034,616 | `['entity_id', 'business_name', 'business_address', 'country']` | 0 | India: 2,017,799, US: 3,016,817 |
| `train_source3` | 5,285,603 | `['entity_id', 'business_name', 'business_address', 'country']` | 0 | US: 3,170,056, India: 2,115,547 |
| `test_source1` | 1,732,544 | `['entity_id', 'business_name', 'business_address', 'country']` | 0 | US: 663,106, France: 259,452, India: 809,986 |
| `test_source2` | 4,887,273 | `['entity_id', 'business_name', 'business_address', 'country']` | 0 | India: 2,312,565, France: 703,378, US: 1,871,330 |
| `test_source3` | 5,082,316 | `['entity_id', 'business_name', 'business_address', 'country']` | 0 | India: 2,405,000, France: 731,615, US: 1,945,701 |

## 2. Missing Values Analysis

### `train_source1`
- Total rows: 2,206,821
- Missing counts:
  - `entity_id`: 0 (0.00%)
  - `business_name`: 0 (0.00%)
  - `business_address`: 0 (0.00%)
  - `country`: 0 (0.00%)

### `train_source2`
- Total rows: 5,034,616
- Missing counts:
  - `entity_id`: 0 (0.00%)
  - `business_name`: 0 (0.00%)
  - `business_address`: 168,967 (3.36%)
  - `country`: 0 (0.00%)

### `train_source3`
- Total rows: 5,285,603
- Missing counts:
  - `entity_id`: 0 (0.00%)
  - `business_name`: 0 (0.00%)
  - `business_address`: 175,916 (3.33%)
  - `country`: 0 (0.00%)

### `train_ground_truth`
- Total rows: 2,206,821
- Missing counts:

### `test_source1`
- Total rows: 1,732,544
- Missing counts:
  - `entity_id`: 0 (0.00%)
  - `business_name`: 0 (0.00%)
  - `business_address`: 0 (0.00%)
  - `country`: 0 (0.00%)

### `test_source2`
- Total rows: 4,887,273
- Missing counts:
  - `entity_id`: 0 (0.00%)
  - `business_name`: 0 (0.00%)
  - `business_address`: 129,408 (2.65%)
  - `country`: 0 (0.00%)

### `test_source3`
- Total rows: 5,082,316
- Missing counts:
  - `entity_id`: 0 (0.00%)
  - `business_name`: 0 (0.00%)
  - `business_address`: 136,098 (2.68%)
  - `country`: 0 (0.00%)

## 3. Ground Truth Matching Statistics

- Total Source 1 Entities: 2,206,821
- Zero Matches (Singletons): 123,247 (5.58%)
- Exactly One Match: 119,157 (5.40%)
- Multiple Matches: 1,964,417 (89.02%)
- Total S2 Matches: 3,693,619
- Total S3 Matches: 3,944,746
- Max Matches for single S1: 11

### Match Count Distribution

| Matches Count | Number of S1 Entities | Percentage |
| --- | --- | --- |
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |
