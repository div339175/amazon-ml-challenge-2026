# Phase 3: Comprehensive Distribution Shift & Gap Analysis

## 1. Feature Distribution Comparison: Train vs Test by Country

### Country: US

| Source | Missing Name % | Missing Addr % | Avg Name Len | Avg Addr Len | PIN Avail % | Num Avail % | Unique Name Ratio | Unique Addr Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train S1 | 0.00% | 0.00% | 22.5 | 34.9 | 11.0% | 99.9% | 0.847 | 0.995 |
| Test S1 | 0.00% | 0.00% | 22.5 | 35.0 | 11.0% | 99.9% | 0.845 | 0.995 |
| Train S2 | 0.00% | 3.65% | 23.5 | 32.8 | 10.3% | 89.7% | 0.912 | 0.990 |
| Test S2 | 0.00% | 2.93% | 24.3 | 32.8 | 10.3% | 91.5% | 0.919 | 0.986 |
| Train S3 | 0.00% | 3.50% | 24.0 | 39.8 | 10.3% | 90.8% | 0.916 | 0.993 |
| Test S3 | 0.00% | 2.89% | 24.6 | 39.9 | 10.6% | 92.4% | 0.920 | 0.988 |

### Country: India

| Source | Missing Name % | Missing Addr % | Avg Name Len | Avg Addr Len | PIN Avail % | Num Avail % | Unique Name Ratio | Unique Addr Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train S1 | 0.00% | 0.00% | 26.4 | 77.7 | 0.3% | 89.2% | 0.703 | 0.994 |
| Test S1 | 0.00% | 0.00% | 26.4 | 77.6 | 0.2% | 89.3% | 0.699 | 0.994 |
| Train S2 | 0.00% | 2.84% | 27.4 | 70.3 | 1.1% | 89.6% | 0.902 | 0.993 |
| Test S2 | 0.00% | 2.30% | 28.2 | 70.3 | 1.1% | 91.5% | 0.916 | 0.994 |
| Train S3 | 0.00% | 3.11% | 27.0 | 61.0 | 1.0% | 88.1% | 0.894 | 0.994 |
| Test S3 | 0.00% | 2.41% | 27.8 | 60.8 | 1.0% | 90.0% | 0.906 | 0.995 |

### Country: France

| Source | Missing Name % | Missing Addr % | Avg Name Len | Avg Addr Len | PIN Avail % | Num Avail % | Unique Name Ratio | Unique Addr Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train S1 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Test S1 | 0.00% | 0.00% | 19.4 | 50.1 | 0.4% | 99.5% | 0.755 | 0.957 |
| Train S2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Test S2 | 0.00% | 3.17% | 21.1 | 40.8 | 0.5% | 90.9% | 0.831 | 0.958 |
| Train S3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Test S3 | 0.00% | 2.95% | 21.1 | 41.2 | 0.5% | 91.2% | 0.833 | 0.960 |

## 2. Ground Truth Match Profile by Source (S1 -> S2 vs S1 -> S3)

- Sample S1 Entities: 100,000
- Matched to Both S2 and S3: 80,427 (80.43%)
- Matched to S2 Only: 6,487 (6.49%)
- Matched to S3 Only: 7,504 (7.50%)
- Singletons (Neither): 5,582 (5.58%)
- Avg S2 Matches per S1: 1.67
- Avg S3 Matches per S1: 1.79

### Breakdown by Country in Ground Truth

| Country | S1 Count | Avg S2 Matches | Avg S3 Matches | Singleton % |
| --- | --- | --- | --- | --- |
| US | 59,936 | 1.67 | 1.78 | 5.65% |
| India | 40,064 | 1.68 | 1.80 | 5.47% |
