# Phase 4: Blocking Audit & Recall Bottleneck Report

## 1. Recall Breakdown by Group

| Group | Ground Truth Matches | Broad Recall | Top-25 Recall | Top-35 Recall | Model Recall (thresh=0.90) |
| --- | --- | --- | --- | --- | --- |
| **overall** | 17,165 | 94.01% | 90.85% | 91.29% | 86.26% |
| **US** | 8,481 | 97.39% | 94.72% | 95.14% | 90.40% |
| **India** | 8,684 | 90.71% | 87.07% | 87.53% | 82.22% |
| **S1_S2** | 8,246 | 93.31% | 90.52% | 90.89% | 86.03% |
| **S1_S3** | 8,919 | 94.66% | 91.15% | 91.66% | 86.48% |

- **Average candidates per S1 (Top-35):** 34.00
- **Median candidates per S1:** 35.0
- **Percentage of S1 with 0 candidates:** 0.02%

## 2. Qualitative Error Examples

### A. Missed in Broad Blocking (Never Generated as Candidate)

- **S1:** `S1-993527714` | Name: `George Saul Inc` | Addr: `Unit UNIT 367, 1400 Great Wolf Drive, WI, Village Of Lake Delton`
  **TR:** `S3-250788566` | Name: `Úmbrayuma` | Addr: `1400 Great Wolf Dr, Unit UNIT 367, Village Of Lake Delton, Wisconsin`

- **S1:** `S1-720468760` | Name: `B & G Exchange` | Addr: `354 Claremont Road, Kingsport, TN`
  **TR:** `S2-703072292` | Name: `B &` | Addr: `365 CLAREMONT ROAD, KINGSPORT, TN`

- **S1:** `S1-753180274` | Name: `Helios LLC` | Addr: `306 Wrights Lane, Bldg BRIAN GRIFFITH DENTISTRY, Prestonsburg, KY`
  **TR:** `S3-434505955` | Name: `Helos LLC` | Addr: `Prestonsburg, Bldg Brian Griffith Dentistry, Kentucky, Wrights Lane`

- **S1:** `S1-110295930` | Name: `PF Bread LLC` | Addr: `Jonesborough, TN, 492 Taylor Bridge Road`
  **TR:** `S2-915385449` | Name: `PF Brrda LLC` | Addr: `492 1/2 TAYLOR BRIDGE RD, JONESBOROUGH, TN`

- **S1:** `S1-900882482` | Name: `Vision Medicine` | Addr: `NY, 32 Carthage Drive, Unit Apartment 1, Rochester`
  **TR:** `S2-322147227` | Name: `Vision (Service)` | Addr: `ROCHESTER, NY, 32 CATHAGE DR`

### B. Pushed Out by Top-K Rank Cutoff

- **S1:** `S1-136894582` | Name: `Economic Development United Fellowship` | Addr: `3841 Forsyth Park Court, Greenville, NC`
  **TR:** `S3-975464331` (Rank 98) | Name: `Economic Dvéegpemnt United Fellowship` | Addr: `Winterville, North Carolina, 3841 Forsyth Park Ct` | Score: 1.78

- **S1:** `S1-633779655` | Name: `Aspen Zhejiang LLC` | Addr: `24304 33rd Avenue, WA, Spanaway`
  **TR:** `S2-334509084` (Rank 69) | Name: `Aspen Zejgang LLC` | Addr: `4304 33RD AVENUE, SPANAWAY, WA` | Score: 1.94

- **S1:** `S1-633779655` | Name: `Aspen Zhejiang LLC` | Addr: `24304 33rd Avenue, WA, Spanaway`
  **TR:** `S2-24729487` (Rank 68) | Name: `Aspen Zhmg LLC` | Addr: `` | Score: 1.94

- **S1:** `S1-753180274` | Name: `Helios LLC` | Addr: `306 Wrights Lane, Bldg BRIAN GRIFFITH DENTISTRY, Prestonsburg, KY`
  **TR:** `S2-122241825` (Rank 43) | Name: `Helios  LLC Center` | Addr: `WRIGHTS LN, PRESTONSBURG, KY` | Score: 3.06

- **S1:** `S1-749675254` | Name: `Great Advanced Airport LLC` | Addr: `3400 Skipping Rock Way, VA, Richmond City`
  **TR:** `S3-548928860` (Rank 59) | Name: `Great Airport Advanced  LLC` | Addr: `Richmond, Virginia, 3400 Skipping Rock Way` | Score: 2.33

### C. Generated as Candidate but Rejected by Model (Threshold 0.90)

- **S1:** `S1-17666546` | Name: `Pacific Learning Laboratories LLC` | Addr: `4810 Nassau Avenue, Sand Springs, OK`
  **TR:** `S2-341313901` (Prob 0.5790) | Name: `Pacific Learning Laboratories L.L.C.` | Addr: ``

- **S1:** `S1-150829563` | Name: `Verron Software, LLC` | Addr: `25432 Park Avenue, Buckeye, AZ`
  **TR:** `S3-824134500` (Prob 0.8839) | Name: `Verron Software,  LLC` | Addr: ``

- **S1:** `S1-335752330` | Name: `Ranna Brill Certified Vulcan Upper Allen Township` | Addr: `413 Reservoir Road, Upper Allen Township, PA`
  **TR:** `S2-897778962` (Prob 0.7652) | Name: `rannabrillcertified.com` | Addr: `RESERVOIR RD, MECHANICSBURG, PA`

- **S1:** `S1-491232871` | Name: `Marthena Mowrer, P.A.` | Addr: `AR, 2703 55th Street, Rogers, Unit Apt 203`
  **TR:** `S2-815856445` (Prob 0.8548) | Name: `Marthena Mowrer, P.A. LP` | Addr: ``

- **S1:** `S1-166332747` | Name: `Orthopedic Cascade Clinic LLC` | Addr: `11203 Lynette Court, Spotsylvania County, VA`
  **TR:** `S2-356398714` (Prob 0.7413) | Name: `Orthopedic Cascade Clgch LLC` | Addr: `1203 Lynette Ct, FREDERICKSBURG, VA`

