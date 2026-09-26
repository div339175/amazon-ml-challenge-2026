# Phase 6: Error Analysis and Quantification

- Total False Negatives Analyzed: 1,144
- Singleton False Positive Merges: 0

## 1. False Negative Breakdown by Category

| Error Category | Count | Percentage | Primary Root Cause |
| --- | --- | --- | --- |
| **missing_target_address** | 366 | 32.0% | Target address empty; model penalizes address features even if name matches
| **transliteration_or_script** | 338 | 29.5% | Non-standard Indic or accented spelling divergence
| **spelling_variation_or_typo** | 214 | 18.7% | Name typo or minor edit distance lowered prob below 0.90
| **dba_or_different_brand_name** | 176 | 15.4% | Completely different trade name / DBA; requires strong address link
| **address_variation_or_abbrev** | 25 | 2.2% | Address formatting difference without house number overlap
| **house_number_difference** | 17 | 1.5% | Different suite / apartment or slight municipal number mismatch
| **word_order_transposition** | 8 | 0.7% | Other

## 2. Representative Examples by Error Category

### transliteration_or_script
- **S1:** `S1-993527714` | `George Saul Inc` | `Unit UNIT 367, 1400 Great Wolf Drive, WI, Village Of Lake Delton`
  **TR:** `S3-250788566` | `Úmbrayuma` | `1400 Great Wolf Dr, Unit UNIT 367, Village Of Lake Delton, Wisconsin`

- **S1:** `S1-832050262` | `Shakti Agro Limited` | `C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa`
  **TR:** `S2-856926544` | `ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍` | `C/O GURNAV SINGH SALUJA, SADAR, Odisha`

- **S1:** `S1-360488265` | `Jai Systems Private Limited` | `19/1276, Indira Nagar, Lucknow, Lucknow, Uttar Pradesh`
  **TR:** `S3-978889721` | `जय सिस्टम्स प्राइवेट लिमिटेड` | `उत्तर प्रदेश, 19/1276., Lucknow, Indira Nagar, Lucknow`

### dba_or_different_brand_name
- **S1:** `S1-169008536` | `Urban Projects` | `C/O Alauddin Molla Noorpur Ramnagar 24 Parganas, Diamond Harbour - I, South 24 Parganas, West Bengal`
  **TR:** `S3-451617552` | `Smt Ariaarc` | `Block C-522 C/o Alauddin Molla Noorpur Ramnagar 24 Parganas, Diamond Harbour - I, South 24 Parganas, পশ্চিমবঙ্গ`

- **S1:** `S1-966027321` | `Probst & Duran Newhold LLC` | `Boston, 31 Floyd Street, MA`
  **TR:** `S2-4379937` | `NYLADREX` | `31D FLOYD STREET, BOSTON, MA`

- **S1:** `S1-891013868` | `Family Trusted Health` | `1109 Southview Drive, Oxon Hill, MD`
  **TR:** `S2-570671307` | `Zephdrex` | `01109 SOUTHVIEW DR, OXON HILL, MD`

### missing_target_address
- **S1:** `S1-633779655` | `Aspen Zhejiang LLC` | `24304 33rd Avenue, WA, Spanaway`
  **TR:** `S2-24729487` | `Aspen Zhmg LLC` | ``

- **S1:** `S1-150829563` | `Verron Software, LLC` | `25432 Park Avenue, Buckeye, AZ`
  **TR:** `S3-824134500` | `Verron Software,  LLC` | ``

- **S1:** `S1-772116613` | `Behavioral Health Silver Clinic Inc` | `13822 Northvalley Drive, Hagerstown, MD`
  **TR:** `S2-996318936` | `Behavioral Health  Silver Clinic Inc` | ``

### address_variation_or_abbrev
- **S1:** `S1-680008372` | `Solar Solutions Private Limited` | `8124, Sector - C, Pocket - 8 Vasant Kunj, New Delhi, South West Delhi, Delhi`
  **TR:** `S3-306689701` | `Solar Solutions  Private Limited` | `8124, Divreportingcircle, DL`

- **S1:** `S1-680008372` | `Solar Solutions Private Limited` | `8124, Sector - C, Pocket - 8 Vasant Kunj, New Delhi, South West Delhi, Delhi`
  **TR:** `S3-319382210` | `Private Solar Solutions Limited` | `8124, Divreportingcircle, DL`

- **S1:** `S1-913785285` | `Empire Express Inc` | `14 Ashberry Lane, Three Way, TN`
  **TR:** `S3-909333749` | `inc empire express` | `Humblodt CITY, Tennessee, Ashberry Lane`

### spelling_variation_or_typo
- **S1:** `S1-890009542` | `SJ Ace Vendome Inc` | `60 Uneeda Street, Madison, WV`
  **TR:** `S3-203500929` | `SJ Ace Venhdome Inc` | `West Virginia, 1 Uneeda Street, Madison`

- **S1:** `S1-196309581` | `UA Investments Private Limited` | `616, Swati Crimson And, Clover, Nr. Shilaj Circle, Daskroi, Ahmedabad, Gujarat`
  **TR:** `S3-361204897` | `UA Investments Privae Limited` | `H.no 176 616, Ahmedabad, Daskroi, GJ`

- **S1:** `S1-355685907` | `Regional Scientific, LLC` | `1424 Cottage View Lane, Nashville, TN`
  **TR:** `S2-979678502` | `CENTER REGIONAL LLC` | `COTTAGE VIEW LN, ANTIOCH, TN`

### word_order_transposition
- **S1:** `S1-613517393` | `Haven Exim` | `A/404, Prathna Greens, Sargasan, Gandhinagar, Gandhinagar, Gandhi Nagar, Gujarat`
  **TR:** `S2-649519831` | `Haven Exim Exim` | `A/##404, PRATHNA GREENS, SARGASAN, GANDHINAGAR, GANDHINAGAR, ગુજરાત`

- **S1:** `S1-744150463` | `Primary Care Physicians LP` | `9850 Cedar Drive, Sun City, AZ`
  **TR:** `S2-493485272` | `Primary LP Physicians Care` | `CEDAR DR, SUN CITY, AZ`

- **S1:** `S1-972491065` | `Mk Academy Private Limited` | `Ground Floor Laxmi Complex, C-Scheme, Rajasthan, Jaipur, Shop No 54, Jaipur`
  **TR:** `S2-276422592` | `Academy Mk Private Limited` | `JAIPUR, NO 54, Rajasthan`

### house_number_difference
- **S1:** `S1-859043162` | `Anchor Wireless Enterprises LLC` | `3863 Bright Street, Madill, OK`
  **TR:** `S2-895188960` | `Anchor Wireless` | `3864 BRIGHT ST, MADIL, OK`

- **S1:** `S1-588360895` | `Urban Bakery LLC` | `3343 J F Scott Drive, Aztec, NM`
  **TR:** `S2-86401923` | `Urban Bakery` | `3329 J F SCOTT DR, AZTEC, NM`

- **S1:** `S1-883050466` | `Nematech Mozayyx` | `NY, 98 River Road, Islip`
  **TR:** `S3-194388079` | `NEMATECH MOZAYYX` | `103 River Road, Great River, New York`

