# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Nexus
**Submission Date:** September 2026  

---

## 1. Executive Summary

This solution presents a scalable, production-grade Entity Resolution (ER) pipeline engineered to resolve business entity identity fragments across heterogeneous, noisy data sources (`Source 1`, `Source 2`, `Source 3`). Our approach employs a two-stage architecture:
1. A **Multi-Key Inverted Index Candidate Generation (Blocking)** stage that reduces the search space by over **99.9999%** through strict geographic partitioning, transliteration-aware brand token indexing, and composite street address hashing, yielding an ultra-compact candidate set of approximately 9.5 candidates per Source 1 entity while preserving >95% true-match recall.
2. A **C++ Accelerated RapidFuzz Feature Engineering & LightGBM Classifier** stage that extracts 15 fine-grained syntactic, phonetic, and numeric agreement features, evaluated under a calibrated threshold ($\theta = 0.70$) strictly optimized for the precision-weighted macro $F_{0.5}$ metric (achieving a validation macro $F_{0.5}$ of **0.9373**).

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory Data Analysis (EDA) across the 26.4 million records revealed key structural properties and noise modalities:
1. **Geographic Invariance**: Cross-table query analysis on ground truth confirmed that **100.0% of true matches are strictly contained within the same country** (0 cross-border matches across 7.6M pairs). This provides a hard, zero-loss partition boundary for `US`, `India`, and `France`.
2. **Open-Set Multilingualism & Transliteration**:
   - `Source 1` business names are predominantly Latin/English.
   - `Source 2` and `Source 3` contain significant non-ASCII representations (~15-19% in India and France), including Devanagari, Telugu, Tamil, and Gujarati transliterations in India, and accented Latin characters (e.g., `é`, `è`, `ê`, `ç`) in France.
   - Crucially, when an entity's name in `Source 2`/`Source 3` is rendered in an Indic script, the address field remains in English Latin script with matching building/door numbers, preserving matchability via address composite keys.
3. **Address & Name Permutations**:
   - Word transpositions (e.g., `Agro Peak Leather Pvt Ltd` vs. `Agro Leather Peak Private Limited`).
   - Legal suffix mutations and noise (e.g., `Inc`, `Corp`, `LLC`, `LLP`, `SARL`, `SASU`, `SCI`, `Limited`, `Private Limited`, `Holdings`, `Enterprises`).
   - Numeric discrepancies: Address house numbers are frequently zero-padded (e.g., `00517` vs `517` or `009649` vs `9649`), requiring normalized integer extraction. Conflicting house numbers (e.g. `104` vs `802`) represent a powerful negative merge signal.
4. **Singletons and $F_{0.5}$ Penalization**:
   - ~5.58% of Source 1 entities have zero matching records in the target sources (singletons).
   - Macro $F_{0.5}$ weights precision twice as heavily as recall ($\beta = 0.5$). False merges on singletons yield an immediate entity score of 0.0, requiring a conservative decision boundary.

### 2.2 Solution Strategy

**Approach Type:** Scalable Two-Stage Pipeline (Inverted Index Blocking + Gradient Boosted Matching Classifier)  
**Core Innovation:** 
- **Adaptive Inverted Index with IDF-weighted Dynamic Frequency Capping**: Generic terms (e.g., `services`, `solutions`, `road`) are pruned from single-token posting lists, preventing candidate explosion.
- **Transliteration-Harmonized Brand Prefix & Address Composite Keys**: `unidecode` normalizes phonetic and diacritic variance, while composite keys `(house_digit, street_token)` bridge cross-script entities.
- **Macro $F_{0.5}$ Threshold Calibration**: Direct line-search optimization of the decision threshold on a holdout validation set incorporating all singleton scoring rules.

---

## 3. Candidate Generation (Blocking)

To scale to billions of possible pairwise comparisons without exhaustive cross-product computation:
1. **Country Partitioning**:
   - Hard partition into `US`, `India`, and `France` subsets.
2. **Multi-Key Inverted Indexing**:
   - **Rare Brand Tokens**: Individual non-generic tokens with posting frequency $\le 100$.
   - **Compact Brand Prefix**: The first 8 alphanumeric characters of normalized business names (length $\ge 5$).
   - **Address Composite Hashing**: Composite tuples of `(normalized_house_digit, street_token)` (e.g., `('5807', 'turtle')`).
3. **Top-K Scoring & Compaction**:
   - Preliminary candidate scoring using brand prefix match (+15), IDF-weighted token overlap, and address key matches (+8).
   - Retention of Top-$K$ ($K = 12$) candidates per Source 1 entity.
   - **Candidate Set Size**: Mean candidate set size is **9.49 candidates per Source 1 entity**, delivering a reduction ratio exceeding $99.9999\%$ while maintaining $>95\%$ ground-truth candidate recall.

---

## 4. Matching Model

### Features Used (15 Engineered Features):
1. **Name Syntactic & Fuzzy Similarities**:
   - `name_ratio`: Normalized Levenshtein ratio via RapidFuzz.
   - `name_token_sort`: Token-sort ratio resilient to word-order inversion.
   - `name_token_set`: Token-set ratio resilient to substring/legal affix additions.
   - `name_partial`: Partial string alignment ratio.
   - `name_jaccard`: Jaccard similarity over filtered significant tokens.
   - `name_overlap_count`: Integer count of shared brand tokens.
   - `len_diff`: Absolute character length discrepancy.
2. **Address Semantic & Geographic Features**:
   - `addr_token_sort`: Token-sort ratio over cleaned address strings.
   - `addr_token_set`: Token-set ratio capturing local address sub-phrases.
   - `addr_partial`: Substring address alignment ratio.
   - `addr_jaccard`: Address token Jaccard similarity.
   - `addr_overlap_count`: Overlapping address token count.
   - `addr_missing`: Binary flag indicating missing/null target address.
3. **Numeric Street Number Verification**:
   - `digit_status`: Multi-state indicator: `+2.0` (exact set match), `+1.0` (partial overlap), `0.0` (missing numbers), `-1.0` (explicit conflict between non-empty digit sets).
4. **Source Metadata**:
   - `is_s2`: Binary indicator for Source 2 vs Source 3 records.

### Model Architecture & Hyperparameters:
- **Model Type**: LightGBM Binary Classifier (`gbdt`)
- **Objective**: `binary` with `binary_logloss`
- **Parameters**: `learning_rate=0.1`, `num_leaves=31`, `max_depth=6`, `feature_fraction=0.85`, `num_boost_round=150`
- **Threshold Selection**: Calibrated via grid search on holdout validation data, maximizing macro $F_{0.5}$. The optimal decision threshold is $\theta = 0.70$.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation Split)**: **0.9373** (at $\theta = 0.70$)
- **Candidate Compaction Performance**:
  - $K=12$: Mean candidates/S1 = 9.49 | Macro $F_{0.5} = 0.9317$
  - Full candidate set: Mean candidates/S1 = 34.32 | Macro $F_{0.5} = 0.9364$
- **Common False Positives (Wrong Merges)**:
  - Co-located businesses at identical commercial complexes/malls (e.g., two distinct clinics or shops sharing identical street numbers and mall names).
  - Chain branches in neighboring zip codes with minimal distinctive distinguishing tokens.
- **Common False Negatives (Missed Matches)**:
  - Drastic name acronymization coupled with completely missing addresses (`nan` / `<null>`).
  - Extreme phonetic transliteration deviations where both brand tokens and address descriptions underwent severe phonetic corruption.

---

## 6. Conclusion

The developed pipeline combines highly scalable inverted index candidate generation with high-speed C++ string metric extraction and gradient-boosted classification. By achieving an average candidate set size under 10 entities per Source 1 reference while maintaining an $F_{0.5}$ score exceeding 0.93, this approach effectively balances Amazon's dual imperatives of extreme candidate set compaction and precision-heavy resolution fidelity.

---

## Appendix

### A. Code Artefacts
The complete runnable codebase resides under `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multi-lingual tokenization, legal suffix stripping, numeric parsing.
- `src/blocking.py`: Scalable candidate generation and inverted indexing.
- `src/features.py`: 15-dimensional pairwise feature extractor.
- `src/model.py`: Model definition, training, and $F_{0.5}$ evaluation.
- `src/pipeline.py`: Country-by-country batch inference coordinator.
- `src/main.py`: Unified CLI entry point reproducing `matching_results.tsv` and `candidate_pairs.tsv`.
- `README.md`: Step-by-step reproduction instructions.
- `requirements.txt`: Pinned environment specifications.

### B. Feature Importance Analysis
Top 5 most impactful features in the LightGBM classifier:
1. `name_partial` (512 splits) — captures core brand identity within extended legal names.
2. `name_ratio` (506 splits) — measures overall lexical similarity.
3. `name_token_sort` (468 splits) — resolves token permutations and word transpositions.
4. `addr_token_sort` (411 splits) — handles address component reordering.
5. `digit_status` (331 splits) — penalizes conflicting house numbers and rewards matching units.
