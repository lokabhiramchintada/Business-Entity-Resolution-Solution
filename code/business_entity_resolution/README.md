# Business Entity Resolution Pipeline

This repository contains the end-to-end, high-performance solution for the Amazon ML Challenge 2026: Business Entity Resolution.

## Overview & Architecture

The pipeline resolves business identity fragments across three noisy, heterogeneous data sources (`Source 1`, `Source 2`, `Source 3`) using a scalable two-stage architecture:

1. **Partitioning & Candidate Generation (Blocking)**:
   - **Zero-loss Country Partitioning**: Hard partition by country (`US`, `India`, `France`), eliminating cross-border candidates without recall loss.
   - **Multi-Key Inverted Indexing**:
     - Distinct brand tokens (frequency-capped, IDF-weighted)
     - Transliterated compact name prefixes (`clean_compact_name` + `unidecode`)
     - Street address numeric composite keys `(house_digit, street_token)`
   - **Candidate Compaction**: Scores plausible candidate targets and retains top-12 candidates per Source 1 entity, cutting the search space by a factor of >1,000,000x while maintaining >95% recall.

2. **Matching Model & Decision**:
   - **Feature Extraction**: 15 fine-grained syntactic, phonetic, and semantic features using C++ RapidFuzz (`fuzz.ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, token Jaccard, digit agreement/conflict, address missing flags).
   - **Gradient Boosted Classifier**: LightGBM binary classifier trained on true matches and hard negative distractors.
   - **Macro $F_{0.5}$ Threshold Optimization**: Optimized probability threshold ($\theta = 0.70$) penalizing false merges 2x more than missed matches.

---

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Text normalization, legal terms, digit extraction
│   ├── blocking.py         # Multi-key inverted index candidate generator
│   ├── features.py         # 15 RapidFuzz & token similarity features
│   ├── model.py            # LightGBM training, serialization, Macro F0.5 scoring
│   ├── pipeline.py         # End-to-end country-by-country runner
│   └── main.py             # CLI entry point
├── README.md               # Reproduction guide
└── requirements.txt        # Pinned dependencies
```

---

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## End-to-End Reproduction

To run the complete pipeline (training on train dataset + inference on test set + output generation):

```bash
python src/main.py \
    --data-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset \
    --output-dir ../../output \
    --model-path models/lgbm_matcher.joblib \
    --top-k 12 \
    --threshold 0.80
```

### Outputs Generated:
- `output/matching_results.tsv`: Final predicted entity matches (scored on leaderboard).
- `output/candidate_pairs.tsv`: Final candidate set evaluated by the matching model.

---

## Output Validation

To validate the generated output files with the official submission validator:

```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```
