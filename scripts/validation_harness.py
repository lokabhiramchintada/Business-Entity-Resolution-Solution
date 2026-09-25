import duckdb
import time
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
from collections import defaultdict, Counter
import rapidfuzz.fuzz as fuzz
from rapidfuzz.distance import Levenshtein
import lightgbm as lgb
from sklearn.model_selection import train_test_split

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Rapid Prototype: Feature Extraction & LightGBM on Validation Split ---")
t0 = time.time()

# 1. Sample 10,000 S1 entities from train (5,000 US, 5,000 India)
val_s1 = con.execute(f"""
    WITH us_sample AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
        WHERE country = 'US'
        USING SAMPLE 5000 (reservoir, 101)
    ),
    in_sample AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
        WHERE country = 'India'
        USING SAMPLE 5000 (reservoir, 101)
    )
    SELECT * FROM us_sample UNION ALL SELECT * FROM in_sample;
""").df()

con.register('val_s1_df', val_s1[['entity_id']])
gt_val = con.execute(f"""
    WITH unnested AS (
        SELECT 
            g.source1_entity_id,
            unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True) g
        JOIN val_s1_df v ON g.source1_entity_id = v.entity_id
        WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != ''
    )
    SELECT * FROM unnested;
""").df()

# Ground truth mapping
gt_dict = defaultdict(set)
for _, r in gt_val.iterrows():
    gt_dict[r['source1_entity_id']].add(r['matched_id'])

# Also record all S1 entities (including singletons)
all_val_s1_ids = list(val_s1['entity_id'].values)
print(f"Validation S1 entities: {len(all_val_s1_ids)} ({len(gt_dict)} with matches, {len(all_val_s1_ids) - len(gt_dict)} singletons)")
total_true_pairs = sum(len(v) for v in gt_dict.values())
print(f"Total true match pairs in validation: {total_true_pairs}")

# Function to compute exact Macro F_0.5 as defined in the challenge
def compute_macro_f05(pred_dict, gt_dict, all_s1_ids):
    """
    pred_dict: {s1_id: [matched_ids]}
    gt_dict: {s1_id: set(matched_ids)}
    all_s1_ids: list/set of all S1 entity IDs
    """
    f05_scores = []
    for s1 in all_s1_ids:
        true_set = gt_dict.get(s1, set())
        pred_list = pred_dict.get(s1, [])
        pred_set = set(pred_list)
        
        # Singleton logic:
        # A Source 1 entity with no true matches scores 1.0 when you correctly predict an empty list,
        # and 0.0 when you predict any match for it.
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        # Non-singleton logic:
        if len(pred_set) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        
        # F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        denom = 0.25 * prec + rec
        if denom == 0:
            score = 0.0
        else:
            score = (1.25 * prec * rec) / denom
        f05_scores.append(score)
        
    return np.mean(f05_scores)

print(f"Validation harness ready. Elapsed: {time.time() - t0:.2f}s")
