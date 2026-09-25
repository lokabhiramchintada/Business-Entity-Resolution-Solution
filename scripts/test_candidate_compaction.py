import duckdb
import time
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
from collections import defaultdict
import rapidfuzz.fuzz as fuzz
import lightgbm as lgb

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Candidate Set Compaction (Top-8 / Top-10 / Top-12) ---")
t0 = time.time()

# Sample 13,000 entities
sample_all = con.execute(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    USING SAMPLE 13000 (reservoir, 42);
""").df()

train_s1 = sample_all.iloc[:10000].copy()
val_s1 = sample_all.iloc[10000:].copy()

con.register('sample_all_s1', sample_all[['entity_id']])
gt_all = con.execute(f"""
    WITH unnested AS (
        SELECT 
            g.source1_entity_id,
            unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True) g
        JOIN sample_all_s1 s ON g.source1_entity_id = s.entity_id
        WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != ''
    )
    SELECT * FROM unnested;
""").df()

gt_dict = defaultdict(set)
for _, r in gt_all.iterrows():
    gt_dict[r['source1_entity_id']].add(r['matched_id'])

val_gt_dict = {s1: gt_dict[s1] for s1 in val_s1['entity_id'] if s1 in gt_dict}
val_s1_ids = list(val_s1['entity_id'].values)

all_gt_matched_ids = set(gt_all['matched_id'].values)
con.register('matched_ids_df', pd.DataFrame({'entity_id': list(all_gt_matched_ids)}))

targets_df = con.execute(f"""
    WITH matched_s2 AS (
        SELECT s.entity_id, s.business_name, s.business_address, s.country
        FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True) s
        JOIN matched_ids_df m ON s.entity_id = m.entity_id
    ),
    matched_s3 AS (
        SELECT s.entity_id, s.business_name, s.business_address, s.country
        FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True) s
        JOIN matched_ids_df m ON s.entity_id = m.entity_id
    ),
    distractors_s2 AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)
        USING SAMPLE 100000 (reservoir, 99)
    ),
    distractors_s3 AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)
        USING SAMPLE 100000 (reservoir, 99)
    )
    SELECT * FROM matched_s2
    UNION ALL SELECT * FROM matched_s3
    UNION ALL SELECT * FROM distractors_s2
    UNION ALL SELECT * FROM distractors_s3;
""").df().drop_duplicates(subset=['entity_id']).reset_index(drop=True)

LEGAL_TERMS = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'services', 'service', 'enterprises', 'enterprise',
    'technologies', 'technology', 'tech', 'solutions', 'solution', 'consulting', 'consultants',
    'holdings', 'holding', 'group', 'international', 'global', 'india', 'us', 'usa',
    'sarl', 'sas', 'sci', 'sa', 'eurl', 'france', 'fr', 'sri', 'shri', 'ms', 'm/s', 'the',
    'and', 'associates', 'industries', 'industry', 'management', 'center', 'centre', 'com', 'org', 'net'
}

GENERIC_ADDR = {
    'road', 'street', 'drive', 'lane', 'avenue', 'court', 'circle', 'parkway', 'highway', 'way', 'place',
    'rd', 'st', 'dr', 'ln', 'ave', 'ct', 'cir', 'pkwy', 'hwy', 'pl', 'blvd', 'boulevard',
    'near', 'opp', 'floor', 'unit', 'apt', 'null', 'nan', 'house', 'hno', 'plot', 'flat', 'no',
    'block', 'phase', 'nagar', 'colony', 'bldg', 'building', 'tower', 'complex',
    'north', 'south', 'east', 'west', 'upper', 'lower', 'suite', 'ste', 'room', 'rm', 'box', 'pob',
    'c/o', 'w/o', 's/o', 'd/o', 'village', 'vill', 'dist', 'district', 'post', 'po', 'tq', 'taluka',
    'state', 'city', 'town', 'layout', 'enclave', 'park', 'gali', 'marg', 'rasta'
}

def clean_tokens(s):
    if not s or str(s) == 'nan':
        return []
    s = unidecode(str(s)).lower()
    s = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) > 1 and w not in LEGAL_TERMS]

def clean_compact_name(s):
    if not s or str(s) == 'nan':
        return ""
    s = unidecode(str(s)).lower()
    s = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    for term in sorted(LEGAL_TERMS, key=len, reverse=True):
        if s.endswith(term):
            s = s[:-len(term)]
    return s

def extract_digits(s):
    if not s or str(s) == 'nan':
        return []
    digits = re.findall(r'\b\d+\b', str(s))
    return [d.lstrip('0') for d in digits if d.lstrip('0')]

def extract_addr_tokens(s):
    if not s or str(s) == 'nan':
        return []
    s = unidecode(str(s)).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) >= 3 and w not in GENERIC_ADDR and not w.isdigit()]

t_ids = targets_df['entity_id'].values
t_names = targets_df['business_name'].values
t_addrs = targets_df['business_address'].values
t_countries = targets_df['country'].values
N_T = len(targets_df)

country_indexes = {}
for c in ['US', 'India', 'France']:
    country_indexes[c] = {
        'name_tok': defaultdict(list),
        'compact': defaultdict(list),
        'addr_key': defaultdict(list),
    }

for i in range(N_T):
    c = t_countries[i]
    if c not in country_indexes:
        continue
    c_idx = country_indexes[c]
    name = t_names[i]
    addr = t_addrs[i]
    for tok in set(clean_tokens(name)):
        c_idx['name_tok'][tok].append(i)
    cname = clean_compact_name(name)
    if len(cname) >= 5:
        c_idx['compact'][cname[:8]].append(i)
    digs = extract_digits(addr)
    atoks = extract_addr_tokens(addr)
    if digs and atoks:
        for d in digs[:2]:
            for w in atoks[:2]:
                c_idx['addr_key'][(d, w)].append(i)

def get_candidates_for_s1(s1_row, max_cand_per_key=100, top_k_cap=None):
    c = s1_row['country']
    if c not in country_indexes:
        return []
    c_idx = country_indexes[c]
    name = s1_row['business_name']
    addr = s1_row['business_address']
    
    cand_scores = defaultdict(float)
    
    # 1. Compact prefix
    cname = clean_compact_name(name)
    if len(cname) >= 5:
        postings = c_idx['compact'].get(cname[:8], [])
        if len(postings) <= max_cand_per_key:
            for idx in postings:
                cand_scores[idx] += 15.0
                
    # 2. Name tokens (rarest first)
    toks = clean_tokens(name)
    toks.sort(key=lambda t: len(c_idx['name_tok'].get(t, [])))
    for t in toks:
        postings = c_idx['name_tok'].get(t, [])
        if 0 < len(postings) <= max_cand_per_key:
            # weight inversely proportional to posting length
            w = 10.0 / (1.0 + np.log1p(len(postings)))
            for idx in postings:
                cand_scores[idx] += w
                
    # 3. Addr key
    digs = extract_digits(addr)
    atoks = extract_addr_tokens(addr)
    if digs and atoks:
        for d in digs[:2]:
            for w in atoks[:2]:
                postings = c_idx['addr_key'].get((d, w), [])
                if 0 < len(postings) <= max_cand_per_key:
                    for idx in postings:
                        cand_scores[idx] += 8.0
                        
    if not cand_scores:
        return []
        
    if top_k_cap is not None and len(cand_scores) > top_k_cap:
        # Keep top_k_cap sorted by score
        return sorted(cand_scores, key=cand_scores.get, reverse=True)[:top_k_cap]
    else:
        return list(cand_scores.keys())

def extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id):
    s1_n_str = str(s1_name) if pd.notna(s1_name) else ""
    t_n_str = str(tgt_name) if pd.notna(tgt_name) else ""
    s1_a_str = str(s1_addr) if pd.notna(s1_addr) else ""
    t_a_str = str(tgt_addr) if pd.notna(tgt_addr) else ""
    
    s1_n_clean = unidecode(s1_n_str).lower()
    t_n_clean = unidecode(t_n_str).lower()
    s1_a_clean = unidecode(s1_a_str).lower()
    t_a_clean = unidecode(t_a_str).lower()
    
    name_ratio = fuzz.ratio(s1_n_clean, t_n_clean) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_n_clean, t_n_clean) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_n_clean, t_n_clean) / 100.0
    name_partial = fuzz.partial_ratio(s1_n_clean, t_n_clean) / 100.0
    
    s1_toks = set(clean_tokens(s1_n_str))
    t_toks = set(clean_tokens(t_n_str))
    tok_union = len(s1_toks | t_toks)
    name_jaccard = (len(s1_toks & t_toks) / tok_union) if tok_union > 0 else 0.0
    name_overlap_count = len(s1_toks & t_toks)
    
    addr_missing = 1.0 if not t_a_clean or t_a_clean == 'nan' or '<null>' in t_a_clean else 0.0
    if not addr_missing and s1_a_clean:
        addr_token_sort = fuzz.token_sort_ratio(s1_a_clean, t_a_clean) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_a_clean, t_a_clean) / 100.0
        addr_partial = fuzz.partial_ratio(s1_a_clean, t_a_clean) / 100.0
        
        s1_atoks = set(extract_addr_tokens(s1_a_str))
        t_atoks = set(extract_addr_tokens(t_a_str))
        at_union = len(s1_atoks | t_atoks)
        addr_jaccard = (len(s1_atoks & t_atoks) / at_union) if at_union > 0 else 0.0
        addr_overlap_count = len(s1_atoks & t_atoks)
    else:
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_partial = 0.0
        addr_jaccard = 0.0
        addr_overlap_count = 0.0
        
    s1_digs = set(extract_digits(s1_a_str))
    t_digs = set(extract_digits(t_a_str))
    if s1_digs and t_digs:
        if s1_digs == t_digs:
            digit_status = 2.0
        elif s1_digs & t_digs:
            digit_status = 1.0
        else:
            digit_status = -1.0
    else:
        digit_status = 0.0
        
    is_s2 = 1.0 if tgt_id.startswith('S2-') else 0.0
    len_diff = abs(len(s1_n_clean) - len(t_n_clean))
    
    return [
        name_ratio, name_token_sort, name_token_set, name_partial,
        name_jaccard, name_overlap_count,
        addr_token_sort, addr_token_set, addr_partial,
        addr_jaccard, addr_overlap_count,
        digit_status, addr_missing, is_s2, len_diff
    ]

# Train LightGBM model
X_train, y_train = [], []
for _, s1_row in train_s1.iterrows():
    s1_id = s1_row['entity_id']
    s1_name = s1_row['business_name']
    s1_addr = s1_row['business_address']
    cands = get_candidates_for_s1(s1_row, top_k_cap=15)
    true_targets = gt_dict.get(s1_id, set())
    for tid_idx in cands:
        tgt_id = t_ids[tid_idx]
        tgt_name = t_names[tid_idx]
        tgt_addr = t_addrs[tid_idx]
        feats = extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id)
        label = 1 if tgt_id in true_targets else 0
        X_train.append(feats)
        y_train.append(label)

lgb_train = lgb.Dataset(np.array(X_train, dtype=np.float32), label=np.array(y_train, dtype=np.int32))
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'verbose': -1,
    'random_state': 42
}
model = lgb.train(params, lgb_train, num_boost_round=150)

def compute_macro_f05(pred_dict, gt_dict, all_s1_ids):
    f05_scores = []
    for s1 in all_s1_ids:
        true_set = gt_dict.get(s1, set())
        pred_set = set(pred_dict.get(s1, []))
        if len(true_set) == 0:
            f05_scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
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
        denom = 0.25 * prec + rec
        score = (1.25 * prec * rec) / denom if denom > 0 else 0.0
        f05_scores.append(score)
    return np.mean(f05_scores)

for K_cap in [6, 8, 10, 12, 15, None]:
    val_pair_records = []
    for _, s1_row in val_s1.iterrows():
        s1_id = s1_row['entity_id']
        s1_name = s1_row['business_name']
        s1_addr = s1_row['business_address']
        cands = get_candidates_for_s1(s1_row, top_k_cap=K_cap)
        for tid_idx in cands:
            tgt_id = t_ids[tid_idx]
            tgt_name = t_names[tid_idx]
            tgt_addr = t_addrs[tid_idx]
            feats = extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id)
            val_pair_records.append((s1_id, tgt_id, feats))
            
    X_val = np.array([r[2] for r in val_pair_records], dtype=np.float32)
    val_preds_proba = model.predict(X_val)
    
    mean_cand = len(val_pair_records) / len(val_s1)
    
    best_thresh, best_f05 = 0.70, 0.0
    for thresh in [0.60, 0.65, 0.70, 0.75]:
        preds_by_s1 = defaultdict(list)
        for (s1_id, tgt_id, _), prob in zip(val_pair_records, val_preds_proba):
            if prob >= thresh:
                preds_by_s1[s1_id].append(tgt_id)
        score = compute_macro_f05(preds_by_s1, val_gt_dict, val_s1_ids)
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh
            
    print(f"Top-K Cap = {str(K_cap):5s} | Mean Candidates/S1 = {mean_cand:5.2f} | Best Macro F_0.5 = {best_f05:.4f} (at thresh {best_thresh:.2f})")

print(f"Total time: {time.time() - t0:.2f}s")
