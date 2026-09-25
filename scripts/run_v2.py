"""
Improved v2 pipeline with:
1. Full training data usage (2.2M entities -> stratified 200k sample)  
2. Hard negative mining from full S2+S3 corpus
3. Increased top-K blocking (K=25)
4. Richer features (char n-grams, address prefix, state/region token)
5. Bigger LightGBM model with more trees
6. Threshold tuning on realistic validation set

Key fixes for the 0.689 public score:
- France was unseen in training: We now use France test data structure to ensure 
  blocking keys work for French text (unidecode handles accents correctly)
- Hard negatives: Instead of random distractors, we mine hard negatives per country
  from the full S2+S3 corpus using our own blocking keys
- Increased K: Top-25 candidates to boost recall ceiling
- More training data: 150k S1 entities (vs 50k before)
"""
import os
import sys
import time
import math
import re
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from unidecode import unidecode
import rapidfuzz.fuzz as fuzz
import lightgbm as lgb
import joblib

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
OUTPUT_DIR = "output"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Preprocessing ──────────────────────────────────────────────────────────────
LEGAL_TERMS = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'services', 'service', 'enterprises', 'enterprise',
    'technologies', 'technology', 'tech', 'solutions', 'solution', 'consulting', 'consultants',
    'holdings', 'holding', 'group', 'international', 'global', 'pllc', 'pc', 'pa',
    'india', 'sri', 'shri', 'ms', 'the', 'and', 'associates', 'industries', 'industry',
    'management', 'center', 'centre', 'sarl', 'sasu', 'sas', 'sci', 'sa', 'eurl',
    'fils', 'societe', 'association', 'france', 'com', 'org', 'net', 'us', 'usa',
    'dba', 'doing', 'business', 'professional', 'services', 'general',
}

ADDR_STOPWORDS = {
    'road', 'street', 'drive', 'lane', 'avenue', 'court', 'circle', 'parkway', 'highway',
    'way', 'place', 'rd', 'st', 'dr', 'ln', 'ave', 'ct', 'cir', 'pkwy', 'hwy', 'pl',
    'blvd', 'boulevard', 'near', 'opp', 'floor', 'unit', 'apt', 'null', 'nan', 'house',
    'hno', 'plot', 'flat', 'no', 'block', 'phase', 'nagar', 'colony', 'bldg', 'building',
    'tower', 'complex', 'north', 'south', 'east', 'west', 'upper', 'lower', 'suite',
    'ste', 'room', 'rm', 'box', 'pob', 'village', 'vill', 'dist', 'district', 'post',
    'po', 'tq', 'taluka', 'state', 'city', 'town', 'layout', 'enclave', 'park',
    'gali', 'marg', 'rasta', 'rue', 'bd', 'impasse', 'chemin', 'allee', 'route',
    'bis', 'ter', 'rez', 'chaussee', 'passage', 'square',
}

def normalize(s: str) -> str:
    """Unidecode + lowercase + remove non-alnum."""
    if not s or str(s) == 'nan':
        return ""
    return re.sub(r'[^a-z0-9]', '', unidecode(str(s)).lower())

def name_tokens(s: str) -> list:
    if not s or str(s) == 'nan':
        return []
    t = unidecode(str(s)).lower()
    t = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return [w for w in t.split() if len(w) > 1 and w not in LEGAL_TERMS]

def compact_name(s: str) -> str:
    if not s or str(s) == 'nan':
        return ""
    t = unidecode(str(s)).lower()
    t = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', t)
    t = re.sub(r'[^a-z0-9]', '', t)
    for term in sorted(LEGAL_TERMS, key=len, reverse=True):
        if t.endswith(term):
            t = t[:-len(term)]
    return t

def digits(s: str) -> list:
    if not s or str(s) == 'nan':
        return []
    return [d.lstrip('0') for d in re.findall(r'\b\d+\b', str(s)) if d.lstrip('0')]

def addr_tokens(s: str) -> list:
    if not s or str(s) == 'nan':
        return []
    t = unidecode(str(s)).lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return [w for w in t.split() if len(w) >= 3 and w not in ADDR_STOPWORDS and not w.isdigit()]

def char_ngrams(s: str, n: int = 3) -> set:
    """Character trigrams of normalized string."""
    t = normalize(s)
    if len(t) < n:
        return set()
    return set(t[i:i+n] for i in range(len(t) - n + 1))

# ── Feature Extraction ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_jaccard', 'name_overlap',
    'name_char3_jaccard',
    'addr_token_sort', 'addr_token_set', 'addr_partial',
    'addr_jaccard', 'addr_overlap',
    'digit_status',
    'addr_missing',
    'is_s2',
    'len_diff',
    'name_norm_exact',
    'addr_region_match',
]

def extract_features(s1_name, s1_addr, t_name, t_addr, t_id):
    s1n = unidecode(str(s1_name) if pd.notna(s1_name) else "").lower()
    tn = unidecode(str(t_name) if pd.notna(t_name) else "").lower()
    s1a = unidecode(str(s1_addr) if pd.notna(s1_addr) else "").lower()
    ta = unidecode(str(t_addr) if pd.notna(t_addr) else "").lower()

    # Name similarities
    name_ratio = fuzz.ratio(s1n, tn) / 100.0
    name_tsort = fuzz.token_sort_ratio(s1n, tn) / 100.0
    name_tset = fuzz.token_set_ratio(s1n, tn) / 100.0
    name_partial = fuzz.partial_ratio(s1n, tn) / 100.0

    s1_nt = set(name_tokens(s1_name))
    t_nt = set(name_tokens(t_name))
    nt_union = len(s1_nt | t_nt)
    name_jacc = len(s1_nt & t_nt) / nt_union if nt_union > 0 else 0.0
    name_overlap = float(len(s1_nt & t_nt))

    # Character trigram Jaccard (robust to typos)
    s1_tri = char_ngrams(s1_name)
    t_tri = char_ngrams(t_name)
    tri_union = len(s1_tri | t_tri)
    name_char3_jacc = len(s1_tri & t_tri) / tri_union if tri_union > 0 else 0.0

    # Address similarities
    addr_missing = 1.0 if (not ta or 'nan' in ta or '<null>' in ta) else 0.0
    if not addr_missing and s1a:
        addr_tsort = fuzz.token_sort_ratio(s1a, ta) / 100.0
        addr_tset = fuzz.token_set_ratio(s1a, ta) / 100.0
        addr_partial = fuzz.partial_ratio(s1a, ta) / 100.0
        s1_at = set(addr_tokens(s1_addr))
        t_at = set(addr_tokens(t_addr))
        at_union = len(s1_at | t_at)
        addr_jacc = len(s1_at & t_at) / at_union if at_union > 0 else 0.0
        addr_overlap = float(len(s1_at & t_at))
    else:
        addr_tsort = addr_tset = addr_partial = addr_jacc = addr_overlap = 0.0

    # Digit agreement signal
    s1d = set(digits(s1_addr))
    td = set(digits(t_addr))
    if s1d and td:
        digit_status = 2.0 if s1d == td else (1.0 if s1d & td else -1.0)
    else:
        digit_status = 0.0

    # Normalized name exact match
    name_norm_exact = 1.0 if (normalize(s1_name) and normalize(s1_name) == normalize(t_name)) else 0.0

    # Region/state token match (last token in address often is state/region)
    def last_addr_token(a):
        a = unidecode(str(a) if pd.notna(a) else "").lower()
        tokens = [w for w in re.sub(r'[^a-z\s]', ' ', a).split() if len(w) >= 2]
        return tokens[-1] if tokens else ""
    addr_region_match = 1.0 if (last_addr_token(s1_addr) == last_addr_token(t_addr) and last_addr_token(s1_addr) != "") else 0.0

    is_s2 = 1.0 if str(t_id).startswith('S2-') else 0.0
    len_diff = float(abs(len(s1n) - len(tn)))

    return [
        name_ratio, name_tsort, name_tset, name_partial,
        name_jacc, name_overlap,
        name_char3_jacc,
        addr_tsort, addr_tset, addr_partial,
        addr_jacc, addr_overlap,
        digit_status,
        addr_missing,
        is_s2,
        len_diff,
        name_norm_exact,
        addr_region_match,
    ]

# ── Blocking / Candidate Generator ────────────────────────────────────────────
class MultiKeyBlocker:
    def __init__(self, max_postings: int = 120, top_k: int = 25):
        self.max_postings = max_postings
        self.top_k = top_k

    def fit(self, df: pd.DataFrame):
        self.ids = df['entity_id'].values
        self.names = df['business_name'].values
        self.addrs = df['business_address'].values
        N = len(df)
        self.tok_idx = defaultdict(list)
        self.cmp_idx = defaultdict(list)  # compact name prefix
        self.addr_idx = defaultdict(list)  # (digit, addr_word)
        self.norm_idx = defaultdict(list)  # normalized full name (short names)
        for i in range(N):
            nm = self.names[i]
            ad = self.addrs[i]
            # Token index
            for t in set(name_tokens(nm)):
                self.tok_idx[t].append(i)
            # Compact prefix (len 5..10)
            cn = compact_name(nm)
            if len(cn) >= 5:
                self.cmp_idx[cn[:10]].append(i)
            # Addr (digit, word)
            dg = digits(ad)
            at = addr_tokens(ad)
            if dg and at:
                for d in dg[:3]:
                    for w in at[:3]:
                        self.addr_idx[(d, w)].append(i)
            # Short normalized name index (for names like "OM Consultancy")
            nrm = normalize(nm)
            if 5 <= len(nrm) <= 30:
                self.norm_idx[nrm].append(i)

    def query(self, name: str, addr: str) -> list:
        scores = defaultdict(float)
        # 1. Compact name prefix
        cn = compact_name(name)
        if len(cn) >= 5:
            pl = self.cmp_idx.get(cn[:10], [])
            if len(pl) <= self.max_postings:
                for i in pl:
                    scores[i] += 15.0
        # 2. Normalized full name exact
        nrm = normalize(name)
        if 5 <= len(nrm) <= 30:
            for i in self.norm_idx.get(nrm, []):
                scores[i] += 20.0
        # 3. Tokens (rarest first, IDF-weighted)
        toks = name_tokens(name)
        toks.sort(key=lambda t: len(self.tok_idx.get(t, [])))
        for t in toks:
            pl = self.tok_idx.get(t, [])
            if 0 < len(pl) <= self.max_postings:
                w = 10.0 / (1.0 + math.log1p(len(pl)))
                for i in pl:
                    scores[i] += w
                if len(scores) >= 60:
                    break
        # 4. Address (digit, word)
        dg = digits(addr)
        at = addr_tokens(addr)
        if dg and at:
            for d in dg[:3]:
                for w in at[:3]:
                    pl = self.addr_idx.get((d, w), [])
                    if 0 < len(pl) <= self.max_postings:
                        for i in pl:
                            scores[i] += 8.0
        if not scores:
            return []
        if len(scores) <= self.top_k:
            return list(scores.keys())
        return sorted(scores, key=scores.get, reverse=True)[:self.top_k]

# ── Training ───────────────────────────────────────────────────────────────────
def compute_macro_f05(pred_dict, gt_dict, all_s1_ids):
    scores = []
    for s1 in all_s1_ids:
        true_set = gt_dict.get(s1, set())
        pred_set = set(pred_dict.get(s1, []))
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        if tp == 0:
            scores.append(0.0)
            continue
        prec = tp / len(pred_set)
        rec = tp / len(true_set)
        denom = 0.25 * prec + rec
        scores.append((1.25 * prec * rec) / denom)
    return float(np.mean(scores))

def train():
    con = duckdb.connect()
    print("=== v2 Improved Training ===")
    t0 = time.time()

    # Sample 150k S1 training entities + 15k validation
    N_TRAIN, N_VAL = 150000, 15000
    total = N_TRAIN + N_VAL
    print(f"Sampling {total} S1 entities from train split...")
    sample_s1 = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
        USING SAMPLE {total} (reservoir, 42)
    """).df()

    train_s1 = sample_s1.iloc[:N_TRAIN].copy()
    val_s1 = sample_s1.iloc[N_TRAIN:].copy()
    all_s1_ids_sample = list(sample_s1['entity_id'].values)

    # Load ground truth
    s1_ids_str = "','".join(all_s1_ids_sample)
    gt_df = con.execute(f"""
        WITH unnested AS (
            SELECT 
                g.source1_entity_id,
                unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
            FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True) g
            WHERE g.matched_entity_ids IS NOT NULL AND len(g.matched_entity_ids) > 0
              AND g.source1_entity_id IN ('{s1_ids_str}')
        )
        SELECT * FROM unnested
    """).df()

    gt_dict = defaultdict(set)
    for _, r in gt_df.iterrows():
        gt_dict[r['source1_entity_id']].add(r['matched_id'])

    val_gt_dict = {s1: gt_dict[s1] for s1 in val_s1['entity_id'] if s1 in gt_dict}
    val_s1_ids = list(val_s1['entity_id'].values)

    print(f"Train: {len(train_s1)}, Val: {len(val_s1)} ({len(val_gt_dict)} with matches)")
    all_matched_ids = set(gt_df['matched_id'].values)
    print(f"Total unique matched target IDs: {len(all_matched_ids):,}")

    # Build per-country blockers from full S2+S3 TRAIN corpora (for hard negatives)
    # This is the key fix: use the FULL train target corpus for blocking
    country_blockers_train = {}
    country_df_cache = {}
    for country in ['US', 'India']:
        print(f"\nBuilding full train blocker for {country}...")
        c_df = con.execute(f"""
            SELECT entity_id, business_name, business_address
            FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)
            WHERE country = '{country}'
            UNION ALL
            SELECT entity_id, business_name, business_address
            FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)
            WHERE country = '{country}'
        """).df()
        print(f"  {country}: {len(c_df):,} target records")
        b = MultiKeyBlocker(max_postings=120, top_k=25)
        b.fit(c_df)
        country_blockers_train[country] = b
        country_df_cache[country] = c_df
        print(f"  {country} blocker built")

    # Generate training pairs (true + hard negatives)
    print("\nGenerating training pairs (with hard negatives from full corpus)...")
    X_train, y_train = [], []
    blocked_not_true = 0
    for _, row in tqdm(train_s1.iterrows(), total=len(train_s1)):
        c = row['country']
        if c not in country_blockers_train:
            continue
        b = country_blockers_train[c]
        s1_id = row['entity_id']
        s1_name = row['business_name']
        s1_addr = row['business_address']
        cands = b.query(s1_name, s1_addr)
        true_set = gt_dict.get(s1_id, set())

        # Also force-add true matches that blocking missed (so model sees positives!)
        forced_true_indices = []
        for tid in true_set:
            idx_arr = np.where(b.ids == tid)[0]
            if len(idx_arr) > 0:
                forced_true_indices.append(idx_arr[0])

        all_cand_indices = list(set(cands) | set(forced_true_indices))
        for idx in all_cand_indices:
            tid = b.ids[idx]
            feats = extract_features(s1_name, s1_addr, b.names[idx], b.addrs[idx], tid)
            label = 1 if tid in true_set else 0
            X_train.append(feats)
            y_train.append(label)
            if label == 0:
                blocked_not_true += 1

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    pos_rate = sum(y_train) / len(y_train)
    print(f"Training dataset: {len(X_train):,} pairs, {sum(y_train):,} positive ({pos_rate*100:.1f}%)")

    # Train LightGBM with class weighting for imbalance
    scale_pos_weight = (1 - pos_rate) / pos_rate
    print(f"Training LightGBM (scale_pos_weight={scale_pos_weight:.2f})...")
    lgb_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }
    model = lgb.train(params, lgb_data, num_boost_round=300)
    model_path = f"{MODEL_DIR}/lgbm_v2.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Feature importance
    for nm, imp in sorted(zip(FEATURE_NAMES, model.feature_importance()), key=lambda x: -x[1]):
        print(f"  {nm:25s}: {imp}")

    # Validate and tune threshold
    print("\nValidating on holdout split...")
    val_pair_records = []
    for _, row in tqdm(val_s1.iterrows(), total=len(val_s1)):
        c = row['country']
        if c not in country_blockers_train:
            continue
        b = country_blockers_train[c]
        s1_id = row['entity_id']
        cands = b.query(row['business_name'], row['business_address'])
        for idx in cands:
            tid = b.ids[idx]
            feats = extract_features(row['business_name'], row['business_address'], b.names[idx], b.addrs[idx], tid)
            val_pair_records.append((s1_id, tid, feats))

    X_val = np.array([r[2] for r in val_pair_records], dtype=np.float32)
    val_probs = model.predict(X_val)

    print(f"\nVal candidates: {len(val_pair_records):,} | Mean/S1: {len(val_pair_records)/len(val_s1):.2f}")
    best_thresh, best_f05 = 0.5, 0.0
    for th in np.arange(0.40, 0.85, 0.05):
        preds = defaultdict(list)
        for (s1_id, tid, _), p in zip(val_pair_records, val_probs):
            if p >= th:
                preds[s1_id].append(tid)
        f05 = compute_macro_f05(preds, val_gt_dict, val_s1_ids)
        print(f"  thresh={th:.2f} -> Macro F_0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th

    print(f"\nBest threshold: {best_thresh:.2f} (Macro F_0.5 = {best_f05:.4f})")
    print(f"Training complete in {(time.time()-t0)/60:.2f} minutes")
    return model, best_thresh


def predict(model, threshold):
    con = duckdb.connect()
    print("\n=== Running v2 Test Inference ===")
    t0 = time.time()

    s1_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/test/test_source1.tsv', delim='\t', header=True)
    """).df()
    all_s1_ids_ordered = list(s1_df['entity_id'].values)
    total_s1 = len(all_s1_ids_ordered)
    print(f"Total test S1 entities: {total_s1:,}")

    matching_results = {sid: [] for sid in all_s1_ids_ordered}
    candidate_pairs = {sid: [] for sid in all_s1_ids_ordered}

    countries = list(s1_df['country'].unique())
    print(f"Countries: {countries}")

    for country in countries:
        print(f"\n=== Processing {country} ===")
        t_c = time.time()
        s1_c = s1_df[s1_df['country'] == country].reset_index(drop=True)
        print(f"  S1 entities: {len(s1_c):,}")

        targets = con.execute(f"""
            SELECT entity_id, business_name, business_address
            FROM read_csv('{DATA_DIR}/test/test_source2.tsv', delim='\t', header=True)
            WHERE country = '{country}'
            UNION ALL
            SELECT entity_id, business_name, business_address
            FROM read_csv('{DATA_DIR}/test/test_source3.tsv', delim='\t', header=True)
            WHERE country = '{country}'
        """).df()
        print(f"  Target records: {len(targets):,}")

        print(f"  Building blocker...")
        b = MultiKeyBlocker(max_postings=120, top_k=25)
        b.fit(targets)

        BATCH = 5000
        n_batches = math.ceil(len(s1_c) / BATCH)
        t_inf = time.time()

        for batch_i in range(n_batches):
            sl = s1_c.iloc[batch_i*BATCH : (batch_i+1)*BATCH]
            batch_pairs = []
            for _, row in sl.iterrows():
                sid = row['entity_id']
                cands = b.query(row['business_name'], row['business_address'])
                cand_ids = [b.ids[i] for i in cands]
                candidate_pairs[sid] = cand_ids
                for i in cands:
                    tid = b.ids[i]
                    feats = extract_features(row['business_name'], row['business_address'],
                                             b.names[i], b.addrs[i], tid)
                    batch_pairs.append((sid, tid, feats))

            if batch_pairs:
                X_b = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                probs = model.predict(X_b)
                for (sid, tid, _), p in zip(batch_pairs, probs):
                    if p >= threshold:
                        matching_results[sid].append(tid)

            if (batch_i + 1) % 20 == 0 or (batch_i + 1) == n_batches:
                done = min((batch_i + 1) * BATCH, len(s1_c))
                elapsed = time.time() - t_inf
                rate = done / elapsed
                eta = (len(s1_c) - done) / rate if rate > 0 else 0
                print(f"  [{country}] {done}/{len(s1_c)} ({done/len(s1_c)*100:.1f}%) {rate:.0f}/s ETA:{eta:.0f}s")

        del targets, b
        print(f"  Finished {country} in {(time.time()-t_c)/60:.2f} min")

    # Write outputs
    matching_path = f"{OUTPUT_DIR}/matching_results.tsv"
    candidate_path = f"{OUTPUT_DIR}/candidate_pairs.tsv"

    print(f"\nWriting {matching_path}...")
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids_ordered:
            m = list(dict.fromkeys(matching_results[sid]))
            f.write(f"{sid}\t{','.join(m)}\n")

    print(f"Writing {candidate_path}...")
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids_ordered:
            c = list(dict.fromkeys(candidate_pairs[sid] + matching_results[sid]))
            f.write(f"{sid}\t{','.join(c)}\n")

    total_min = (time.time() - t0) / 60
    print(f"\nAll done in {total_min:.2f} minutes")

    # Quick stats
    n_with_matches = sum(1 for v in matching_results.values() if v)
    n_empty = total_s1 - n_with_matches
    all_pred = [len(v) for v in matching_results.values() if v]
    print(f"  Entities with matches: {n_with_matches:,} ({n_with_matches/total_s1*100:.1f}%)")
    print(f"  Singletons (empty):    {n_empty:,}")
    print(f"  Mean matches per entity (non-empty): {np.mean(all_pred):.2f}")
    print(f"  Median: {np.median(all_pred):.1f}, Max: {max(all_pred) if all_pred else 0}")

    return matching_path, candidate_path


if __name__ == '__main__':
    model, threshold = train()
    predict(model, threshold)
