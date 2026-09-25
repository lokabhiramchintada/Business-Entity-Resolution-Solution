import duckdb
import time
import re
import math
from unidecode import unidecode
from collections import defaultdict, Counter
import numpy as np

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Top-K Candidate Selection with Fast Scoring ---")
t0 = time.time()

# 1. Validation S1 sample (3,000 US entities)
val_s1_us = con.execute(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    WHERE country = 'US'
    USING SAMPLE 3000 (reservoir, 42);
""").df()

con.register('val_s1_df', val_s1_us[['entity_id']])
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

true_matches = set((r['source1_entity_id'], r['matched_id']) for _, r in gt_val.iterrows())
gt_dict = defaultdict(set)
for s1, tgt in true_matches:
    gt_dict[s1].add(tgt)

print(f"US Validation S1 entities: {len(val_s1_us)}, true match pairs: {len(true_matches)}")

# Load US targets
print("Loading US targets...")
us_targets = con.execute(f"""
    SELECT entity_id, business_name, business_address 
    FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)
    WHERE country = 'US'
    UNION ALL
    SELECT entity_id, business_name, business_address 
    FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)
    WHERE country = 'US';
""").df()
print(f"Loaded {len(us_targets)} US target records.")

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

# Pre-extract target features
N_TARGETS = len(us_targets)
t_idx = time.time()

# Token document frequency
df_name = Counter()
for name in us_targets['business_name']:
    toks = set(clean_tokens(name))
    for t in toks:
        df_name[t] += 1

print(f"Computed doc frequencies for {len(df_name)} tokens.")

# Inverted index: only include tokens with df <= 5000 (skip ultra-frequent stop words)
name_index = defaultdict(list)
compact_index = defaultdict(list)
addr_index = defaultdict(list)

# We map target entity_id to an integer ID 0..N-1 for fast indexing
t_ids = us_targets['entity_id'].values
t_names = us_targets['business_name'].values
t_addrs = us_targets['business_address'].values

for i in range(N_TARGETS):
    name = t_names[i]
    addr = t_addrs[i]
    
    # Name tokens
    toks = set(clean_tokens(name))
    for t in toks:
        if df_name[t] <= 5000:
            name_index[t].append(i)
            
    # Compact name prefix (len >= 4)
    cname = clean_compact_name(name)
    if len(cname) >= 4:
        compact_index[cname[:10]].append(i)
        
    # Address: (digit, word) with df check
    digs = extract_digits(addr)
    atoks = extract_addr_tokens(addr)
    if digs and atoks:
        d0 = digs[0]
        for w in atoks[:2]:
            addr_index[(d0, w)].append(i)

print(f"Indexes built in {time.time() - t_idx:.2f}s.")

# Precompute IDF for tokens
idf_dict = {t: math.log(N_TARGETS / (1 + count)) for t, count in df_name.items()}

# Now for each validation S1, score candidates using fast accumulators
t_query = time.time()

for K in [5, 10, 15, 20]:
    recalled_pairs = 0
    total_candidates_stored = 0
    
    for _, row in val_s1_us.iterrows():
        s1_id = row['entity_id']
        name = row['business_name']
        addr = row['business_address']
        
        # Accumulate candidate scores
        cand_scores = defaultdict(float)
        
        # 1. Compact name prefix (high weight)
        cname = clean_compact_name(name)
        if len(cname) >= 4:
            for tid_idx in compact_index.get(cname[:10], []):
                cand_scores[tid_idx] += 10.0
                
        # 2. Name tokens weighted by IDF
        s1_toks = set(clean_tokens(name))
        for t in s1_toks:
            if df_name.get(t, 0) <= 5000:
                w = idf_dict.get(t, 2.0)
                for tid_idx in name_index.get(t, []):
                    cand_scores[tid_idx] += w
                    
        # 3. Address keys (digit + addr word)
        digs = extract_digits(addr)
        atoks = extract_addr_tokens(addr)
        if digs and atoks:
            for d in digs[:2]:
                for w in atoks[:2]:
                    for tid_idx in addr_index.get((d, w), []):
                        cand_scores[tid_idx] += 5.0
                        
        if not cand_scores:
            continue
            
        # Top-K selection
        if len(cand_scores) <= K:
            top_k_indices = list(cand_scores.keys())
        else:
            # Sort top K
            top_k_indices = sorted(cand_scores, key=cand_scores.get, reverse=True)[:K]
            
        top_k_tids = {t_ids[idx] for idx in top_k_indices}
        total_candidates_stored += len(top_k_tids)
        
        # Check recall
        true_for_s1 = gt_dict.get(s1_id, set())
        recalled_pairs += len(true_for_s1 & top_k_tids)
        
    mean_cand = total_candidates_stored / len(val_s1_us)
    recall = recalled_pairs / len(true_matches) * 100
    print(f"Top-{K}: Recall = {recalled_pairs}/{len(true_matches)} ({recall:.2f}%), Mean Candidates/S1 = {mean_cand:.2f}")

print(f"Total time: {time.time() - t_query:.2f}s")
