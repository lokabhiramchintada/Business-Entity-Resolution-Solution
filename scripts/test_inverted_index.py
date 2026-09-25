import duckdb
import time
import re
from unidecode import unidecode
from collections import defaultdict
import numpy as np

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Inverted Index Blocking on Full S2 & S3 Targets ---")
t0 = time.time()

# 1. Validation S1 sample (10,000 entities)
val_s1 = con.execute(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    USING SAMPLE 10000 (reservoir, 42);
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

true_matches = set((r['source1_entity_id'], r['matched_id']) for _, r in gt_val.iterrows())
print(f"Validation: {len(val_s1)} S1 entities, {len(true_matches)} true match pairs.")

# Common legal terms
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

# Test indexing on US subset first to profile speed and recall
print("\n--- Testing on US subset ---")
val_s1_us = val_s1[val_s1['country'] == 'US']
us_true_matches = set((s1, tgt) for s1, tgt in true_matches if s1 in set(val_s1_us['entity_id']))
print(f"US Validation S1 entities: {len(val_s1_us)}, US true match pairs: {len(us_true_matches)}")

# Load US targets from train_source2 and train_source3 (total ~6.2M records)
t_load = time.time()
print("Loading US targets from train_source2 and train_source3...")
us_targets = con.execute(f"""
    SELECT entity_id, business_name, business_address 
    FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)
    WHERE country = 'US'
    UNION ALL
    SELECT entity_id, business_name, business_address 
    FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)
    WHERE country = 'US';
""").df()
print(f"Loaded {len(us_targets)} US target records in {time.time() - t_load:.2f}s.")

# Build inverted indexes on US targets:
# 1. name_token -> list of target_ids
# 2. compact_name -> list of target_ids
# 3. (house_digit, addr_word) -> list of target_ids

t_idx = time.time()
name_index = defaultdict(list)
compact_index = defaultdict(list)
addr_index = defaultdict(list)

for idx, row in us_targets.iterrows():
    tid = row['entity_id']
    name = row['business_name']
    addr = row['business_address']
    
    # 1. Tokens
    tokens = clean_tokens(name)
    for tok in set(tokens):
        name_index[tok].append(tid)
        
    # 2. Compact name
    cname = clean_compact_name(name)
    if len(cname) >= 4:
        compact_index[cname[:12]].append(tid)
        
    # 3. Addr: (digit, word)
    digs = extract_digits(addr)
    atoks = extract_addr_tokens(addr)
    if digs and atoks:
        d0 = digs[0]
        for w in atoks[:2]:
            addr_index[(d0, w)].append(tid)

print(f"Index built in {time.time() - t_idx:.2f}s.")
print(f"Unique name tokens in index: {len(name_index)}")
print(f"Unique compact prefixes: {len(compact_index)}")
print(f"Unique (digit, addr_word) keys: {len(addr_index)}")

# Frequency filter for name tokens:
# If a token appears in > 1000 targets, it's too frequent to block on as a single word!
FREQ_CAP = 1000
frequent_tokens = set(tok for tok, lst in name_index.items() if len(lst) > FREQ_CAP)
print(f"Number of frequent name tokens (> {FREQ_CAP} occurrences): {len(frequent_tokens)}")

# Now generate candidates for each US validation S1 record
t_cand = time.time()
val_candidates = {}
total_candidates_count = 0
found_true_count = 0

for _, row in val_s1_us.iterrows():
    s1_id = row['entity_id']
    name = row['business_name']
    addr = row['business_address']
    
    cands = set()
    
    # 1. Query compact name prefix
    cname = clean_compact_name(name)
    if len(cname) >= 4:
        cands.update(compact_index.get(cname[:12], []))
        
    # 2. Query rare name tokens (sorted by rarest first)
    tokens = [t for t in set(clean_tokens(name)) if t not in frequent_tokens]
    if tokens:
        # Sort by frequency in index so we pick the most discriminative tokens
        tokens.sort(key=lambda t: len(name_index.get(t, [])))
        for t in tokens[:2]: # top 2 rarest tokens
            cands.update(name_index.get(t, []))
            
    # 3. Query (digit, addr_word)
    digs = extract_digits(addr)
    atoks = extract_addr_tokens(addr)
    if digs and atoks:
        for d in digs[:2]:
            for w in atoks[:2]:
                cands.update(addr_index.get((d, w), []))
                
    # If candidate set is too big (e.g. > 100), we can cap it, but let's see its raw size
    val_candidates[s1_id] = cands
    total_candidates_count += len(cands)

elapsed_cand = time.time() - t_cand
cand_sizes = [len(cands) for cands in val_candidates.values()]
print(f"\nCandidate generation finished in {elapsed_cand:.2f}s ({elapsed_cand/len(val_s1_us)*1000:.2f} ms/entity).")
print(f"Mean candidates per S1 entity: {np.mean(cand_sizes):.2f}")
print(f"Median candidates per S1 entity: {np.median(cand_sizes):.2f}")
print(f"90th percentile candidates: {np.percentile(cand_sizes, 90):.2f}")
print(f"Max candidates: {np.max(cand_sizes)}")
print(f"S1 entities with 0 candidates: {sum(1 for s in cand_sizes if s == 0)} / {len(val_s1_us)}")

# Evaluate Recall on US true matches
for s1_id, tgt_id in us_true_matches:
    if tgt_id in val_candidates.get(s1_id, set()):
        found_true_count += 1

print(f"\n--- Ground Truth Coverage on US Targets ---")
print(f"Total True Pairs: {len(us_true_matches)}")
print(f"Found in Candidate Sets: {found_true_count} ({found_true_count / len(us_true_matches) * 100:.2f}%)")
print(f"Missed: {len(us_true_matches) - found_true_count} ({(len(us_true_matches) - found_true_count) / len(us_true_matches) * 100:.2f}%)")
