import duckdb
import time
import re
from unidecode import unidecode
from collections import defaultdict, Counter
import numpy as np

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Selective High-Precision Blocking Keys ---")
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

N_TARGETS = len(us_targets)
t_idx = time.time()

t_ids = us_targets['entity_id'].values
t_names = us_targets['business_name'].values
t_addrs = us_targets['business_address'].values

# Build raw index first
name_index = defaultdict(list)
compact_index = defaultdict(list)
addr_index = defaultdict(list)

for i in range(N_TARGETS):
    name = t_names[i]
    addr = t_addrs[i]
    
    # 1. Name tokens
    toks = set(clean_tokens(name))
    for t in toks:
        name_index[t].append(i)
        
    # 2. Compact name prefix (len >= 6)
    cname = clean_compact_name(name)
    if len(cname) >= 6:
        compact_index[cname[:10]].append(i)
        
    # 3. Addr: (digit, word)
    digs = extract_digits(addr)
    atoks = extract_addr_tokens(addr)
    if digs and atoks:
        for d in digs[:2]:
            for w in atoks[:2]:
                addr_index[(d, w)].append(i)

print(f"Raw indexes built in {time.time() - t_idx:.2f}s.")

# Now test different frequency caps!
for max_postings in [50, 100, 200]:
    print(f"\n--- Testing with MAX_POSTINGS = {max_postings} ---")
    recalled_pairs = 0
    total_candidates_stored = 0
    cand_sizes = []
    
    for _, row in val_s1_us.iterrows():
        s1_id = row['entity_id']
        name = row['business_name']
        addr = row['business_address']
        
        cands = set()
        
        # 1. Compact name
        cname = clean_compact_name(name)
        if len(cname) >= 6:
            postings = compact_index.get(cname[:10], [])
            if len(postings) <= max_postings:
                cands.update(postings)
                
        # 2. Name tokens
        toks = clean_tokens(name)
        # Sort tokens by posting list length (rarest first!)
        toks.sort(key=lambda t: len(name_index.get(t, [])))
        for t in toks:
            postings = name_index.get(t, [])
            if 0 < len(postings) <= max_postings:
                cands.update(postings)
                if len(cands) >= 30: # early stop when we have enough high quality candidates
                    break
                    
        # 3. Addr: (digit, word)
        digs = extract_digits(addr)
        atoks = extract_addr_tokens(addr)
        if digs and atoks:
            for d in digs[:2]:
                for w in atoks[:2]:
                    postings = addr_index.get((d, w), [])
                    if 0 < len(postings) <= max_postings:
                        cands.update(postings)
                        
        cand_sizes.append(len(cands))
        total_candidates_stored += len(cands)
        
        # Check recall
        true_for_s1 = gt_dict.get(s1_id, set())
        recalled_pairs += len(true_for_s1 & {t_ids[idx] for idx in cands})
        
    mean_cand = np.mean(cand_sizes)
    med_cand = np.median(cand_sizes)
    p90_cand = np.percentile(cand_sizes, 90)
    p99_cand = np.percentile(cand_sizes, 99)
    recall = recalled_pairs / len(true_matches) * 100
    print(f"Recall: {recalled_pairs}/{len(true_matches)} ({recall:.2f}%)")
    print(f"Candidates per S1 -> Mean: {mean_cand:.2f}, Median: {med_cand:.1f}, P90: {p90_cand:.1f}, P99: {p99_cand:.1f}, Max: {max(cand_sizes)}")
