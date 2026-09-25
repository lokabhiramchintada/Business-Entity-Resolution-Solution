import duckdb
import time
import re
from unidecode import unidecode
from collections import defaultdict

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Enhanced Blocking Coverage ---")
t0 = time.time()

# Let's take 20,000 S1 records from train
s1_df = con.execute(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    USING SAMPLE 20000 (reservoir, 123);
""").df()

con.register('sample_s1_table', s1_df[['entity_id']])
gt_df = con.execute(f"""
    WITH unnested AS (
        SELECT 
            g.source1_entity_id,
            unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True) g
        JOIN sample_s1_table s ON g.source1_entity_id = s.entity_id
        WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != ''
    )
    SELECT * FROM unnested;
""").df()

all_true_matches = set((r['source1_entity_id'], r['matched_id']) for _, r in gt_df.iterrows())
total_true_pairs = len(all_true_matches)
print(f"Sampled {len(s1_df)} S1 entities. Total {total_true_pairs} true pairs.")

con.register('sample_gt_table', gt_df[['matched_id']])
target_df = con.execute(f"""
    WITH targets AS (SELECT DISTINCT matched_id FROM sample_gt_table),
    s2 AS (SELECT entity_id, business_name, business_address, country FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)),
    s3 AS (SELECT entity_id, business_name, business_address, country FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)),
    s_both AS (SELECT * FROM s2 UNION ALL SELECT * FROM s3)
    SELECT s.* FROM s_both s JOIN targets t ON s.entity_id = t.matched_id;
""").df()

target_dict = {r['entity_id']: r for _, r in target_df.iterrows()}
s1_row_dict = {r['entity_id']: r for _, r in s1_df.iterrows()}

LEGAL_TERMS = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'services', 'service', 'enterprises', 'enterprise',
    'technologies', 'technology', 'tech', 'solutions', 'solution', 'consulting', 'consultants',
    'holdings', 'holding', 'group', 'international', 'global', 'india', 'us', 'usa',
    'sarl', 'sas', 'sci', 'sa', 'eurl', 'france', 'fr', 'sri', 'shri', 'ms', 'm/s', 'the',
    'and', 'associates', 'industries', 'industry', 'management', 'center', 'centre', 'com', 'org', 'net'
}

GENERIC_ADDR_WORDS = {
    'road', 'street', 'drive', 'lane', 'avenue', 'court', 'circle', 'parkway', 'highway', 'way', 'place',
    'rd', 'st', 'dr', 'ln', 'ave', 'ct', 'cir', 'pkwy', 'hwy', 'pl', 'blvd', 'boulevard',
    'near', 'opp', 'floor', 'unit', 'apt', 'null', 'nan', 'house', 'hno', 'plot', 'flat', 'no',
    'floor', 'block', 'phase', 'nagar', 'colony', 'bldg', 'building', 'tower', 'complex',
    'north', 'south', 'east', 'west', 'upper', 'lower', 'suite', 'ste', 'room', 'rm', 'box', 'pob',
    'c/o', 'w/o', 's/o', 'd/o', 'village', 'vill', 'dist', 'district', 'post', 'po', 'tq', 'taluka',
    'state', 'city', 'town', 'layout', 'enclave', 'park', 'road', 'gali', 'marg', 'rasta'
}

def clean_name_tokens(s):
    if not s or str(s) == 'nan':
        return set()
    s = unidecode(str(s)).lower()
    # Strip URL extensions
    s = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [w for w in s.split() if len(w) > 1 and w not in LEGAL_TERMS]
    return set(tokens)

def clean_name_compact(s):
    if not s or str(s) == 'nan':
        return ""
    s = unidecode(str(s)).lower()
    s = re.sub(r'\.(com|org|net|in|co|us|fr)\b', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    # Remove legal suffixes from compact string
    for term in sorted(LEGAL_TERMS, key=len, reverse=True):
        if s.endswith(term):
            s = s[:-len(term)]
    return s

def extract_clean_digits(s):
    if not s or str(s) == 'nan':
        return set()
    digits = re.findall(r'\b\d+\b', str(s))
    # Normalize by stripping leading zeros
    return set(d.lstrip('0') for d in digits if d.lstrip('0'))

def extract_addr_clean_tokens(s):
    if not s or str(s) == 'nan':
        return set()
    s = unidecode(str(s)).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [w for w in s.split() if len(w) >= 3 and w not in GENERIC_ADDR_WORDS and not w.isdigit()]
    return set(tokens)

matched_count = 0
unmatched = []

for s1_id, tgt_id in all_true_matches:
    s1_row = s1_row_dict[s1_id]
    tgt_row = target_dict[tgt_id]
    
    s1_nt = clean_name_tokens(s1_row['business_name'])
    tgt_nt = clean_name_tokens(tgt_row['business_name'])
    
    s1_nc = clean_name_compact(s1_row['business_name'])
    tgt_nc = clean_name_compact(tgt_row['business_name'])
    
    s1_dig = extract_clean_digits(s1_row['business_address'])
    tgt_dig = extract_clean_digits(tgt_row['business_address'])
    
    s1_at = extract_addr_clean_tokens(s1_row['business_address'])
    tgt_at = extract_addr_clean_tokens(tgt_row['business_address'])
    
    # 1. Exact or token name overlap
    name_tok_match = bool(s1_nt & tgt_nt)
    
    # 2. Compact substring match (e.g. lifeengineering vs life engineering)
    compact_sub = (len(s1_nc) >= 4 and len(tgt_nc) >= 4 and (s1_nc in tgt_nc or tgt_nc in s1_nc))
    
    # 3. Address match:
    # Digits overlap AND at least 1 address word overlap
    addr_dig_word = bool(s1_dig & tgt_dig) and bool(s1_at & tgt_at)
    
    # Or 2 address words overlap (even without digits, e.g. "melrose" + "aligarh" or "neshar" + "nautan")
    addr_two_words = len(s1_at & tgt_at) >= 2
    
    if name_tok_match or compact_sub or addr_dig_word or addr_two_words:
        matched_count += 1
    else:
        unmatched.append((s1_row, tgt_row))

print(f"Total True Pairs: {total_true_pairs}")
print(f"Covered: {matched_count} ({matched_count / total_true_pairs * 100:.2f}%)")
print(f"Unmatched: {len(unmatched)} ({len(unmatched) / total_true_pairs * 100:.2f}%)")

print("\n--- Remaining Unmatched Samples (first 10) ---")
for s1, tgt in unmatched[:10]:
    print(f"[{s1['country']}] S1: {s1['business_name']} | Addr: {s1['business_address']}")
    print(f"       Tgt: {tgt['business_name']} | Addr: {tgt['business_address']}")
    print("-" * 60)
