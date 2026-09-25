import duckdb
import time
import re
from unidecode import unidecode
from collections import defaultdict

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Blocking Coverage on Ground Truth ---")
t0 = time.time()

# Let's take a sample of 20,000 S1 records from train
s1_df = con.execute(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    USING SAMPLE 20000 (reservoir, 123);
""").df()

s1_ids_set = set(s1_df['entity_id'])

# Get their ground truth matches
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

# Build true match dictionary: s1_id -> set of matched_ids
gt_dict = defaultdict(set)
for _, r in gt_df.iterrows():
    gt_dict[r['source1_entity_id']].add(r['matched_id'])

all_true_matches = set((r['source1_entity_id'], r['matched_id']) for _, r in gt_df.iterrows())
total_true_pairs = len(all_true_matches)
print(f"Sampled {len(s1_df)} S1 entities. Found {len(gt_dict)} entities with matches, total {total_true_pairs} true pairs.")

# Now load the corresponding S2 and S3 rows for the sample
# In real blocking, we index ALL S2 and S3 of that country.
# Let's see what happens if we test blocking on the true pairs first to see what fraction share various keys.

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

print(f"Loaded {len(target_dict)} target records for the ground truth pairs.")

# Common stop words/suffixes across English, Indian, French business names
LEGAL_TERMS = {
    'ltd', 'limited', 'pvt', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'services', 'service', 'enterprises', 'enterprise',
    'technologies', 'technology', 'tech', 'solutions', 'solution', 'consulting', 'consultants',
    'holdings', 'holding', 'group', 'international', 'global', 'india', 'us', 'usa',
    'sarl', 'sas', 'sci', 'sa', 'eurl', 'france', 'fr', 'sri', 'shri', 'ms', 'm/s', 'the',
    'and', 'associates', 'industries', 'industry', 'management', 'center', 'centre'
}

def clean_name(s):
    if not s or str(s) == 'nan':
        return ""
    s = unidecode(str(s)).lower()
    # Replace non-alphanumeric with space
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [w for w in s.split() if len(w) > 1 and w not in LEGAL_TERMS]
    return tokens

def clean_norm_str(s):
    if not s or str(s) == 'nan':
        return ""
    s = unidecode(str(s)).lower()
    return re.sub(r'[^a-z0-9]', '', s)

def extract_addr_tokens(s):
    if not s or str(s) == 'nan':
        return set()
    s = unidecode(str(s)).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Return set of tokens of length >= 2
    return set(w for w in s.split() if len(w) >= 2 and w not in {'road', 'street', 'drive', 'lane', 'avenue', 'rd', 'st', 'dr', 'ln', 'ave', 'blvd', 'near', 'opp', 'floor', 'unit', 'apt', 'null', 'nan'})

def extract_digits(s):
    if not s or str(s) == 'nan':
        return set()
    # Extract numbers like 1490, 5807, 31415
    return set(re.findall(r'\b\d+\b', str(s)))

# Now analyze coverage on all true pairs
matched_by_token_overlap = 0
matched_by_first_token = 0
matched_by_any_rare_token = 0
matched_by_addr_digit_and_token = 0
matched_by_either = 0
unmatched_pairs = []

for s1_id, tgt_id in all_true_matches:
    s1_row = s1_row_dict[s1_id]
    tgt_row = target_dict[tgt_id]
    
    s1_tokens = set(clean_name(s1_row['business_name']))
    tgt_tokens = set(clean_name(tgt_row['business_name']))
    
    s1_digits = extract_digits(s1_row['business_address'])
    tgt_digits = extract_digits(tgt_row['business_address'])
    
    s1_addr_tokens = extract_addr_tokens(s1_row['business_address'])
    tgt_addr_tokens = extract_addr_tokens(tgt_row['business_address'])
    
    name_overlap = bool(s1_tokens & tgt_tokens)
    norm_name_match = (clean_norm_str(s1_row['business_name']) == clean_norm_str(tgt_row['business_name']))
    
    # Check digits match and address word match
    digit_match = bool(s1_digits & tgt_digits)
    addr_match = bool(s1_addr_tokens & tgt_addr_tokens)
    addr_both = digit_match and addr_match
    
    if name_overlap or norm_name_match:
        matched_by_token_overlap += 1
    if addr_both:
        matched_by_addr_digit_and_token += 1
    if name_overlap or norm_name_match or addr_both:
        matched_by_either += 1
    else:
        unmatched_pairs.append((s1_row, tgt_row))

print(f"\nTotal True Pairs: {total_true_pairs}")
print(f"Covered by Clean Name Token Overlap / Norm Name: {matched_by_token_overlap} ({matched_by_token_overlap / total_true_pairs * 100:.2f}%)")
print(f"Covered by Address (Matching Digit + Address Token): {matched_by_addr_digit_and_token} ({matched_by_addr_digit_and_token / total_true_pairs * 100:.2f}%)")
print(f"Covered by EITHER Name Token OR Address (Digit+AddrToken): {matched_by_either} ({matched_by_either / total_true_pairs * 100:.2f}%)")
print(f"Unmatched: {len(unmatched_pairs)} ({len(unmatched_pairs) / total_true_pairs * 100:.2f}%)")

print("\n--- Inspecting first 10 Unmatched Pairs ---")
for s1, tgt in unmatched_pairs[:10]:
    print(f"[{s1['country']}] S1: {s1['business_name']} | Addr: {s1['business_address']}")
    print(f"       Tgt: {tgt['business_name']} | Addr: {tgt['business_address']}")
    print("-" * 60)
