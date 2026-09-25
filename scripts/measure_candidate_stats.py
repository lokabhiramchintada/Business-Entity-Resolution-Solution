import duckdb
import time

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Scalable Blocking with Inverted Index & Frequency Capping ---")
t0 = time.time()

# Let's hold out 20,000 S1 records for validation: 10k US, 10k India
val_s1 = con.execute(f"""
    WITH us_sample AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
        WHERE country = 'US'
        USING SAMPLE 10000 (reservoir, 100)
    ),
    in_sample AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
        WHERE country = 'India'
        USING SAMPLE 10000 (reservoir, 100)
    )
    SELECT * FROM us_sample UNION ALL SELECT * FROM in_sample;
""").df()

val_s1_ids = set(val_s1['entity_id'])
print(f"Validation set: {len(val_s1)} S1 entities.")

# Get ground truth for validation entities
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

val_true_pairs = set((r['source1_entity_id'], r['matched_id']) for _, r in gt_val.iterrows())
val_entities_with_matches = len(set(r['source1_entity_id'] for _, r in gt_val.iterrows()))
print(f"Validation entities with matches: {val_entities_with_matches} / {len(val_s1)}. Total true pairs: {len(val_true_pairs)}.")
print(f"Setup time: {time.time() - t0:.2f}s")
