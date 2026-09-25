import duckdb
import time

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Testing Blocking Key Coverage on Ground Truth Sample (100,000 S1 entities) ---")
t0 = time.time()

# Sample 50,000 S1 entities from train
con.execute(f"""
    CREATE TEMPORARY TABLE s1_sample AS 
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    USING SAMPLE 50000 (reservoir, 42);
""")

con.execute(f"""
    CREATE TEMPORARY TABLE gt_sample AS
    SELECT 
        g.source1_entity_id,
        unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
    FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True) g
    JOIN s1_sample s ON g.source1_entity_id = s.entity_id
    WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != '';
""")

total_gt_pairs = con.execute("SELECT COUNT(*) FROM gt_sample").fetchone()[0]
total_s1 = con.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM gt_sample").fetchone()[0]
print(f"Sample contains {total_s1} S1 entities with matches, total {total_gt_pairs} ground truth pairs.")

# Load matching S2 and S3 records for these pairs
con.execute(f"""
    CREATE TEMPORARY TABLE s2_all AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True);
    
    CREATE TEMPORARY TABLE s3_all AS
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True);
    
    CREATE TEMPORARY TABLE s_target AS
    SELECT * FROM s2_all WHERE entity_id IN (SELECT matched_id FROM gt_sample)
    UNION ALL
    SELECT * FROM s3_all WHERE entity_id IN (SELECT matched_id FROM gt_sample);
""")

print("Loaded target matches. Now analyzing patterns...")

# Join S1 and target to analyze relationship
con.execute(f"""
    CREATE TEMPORARY TABLE pair_details AS
    SELECT 
        g.source1_entity_id,
        g.matched_id,
        s1.country,
        s1.business_name AS s1_name,
        t.business_name AS target_name,
        s1.business_address AS s1_addr,
        t.business_address AS target_addr,
        -- normalized names: lowercase, alphanumeric only
        regexp_replace(lower(s1.business_name), '[^a-z0-9]', '', 'g') AS s1_norm_name,
        regexp_replace(lower(t.business_name), '[^a-z0-9]', '', 'g') AS target_norm_name,
        -- normalized addresses: lowercase, alphanumeric only
        regexp_replace(lower(coalesce(s1.business_address, '')), '[^a-z0-9]', '', 'g') AS s1_norm_addr,
        regexp_replace(lower(coalesce(t.business_address, '')), '[^a-z0-9]', '', 'g') AS target_norm_addr
    FROM gt_sample g
    JOIN s1_sample s1 ON g.source1_entity_id = s1.entity_id
    JOIN s_target t ON g.matched_id = t.entity_id;
""")

stats = con.execute("""
    SELECT 
        COUNT(*) as total_pairs,
        COUNT(CASE WHEN s1_name = target_name THEN 1 END) as exact_raw_name,
        COUNT(CASE WHEN s1_norm_name = target_norm_name THEN 1 END) as exact_norm_name,
        COUNT(CASE WHEN s1_norm_name != '' AND (target_norm_name LIKE '%' || s1_norm_name || '%' OR s1_norm_name LIKE '%' || target_norm_name || '%') THEN 1 END) as substring_norm_name,
        COUNT(CASE WHEN s1_addr = target_addr THEN 1 END) as exact_raw_addr,
        COUNT(CASE WHEN s1_norm_addr != '' AND s1_norm_addr = target_norm_addr THEN 1 END) as exact_norm_addr,
        COUNT(CASE WHEN s1_norm_name = target_norm_name OR s1_norm_addr = target_norm_addr THEN 1 END) as either_exact_norm
    FROM pair_details
""").df()

print("\n--- Match Stats ---")
print(stats.to_string())

print(f"\nElapsed: {time.time() - t0:.2f}s")
