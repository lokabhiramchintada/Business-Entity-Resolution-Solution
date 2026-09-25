import duckdb

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

query = f"""
    WITH s1_sample AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
        USING SAMPLE 10000 (reservoir, 42)
    ),
    gt_sample AS (
        SELECT 
            g.source1_entity_id,
            unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True) g
        JOIN s1_sample s ON g.source1_entity_id = s.entity_id
        WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != ''
    ),
    s2_all AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)
    ),
    s3_all AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)
    ),
    s_target AS (
        SELECT * FROM s2_all WHERE entity_id IN (SELECT matched_id FROM gt_sample)
        UNION ALL
        SELECT * FROM s3_all WHERE entity_id IN (SELECT matched_id FROM gt_sample)
    )
    SELECT 
        g.source1_entity_id,
        s1.business_name AS s1_name,
        s1.business_address AS s1_addr,
        s1.country,
        t.entity_id AS target_id,
        t.business_name AS target_name,
        t.business_address AS target_addr
    FROM gt_sample g
    JOIN s1_sample s1 ON g.source1_entity_id = s1.entity_id
    JOIN s_target t ON g.matched_id = t.entity_id
    WHERE regexp_replace(lower(s1.business_name), '[^a-z0-9]', '', 'g') != regexp_replace(lower(t.business_name), '[^a-z0-9]', '', 'g')
      AND regexp_replace(lower(t.business_name), '[^a-z0-9]', '', 'g') NOT LIKE '%' || regexp_replace(lower(s1.business_name), '[^a-z0-9]', '', 'g') || '%'
      AND regexp_replace(lower(s1.business_name), '[^a-z0-9]', '', 'g') NOT LIKE '%' || regexp_replace(lower(t.business_name), '[^a-z0-9]', '', 'g') || '%'
    LIMIT 40;
"""

df_diff = con.execute(query).df()
for i, row in df_diff.iterrows():
    print(f"[{row['country']}] S1 ({row['source1_entity_id']}): {row['s1_name']} || Addr: {row['s1_addr']}")
    print(f"       -> Match ({row['target_id']}): {row['target_name']} || Addr: {row['target_addr']}")
    print("-" * 80)
