import duckdb

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

query = f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
        LIMIT 200
    ),
    s1 AS (
        SELECT entity_id, business_name as s1_name, business_address as s1_addr, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    ),
    s2 AS (
        SELECT entity_id, business_name as s_name, business_address as s_addr, country
        FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True)
    ),
    s3 AS (
        SELECT entity_id, business_name as s_name, business_address as s_addr, country
        FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True)
    ),
    s_all AS (
        SELECT * FROM s2 UNION ALL SELECT * FROM s3
    )
    SELECT 
        u.source1_entity_id,
        s1.s1_name,
        s1.s1_addr,
        s1.country,
        u.matched_id,
        s_all.s_name,
        s_all.s_addr
    FROM unnested u
    JOIN s1 ON u.source1_entity_id = s1.entity_id
    JOIN s_all ON u.matched_id = s_all.entity_id
    LIMIT 30;
"""

df_sample = con.execute(query).df()
for i, row in df_sample.iterrows():
    print(f"[{row['country']}] S1: {row['source1_entity_id']} | Name: {row['s1_name']} | Addr: {row['s1_addr']}")
    print(f"       -> Match: {row['matched_id']} | Name: {row['s_name']} | Addr: {row['s_addr']}")
    print("-" * 80)
