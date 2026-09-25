import duckdb

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

query = f"""
    WITH singletons AS (
        SELECT source1_entity_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids IS NULL OR matched_entity_ids = ''
        LIMIT 20
    ),
    s1 AS (
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)
    )
    SELECT 
        s.source1_entity_id,
        s1.business_name,
        s1.business_address,
        s1.country
    FROM singletons s
    JOIN s1 ON s.source1_entity_id = s1.entity_id;
"""

df_singletons = con.execute(query).df()
print("Sample Singletons:")
for i, row in df_singletons.iterrows():
    print(f"[{row['country']}] S1: {row['source1_entity_id']} | Name: {row['business_name']} | Addr: {row['business_address']}")
