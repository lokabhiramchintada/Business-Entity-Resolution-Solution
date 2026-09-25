import os
import duckdb
import time

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"

con = duckdb.connect()

print("--- Checking Row Counts & Basic Stats ---")
t0 = time.time()

# Let's inspect ground truth
df_gt_stats = con.execute(f"""
    SELECT 
        COUNT(*) as total_s1,
        COUNT(CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = '' THEN 1 END) as singletons,
        COUNT(CASE WHEN matched_entity_ids IS NOT NULL AND matched_entity_ids != '' THEN 1 END) as with_matches
    FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
""").df()
print("Ground truth stats:")
print(df_gt_stats)

# Country distribution in train_source1, 2, 3
for name in ['train_source1', 'train_source2', 'train_source3']:
    path = f"{DATA_DIR}/train/{name}.tsv"
    dist = con.execute(f"""
        SELECT country, count(*) as count 
        FROM read_csv('{path}', delim='\t', header=True) 
        GROUP BY country
    """).df()
    print(f"\n{name} country distribution:")
    print(dist)

# Country distribution in test_source1, 2, 3
for name in ['test_source1', 'test_source2', 'test_source3']:
    path = f"{DATA_DIR}/test/{name}.tsv"
    dist = con.execute(f"""
        SELECT country, count(*) as count 
        FROM read_csv('{path}', delim='\t', header=True) 
        GROUP BY country
    """).df()
    print(f"\n{name} country distribution:")
    print(dist)

print(f"\nTime taken: {time.time() - t0:.2f}s")
