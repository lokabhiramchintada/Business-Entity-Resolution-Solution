import duckdb
import time

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

print("--- Checking Country Consistency Across Matches ---")
t0 = time.time()

# Let's unnest matched_entity_ids and check country of S1 vs country of matched entity
# First, load S1, S2, S3 with entity_id and country
con.execute(f"""
    CREATE TABLE s1 AS SELECT entity_id, country FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True);
    CREATE TABLE s2 AS SELECT entity_id, country FROM read_csv('{DATA_DIR}/train/train_source2.tsv', delim='\t', header=True);
    CREATE TABLE s3 AS SELECT entity_id, country FROM read_csv('{DATA_DIR}/train/train_source3.tsv', delim='\t', header=True);
    CREATE TABLE s_all AS 
        SELECT entity_id, country FROM s2
        UNION ALL
        SELECT entity_id, country FROM s3;
    CREATE INDEX idx_sall ON s_all(entity_id);
""")

print("Loaded S1, S2, S3 tables. Now unnesting matches...")

check_query = f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    )
    SELECT 
        s1.country AS s1_country,
        s_all.country AS matched_country,
        count(*) AS pair_count
    FROM unnested u
    JOIN s1 ON u.source1_entity_id = s1.entity_id
    JOIN s_all ON u.matched_id = s_all.entity_id
    GROUP BY s1.country, s_all.country;
"""

res = con.execute(check_query).df()
print(res)

# Also check number of matches per S1 entity
match_counts = con.execute(f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            len(string_split(matched_entity_ids, ',')) AS match_count
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    )
    SELECT 
        match_count,
        count(*) as num_s1
    FROM unnested
    GROUP BY match_count
    ORDER BY match_count;
""").df()
print("\nMatch count distribution:")
print(match_counts.head(20))

# Check S2 vs S3 match counts
source_breakdown = con.execute(f"""
    WITH unnested AS (
        SELECT 
            source1_entity_id,
            unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
    )
    SELECT 
        substr(matched_id, 1, 2) AS source,
        count(*) as count
    FROM unnested
    GROUP BY source;
""").df()
print("\nMatches by source:")
print(source_breakdown)

print(f"\nTotal time: {time.time() - t0:.2f}s")
