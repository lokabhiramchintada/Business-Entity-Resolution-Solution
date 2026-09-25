import duckdb
con = duckdb.connect()
DATA_DIR = '6ab10eb3b23ba_student_resource/student_resource/dataset'

print('=== Training Data Stats ===')
s1_total = con.execute(f"SELECT COUNT(*) FROM read_csv('{DATA_DIR}/train/train_source1.tsv', delim='\t', header=True)").fetchone()[0]
print(f'Total train S1: {s1_total:,}')

gt_stats = con.execute(f"""
    SELECT
        COUNT(*) as total_s1,
        COUNT(CASE WHEN matched_entity_ids IS NOT NULL AND len(matched_entity_ids) > 0 THEN 1 END) as with_matches,
        COUNT(CASE WHEN matched_entity_ids IS NULL OR len(matched_entity_ids) = 0 THEN 1 END) as singletons
    FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
""").df()
print(gt_stats)

# Average number of true matches per S1 entity (non-singletons only)
avg = con.execute(f"""
    SELECT
        avg(len(string_split(matched_entity_ids, ','))) as avg_matches_non_singleton,
        max(len(string_split(matched_entity_ids, ','))) as max_matches,
        approx_quantile(len(string_split(matched_entity_ids, ',')), 0.5) as median_matches
    FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
    WHERE matched_entity_ids IS NOT NULL AND len(matched_entity_ids) > 0
""").df()
print('Per-entity match count stats (non-singletons):', avg)

# Distribution: how many entities have exactly 1, 2, 3 ... matches?
dist = con.execute(f"""
    SELECT
        len(string_split(matched_entity_ids, ',')) as match_count,
        COUNT(*) as n_entities
    FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
    WHERE matched_entity_ids IS NOT NULL AND len(matched_entity_ids) > 0
    GROUP BY 1
    ORDER BY 1
""").df()
print(dist)

# How many unique target S2+S3 appear in the ground truth?
unique_targets = con.execute(f"""
    WITH unnested AS (
        SELECT unnest(string_split(matched_entity_ids, ',')) AS matched_id
        FROM read_csv('{DATA_DIR}/train/train_ground_truth.tsv', delim='\t', header=True)
        WHERE matched_entity_ids IS NOT NULL AND len(matched_entity_ids) > 0
    )
    SELECT
        COUNT(*) as total_match_pairs,
        COUNT(DISTINCT matched_id) as unique_matched_targets
    FROM unnested
""").df()
print(unique_targets)
