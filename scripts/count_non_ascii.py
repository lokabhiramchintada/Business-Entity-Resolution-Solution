import duckdb

DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
con = duckdb.connect()

for name in ['train_source1', 'train_source2', 'train_source3', 'test_source1', 'test_source2', 'test_source3']:
    res = con.execute(f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN regexp_matches(business_name, '[^\\x00-\\x7F]') THEN 1 END) as non_ascii_names,
            COUNT(CASE WHEN regexp_matches(business_address, '[^\\x00-\\x7F]') THEN 1 END) as non_ascii_addrs
        FROM read_csv('{DATA_DIR}/{name.split("_")[0]}/{name}.tsv', delim='\t', header=True);
    """).df()
    print(f"{name}: total={res['total'][0]}, non-ascii names={res['non_ascii_names'][0]} ({res['non_ascii_names'][0]/res['total'][0]*100:.2f}%), non-ascii addrs={res['non_ascii_addrs'][0]} ({res['non_ascii_addrs'][0]/res['total'][0]*100:.2f}%)")
