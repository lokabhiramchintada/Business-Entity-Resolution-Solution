import os
import sys
import time
import math
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from .preprocessing import clean_tokens, clean_compact_name, extract_digits, extract_addr_tokens
from .blocking import CandidateGenerator
from .features import extract_pair_features, FEATURE_NAMES
from .model import train_matching_model, load_matching_model, compute_macro_f05

class EntityResolutionPipeline:
    def __init__(self, top_k_candidates=12, threshold=0.70):
        self.top_k_candidates = top_k_candidates
        self.threshold = threshold
        self.model = None

    def train(self, data_dir, model_save_path, n_train_samples=40000, n_val_samples=5000):
        """Train LightGBM matcher on a balanced sample from train split."""
        print(f"=== Training Pipeline (n_samples={n_train_samples}) ===")
        t0 = time.time()
        con = duckdb.connect()
        
        # Sample training and validation S1 entities
        total_samples = n_train_samples + n_val_samples
        print(f"Sampling {total_samples} S1 entities from {data_dir}/train/train_source1.tsv...")
        sample_s1 = con.execute(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{data_dir}/train/train_source1.tsv', delim='\t', header=True)
            USING SAMPLE {total_samples} (reservoir, 42);
        """).df()
        
        train_s1 = sample_s1.iloc[:n_train_samples].copy()
        val_s1 = sample_s1.iloc[n_train_samples:].copy()
        
        # Load ground truth for sampled entities
        con.register('sample_s1_df', sample_s1[['entity_id']])
        gt_df = con.execute(f"""
            WITH unnested AS (
                SELECT 
                    g.source1_entity_id,
                    unnest(string_split(g.matched_entity_ids, ',')) AS matched_id
                FROM read_csv('{data_dir}/train/train_ground_truth.tsv', delim='\t', header=True) g
                JOIN sample_s1_df s ON g.source1_entity_id = s.entity_id
                WHERE g.matched_entity_ids IS NOT NULL AND g.matched_entity_ids != ''
            )
            SELECT * FROM unnested;
        """).df()
        
        gt_dict = defaultdict(set)
        for _, r in gt_df.iterrows():
            gt_dict[r['source1_entity_id']].add(r['matched_id'])
            
        all_gt_matched_ids = set(gt_df['matched_id'].values)
        con.register('matched_ids_df', pd.DataFrame({'entity_id': list(all_gt_matched_ids)}))
        
        # Load matched targets + negative distractors
        print("Loading training targets (true matches + negative distractors)...")
        targets_df = con.execute(f"""
            WITH matched_s2 AS (
                SELECT s.entity_id, s.business_name, s.business_address, s.country
                FROM read_csv('{data_dir}/train/train_source2.tsv', delim='\t', header=True) s
                JOIN matched_ids_df m ON s.entity_id = m.entity_id
            ),
            matched_s3 AS (
                SELECT s.entity_id, s.business_name, s.business_address, s.country
                FROM read_csv('{data_dir}/train/train_source3.tsv', delim='\t', header=True) s
                JOIN matched_ids_df m ON s.entity_id = m.entity_id
            ),
            distractors_s2 AS (
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{data_dir}/train/train_source2.tsv', delim='\t', header=True)
                USING SAMPLE 150000 (reservoir, 99)
            ),
            distractors_s3 AS (
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{data_dir}/train/train_source3.tsv', delim='\t', header=True)
                USING SAMPLE 150000 (reservoir, 99)
            )
            SELECT * FROM matched_s2
            UNION ALL SELECT * FROM matched_s3
            UNION ALL SELECT * FROM distractors_s2
            UNION ALL SELECT * FROM distractors_s3;
        """).df().drop_duplicates(subset=['entity_id']).reset_index(drop=True)
        
        print(f"Loaded {len(targets_df)} target records for training.")
        
        # Build candidate generators partitioned by country
        country_generators = {}
        for country in ['US', 'India']:
            c_targets = targets_df[targets_df['country'] == country].reset_index(drop=True)
            if len(c_targets) > 0:
                gen = CandidateGenerator(max_cand_per_key=100, default_top_k=self.top_k_candidates)
                gen.fit(c_targets)
                country_generators[country] = gen
                
        # Generate training dataset
        print("Generating training candidate pairs and computing features...")
        X_train, y_train = [], []
        
        for _, row in train_s1.iterrows():
            c = row['country']
            if c not in country_generators:
                continue
            gen = country_generators[c]
            s1_id = row['entity_id']
            s1_name = row['business_name']
            s1_addr = row['business_address']
            
            cands = gen.query(s1_name, s1_addr, top_k=self.top_k_candidates)
            true_targets = gt_dict.get(s1_id, set())
            
            for tid_idx in cands:
                tgt_id = gen.t_ids[tid_idx]
                tgt_name = gen.t_names[tid_idx]
                tgt_addr = gen.t_addrs[tid_idx]
                
                feats = extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id)
                label = 1 if tgt_id in true_targets else 0
                
                X_train.append(feats)
                y_train.append(label)
                
        print(f"Training dataset: {len(X_train)} candidate pairs ({sum(y_train)} positive, {len(y_train)-sum(y_train)} negative).")
        
        # Train model
        self.model = train_matching_model(X_train, y_train, model_path=model_save_path)
        print(f"Model saved to {model_save_path}.")
        
        # Validate and optimize threshold on validation split
        print("Validating model and tuning threshold on holdout set...")
        val_gt_dict = {s1: gt_dict[s1] for s1 in val_s1['entity_id'] if s1 in gt_dict}
        val_s1_ids = list(val_s1['entity_id'].values)
        
        val_pair_records = []
        for _, row in val_s1.iterrows():
            c = row['country']
            if c not in country_generators:
                continue
            gen = country_generators[c]
            s1_id = row['entity_id']
            s1_name = row['business_name']
            s1_addr = row['business_address']
            
            cands = gen.query(s1_name, s1_addr, top_k=self.top_k_candidates)
            for tid_idx in cands:
                tgt_id = gen.t_ids[tid_idx]
                tgt_name = gen.t_names[tid_idx]
                tgt_addr = gen.t_addrs[tid_idx]
                feats = extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id)
                val_pair_records.append((s1_id, tgt_id, feats))
                
        if val_pair_records:
            X_val = np.array([r[2] for r in val_pair_records], dtype=np.float32)
            val_probs = self.model.predict(X_val)
            
            best_thresh, best_f05 = 0.70, 0.0
            for th in np.arange(0.50, 0.85, 0.05):
                preds = defaultdict(list)
                for (s1_id, tgt_id, _), p in zip(val_pair_records, val_probs):
                    if p >= th:
                        preds[s1_id].append(tgt_id)
                score = compute_macro_f05(preds, val_gt_dict, val_s1_ids)
                if score > best_f05:
                    best_f05 = score
                    best_thresh = th
                    
            print(f"Validation Optimal Threshold: {best_thresh:.2f} (Macro F_0.5 = {best_f05:.4f})")
            self.threshold = float(best_thresh)
            
        print(f"Training completed in {time.time() - t0:.2f}s.")
        return self.model

    def predict_test(self, test_dir, output_dir, model_path=None):
        """Run blocking and matching on full test set, country by country."""
        print("=== Running End-to-End Prediction on Test Set ===")
        t_start = time.time()
        
        if self.model is None:
            if model_path and os.path.exists(model_path):
                print(f"Loading trained model from {model_path}...")
                self.model = load_matching_model(model_path)
            else:
                raise ValueError("Model not trained and model_path does not exist.")
                
        con = duckdb.connect()
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load test_source1 metadata
        s1_file = os.path.join(test_dir, "test_source1.tsv")
        s2_file = os.path.join(test_dir, "test_source2.tsv")
        s3_file = os.path.join(test_dir, "test_source3.tsv")
        
        print(f"Reading S1 entities from {s1_file}...")
        s1_df = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{s1_file}', delim='\\t', header=True)").df()
        all_s1_order = list(s1_df['entity_id'].values)
        total_s1 = len(all_s1_order)
        print(f"Total Source 1 test entities: {total_s1}")
        
        countries = list(s1_df['country'].unique())
        print(f"Countries to process: {countries}")
        
        # Dictionaries to store outputs: s1_id -> list of IDs
        matching_results = {s1: [] for s1 in all_s1_order}
        candidate_pairs = {s1: [] for s1 in all_s1_order}
        
        # Process country by country to keep memory bounded and maximize cache locality
        for country in countries:
            print(f"\n--- Processing Country: {country} ---")
            t_c = time.time()
            s1_country_df = s1_df[s1_df['country'] == country].reset_index(drop=True)
            n_s1_country = len(s1_country_df)
            print(f"Country {country}: {n_s1_country} Source 1 entities.")
            
            # Load targets for this country from S2 and S3
            print(f"Loading Source 2 and Source 3 targets for {country}...")
            targets_country_df = con.execute(f"""
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{s2_file}', delim='\\t', header=True)
                WHERE country = '{country}'
                UNION ALL
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{s3_file}', delim='\\t', header=True)
                WHERE country = '{country}';
            """).df()
            print(f"Loaded {len(targets_country_df)} targets for {country} in {time.time() - t_c:.2f}s.")
            
            # Build inverted index
            t_idx = time.time()
            gen = CandidateGenerator(max_cand_per_key=100, default_top_k=self.top_k_candidates)
            gen.fit(targets_country_df)
            print(f"Index built for {country} in {time.time() - t_idx:.2f}s.")
            
            # Run inference in batches of S1 entities
            BATCH_SIZE = 10000
            total_batches = math.ceil(n_s1_country / BATCH_SIZE)
            
            t_inf = time.time()
            for b_idx in range(total_batches):
                b_start = b_idx * BATCH_SIZE
                b_end = min(b_start + BATCH_SIZE, n_s1_country)
                batch_slice = s1_country_df.iloc[b_start:b_end]
                
                # Collect candidate pairs for this batch
                batch_pairs = []
                for _, row in batch_slice.iterrows():
                    s1_id = row['entity_id']
                    s1_name = row['business_name']
                    s1_addr = row['business_address']
                    
                    cands = gen.query(s1_name, s1_addr, top_k=self.top_k_candidates)
                    cand_ids = [gen.t_ids[idx] for idx in cands]
                    candidate_pairs[s1_id] = cand_ids
                    
                    for tid_idx in cands:
                        tgt_id = gen.t_ids[tid_idx]
                        tgt_name = gen.t_names[tid_idx]
                        tgt_addr = gen.t_addrs[tid_idx]
                        feats = extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id)
                        batch_pairs.append((s1_id, tgt_id, feats))
                        
                # Score batch with model
                if batch_pairs:
                    X_b = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                    probs = self.model.predict(X_b)
                    
                    for (s1_id, tgt_id, _), prob in zip(batch_pairs, probs):
                        if prob >= self.threshold:
                            matching_results[s1_id].append(tgt_id)
                            
                if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                    elapsed = time.time() - t_inf
                    rate = b_end / elapsed
                    eta = (n_s1_country - b_end) / rate if rate > 0 else 0
                    print(f"  [{country}] Processed {b_end}/{n_s1_country} S1 entities ({b_end/n_s1_country*100:.1f}%) - {rate:.0f} ent/s - ETA: {eta:.0f}s")
                    
            del targets_country_df
            del gen
            print(f"Finished {country} in {time.time() - t_c:.2f}s.")
            
        # Write output files
        matching_file = os.path.join(output_dir, "matching_results.tsv")
        candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
        
        print(f"\nWriting matching results to {matching_file}...")
        with open(matching_file, "w", encoding="utf-8") as f_m:
            f_m.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in all_s1_order:
                m_list = matching_results[s1_id]
                # Ensure no duplicates within list and strictly valid
                m_list_unique = list(dict.fromkeys(m_list))
                f_m.write(f"{s1_id}\t{','.join(m_list_unique)}\n")
                
        print(f"Writing candidate pairs to {candidate_file}...")
        with open(candidate_file, "w", encoding="utf-8") as f_c:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in all_s1_order:
                c_list = candidate_pairs[s1_id]
                # Guarantee matching IDs are always included in candidate set
                m_list = matching_results[s1_id]
                all_candidates = list(dict.fromkeys(c_list + m_list))
                f_c.write(f"{s1_id}\t{','.join(all_candidates)}\n")
                
        total_time = time.time() - t_start
        print(f"=== All Test Predictions Finished in {total_time/60:.2f} minutes ===")
        return matching_file, candidate_file
