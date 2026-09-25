# src/pipeline.py

import os
import sys
import time
import math

import duckdb
import numpy as np
import pandas as pd

from collections import defaultdict

from .blocking import CandidateGenerator
from .features import extract_pair_features, FEATURE_NAMES
from .model import (
    train_matching_model,
    load_matching_model,
    compute_macro_f05,
)


class EntityResolutionPipeline:

    def __init__(
        self,
        top_k_candidates=24,
        threshold=0.80,
        threshold_min=0.50,
        threshold_max=0.99,
        threshold_step=0.01,
    ):
        self.top_k_candidates = int(top_k_candidates)

        self.threshold = float(threshold)

        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.threshold_step = float(threshold_step)

        self.model = None

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------

    @staticmethod
    def _read_source(con, path):
        return con.execute(
            f"""
            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM read_csv(
                '{path}',
                delim='\\t',
                header=True
            )
            """
        ).df()

    @staticmethod
    def _load_ground_truth(con, data_dir, s1_ids):
        """
        Load ground truth only for the requested S1 IDs.
        """

        if not s1_ids:
            return defaultdict(set)

        s1_df = pd.DataFrame(
            {
                "entity_id": list(s1_ids)
            }
        )

        con.register("requested_s1", s1_df)

        gt_df = con.execute(
            f"""
            WITH unnested AS (
                SELECT
                    g.source1_entity_id,
                    unnest(
                        string_split(
                            g.matched_entity_ids,
                            ','
                        )
                    ) AS matched_id
                FROM read_csv(
                    '{data_dir}/train/train_ground_truth.tsv',
                    delim='\\t',
                    header=True
                ) g
                JOIN requested_s1 s
                    ON g.source1_entity_id = s.entity_id
                WHERE
                    g.matched_entity_ids IS NOT NULL
                    AND g.matched_entity_ids != ''
            )
            SELECT
                source1_entity_id,
                matched_id
            FROM unnested
            """
        ).df()

        gt_dict = defaultdict(set)

        for _, row in gt_df.iterrows():
            gt_dict[row["source1_entity_id"]].add(
                row["matched_id"]
            )

        return gt_dict

    # ------------------------------------------------------------------
    # CANDIDATE GENERATOR
    # ------------------------------------------------------------------

    def _build_country_generators(self, targets_df):
        """
        Build candidate generators using the COMPLETE target population.

        This is intentionally different from the previous implementation,
        which used only matched records + random distractors.
        """

        generators = {}

        countries = [
            c
            for c in targets_df["country"].dropna().unique()
        ]

        print(
            f"Building candidate generators for "
            f"{len(countries)} countries..."
        )

        for country in countries:

            country_targets = targets_df[
                targets_df["country"] == country
            ].reset_index(drop=True)

            if len(country_targets) == 0:
                continue

            print(
                f"  Building index for {country}: "
                f"{len(country_targets):,} targets"
            )

            generator = CandidateGenerator(
                max_cand_per_key=100,
                default_top_k=self.top_k_candidates,
            )

            generator.fit(country_targets)

            generators[country] = generator

        return generators

    # ------------------------------------------------------------------
    # CANDIDATE GENERATION
    # ------------------------------------------------------------------

    def _generate_pairs(
        self,
        s1_df,
        generators,
        gt_dict=None,
        collect_features=True,
        calculate_recall=False,
    ):
        """
        Generate candidate pairs.

        Returns:
            pair_records
            candidate_pairs
            recall_stats
        """

        if gt_dict is None:
            gt_dict = {}

        pair_records = []
        candidate_pairs = {}

        total_true = 0
        total_found = 0

        for idx, row in s1_df.iterrows():

            s1_id = row["entity_id"]
            country = row["country"]

            generator = generators.get(country)

            if generator is None:
                candidate_pairs[s1_id] = []
                continue

            s1_name = row["business_name"]
            s1_addr = row["business_address"]

            candidates = generator.query(
                s1_name,
                s1_addr,
                top_k=self.top_k_candidates,
            )

            candidate_ids = []

            true_targets = set(
                gt_dict.get(s1_id, set())
            )

            found_targets = set()

            for tid_idx in candidates:

                tgt_id = generator.t_ids[tid_idx]

                candidate_ids.append(tgt_id)

                if tgt_id in true_targets:
                    found_targets.add(tgt_id)

                if collect_features:

                    tgt_name = generator.t_names[tid_idx]
                    tgt_addr = generator.t_addrs[tid_idx]

                    features = extract_pair_features(
                        s1_name,
                        s1_addr,
                        tgt_name,
                        tgt_addr,
                        tgt_id,
                    )

                    label = (
                        1
                        if tgt_id in true_targets
                        else 0
                    )

                    pair_records.append(
                        (
                            s1_id,
                            tgt_id,
                            features,
                            label,
                        )
                    )

            candidate_pairs[s1_id] = list(
                dict.fromkeys(candidate_ids)
            )

            if calculate_recall:

                total_true += len(true_targets)
                total_found += len(found_targets)

        recall = (
            total_found / total_true
            if total_true > 0
            else 1.0
        )

        return (
            pair_records,
            candidate_pairs,
            {
                "true_matches": total_true,
                "found_matches": total_found,
                "candidate_recall": recall,
            },
        )

    # ------------------------------------------------------------------
    # THRESHOLD OPTIMIZATION
    # ------------------------------------------------------------------

    def _find_best_threshold(
        self,
        val_pair_records,
        val_gt_dict,
        val_s1_ids,
    ):

        if not val_pair_records:
            print("No validation pairs.")
            return self.threshold, 0.0

        X_val = np.asarray(
            [
                record[2]
                for record in val_pair_records
            ],
            dtype=np.float32,
        )

        print(
            f"Scoring {len(X_val):,} validation pairs..."
        )

        val_probs = self.model.predict_proba(
            X_val
        )

        best_threshold = self.threshold
        best_score = -1.0

        thresholds = np.arange(
            self.threshold_min,
            self.threshold_max + 1e-9,
            self.threshold_step,
        )

        print(
            f"Threshold sweep: "
            f"{self.threshold_min:.2f} → "
            f"{self.threshold_max:.2f} "
            f"step={self.threshold_step:.2f}"
        )

        for threshold in thresholds:

            preds = defaultdict(list)

            for record, probability in zip(
                val_pair_records,
                val_probs,
            ):

                s1_id = record[0]
                tgt_id = record[1]

                if probability >= threshold:
                    preds[s1_id].append(tgt_id)

            score = compute_macro_f05(
                preds,
                val_gt_dict,
                val_s1_ids,
            )

            print(
                f"  threshold={threshold:.2f} "
                f"F0.5={score:.6f}"
            )

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        return best_threshold, best_score

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    def train(
        self,
        data_dir,
        model_save_path,
        n_train_samples=60000,
        n_val_samples=10000,
    ):

        print(
            "\n"
            "====================================================\n"
            "TRAINING PIPELINE\n"
            "===================================================="
        )

        t0 = time.time()

        con = duckdb.connect()

        train_s1_path = os.path.join(
            data_dir,
            "train",
            "train_source1.tsv",
        )

        train_s2_path = os.path.join(
            data_dir,
            "train",
            "train_source2.tsv",
        )

        train_s3_path = os.path.join(
            data_dir,
            "train",
            "train_source3.tsv",
        )

        # --------------------------------------------------------------
        # SAMPLE S1
        # --------------------------------------------------------------

        total_samples = (
            n_train_samples
            + n_val_samples
        )

        print(
            f"Sampling {total_samples:,} "
            f"Source-1 entities..."
        )

        sample_s1 = con.execute(
            f"""
            SELECT
                entity_id,
                business_name,
                business_address,
                country
            FROM read_csv(
                '{train_s1_path}',
                delim='\\t',
                header=True
            )
            USING SAMPLE
                {total_samples}
                (reservoir, 42)
            """
        ).df()

        if len(sample_s1) < total_samples:
            raise RuntimeError(
                "Could not sample requested number of S1 entities."
            )

        train_s1 = sample_s1.iloc[
            :n_train_samples
        ].copy()

        val_s1 = sample_s1.iloc[
            n_train_samples:
        ].copy()

        train_s1_ids = set(
            train_s1["entity_id"].values
        )

        val_s1_ids = list(
            val_s1["entity_id"].values
        )

        # --------------------------------------------------------------
        # GROUND TRUTH
        # --------------------------------------------------------------

        print("Loading ground truth...")

        all_sample_ids = set(
            sample_s1["entity_id"].values
        )

        gt_dict = self._load_ground_truth(
            con,
            data_dir,
            all_sample_ids,
        )

        train_gt_dict = {
            s1_id: gt_dict.get(
                s1_id,
                set()
            )
            for s1_id in train_s1_ids
        }

        val_gt_dict = {
            s1_id: gt_dict.get(
                s1_id,
                set()
            )
            for s1_id in val_s1_ids
        }

        print(
            f"Ground-truth S1 entities: "
            f"{len(gt_dict):,}"
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        # FULL TARGET POPULATION
        # --------------------------------------------------------------

        print(
            "\nLoading COMPLETE training target population..."
        )

        train_s2 = self._read_source(
            con,
            train_s2_path,
        )

        train_s3 = self._read_source(
            con,
            train_s3_path,
        )

        targets_df = pd.concat(
            [
                train_s2,
                train_s3,
            ],
            ignore_index=True,
        )

        targets_df = targets_df.drop_duplicates(
            subset=["entity_id"]
        ).reset_index(drop=True)

        print(
            f"Full training targets: "
            f"{len(targets_df):,}"
        )

        # --------------------------------------------------------------
        # BUILD FULL COUNTRY INDEX
        # --------------------------------------------------------------

        index_start = time.time()

        country_generators = (
            self._build_country_generators(
                targets_df
            )
        )

        print(
            f"Candidate indexes built in "
            f"{time.time() - index_start:.2f}s"
        )

        # --------------------------------------------------------------
        # TRAIN CANDIDATES
        # --------------------------------------------------------------

        print(
            "\nGenerating TRAIN candidate pairs..."
        )

        train_pairs, _, train_recall = (
            self._generate_pairs(
                train_s1,
                country_generators,
                gt_dict=train_gt_dict,
                collect_features=True,
                calculate_recall=True,
            )
        )

        print(
            "\nTRAIN candidate recall:"
        )

        print(
            f"  true matches : "
            f"{train_recall['true_matches']:,}"
        )

        print(
            f"  found        : "
            f"{train_recall['found_matches']:,}"
        )

        print(
            f"  recall       : "
            f"{train_recall['candidate_recall']:.6f}"
        )

        if not train_pairs:
            raise RuntimeError(
                "No training candidate pairs were generated."
            )

        # --------------------------------------------------------------
        # BUILD TRAIN MATRIX
        # --------------------------------------------------------------

        X_train = np.asarray(
            [
                record[2]
                for record in train_pairs
            ],
            dtype=np.float32,
        )

        y_train = np.asarray(
            [
                record[3]
                for record in train_pairs
            ],
            dtype=np.int8,
        )

        positives = int(y_train.sum())
        negatives = int(
            len(y_train) - positives
        )

        print(
            "\nTraining dataset:"
        )

        print(
            f"  pairs      : {len(y_train):,}"
        )

        print(
            f"  positives  : {positives:,}"
        )

        print(
            f"  negatives  : {negatives:,}"
        )

        print(
            f"  pos ratio  : "
            f"{positives / max(len(y_train), 1):.4f}"
        )

        # --------------------------------------------------------------
        # TRAIN MODEL
        # --------------------------------------------------------------

        self.model = train_matching_model(
            X_train,
            y_train,
            model_path=model_save_path,
            feature_names=FEATURE_NAMES,
        )

        # --------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------

        print(
            "\nGenerating VALIDATION candidate pairs..."
        )

        val_pairs, _, val_recall = (
            self._generate_pairs(
                val_s1,
                country_generators,
                gt_dict=val_gt_dict,
                collect_features=True,
                calculate_recall=True,
            )
        )

        print(
            "\n===================================================="
        )
        print(
            "VALIDATION CANDIDATE RECALL"
        )
        print(
            "===================================================="
        )

        print(
            f"True matches : "
            f"{val_recall['true_matches']:,}"
        )

        print(
            f"Found        : "
            f"{val_recall['found_matches']:,}"
        )

        print(
            f"Recall       : "
            f"{val_recall['candidate_recall']:.6f}"
        )

        if val_recall["candidate_recall"] < 0.98:
            print(
                "\nWARNING:"
            )
            print(
                "Candidate recall is below 98%."
            )
            print(
                "The matcher cannot achieve 98% overall "
                "F0.5 unless the blocker improves."
            )

        # --------------------------------------------------------------
        # VALIDATION MATCHER
        # --------------------------------------------------------------

        best_threshold, best_f05 = (
            self._find_best_threshold(
                val_pairs,
                val_gt_dict,
                val_s1_ids,
            )
        )

        self.threshold = best_threshold

        print(
            "\n===================================================="
        )
        print(
            "VALIDATION RESULT"
        )
        print(
            "===================================================="
        )

        print(
            f"Best threshold : "
            f"{best_threshold:.4f}"
        )

        print(
            f"Macro F0.5     : "
            f"{best_f05:.6f}"
        )

        print(
            f"Training time  : "
            f"{time.time() - t0:.2f}s"
        )

        print(
            "\nModel and validation completed."
        )

        return self.model

    # ------------------------------------------------------------------
    # TEST PREDICTION
    # ------------------------------------------------------------------

    def predict_test(
        self,
        test_dir,
        output_dir,
        model_path=None,
    ):

        print(
            "\n"
            "====================================================\n"
            "TEST PREDICTION\n"
            "===================================================="
        )

        t_start = time.time()

        if self.model is None:

            if (
                model_path
                and os.path.exists(model_path)
            ):
                self.model = load_matching_model(
                    model_path
                )
            else:
                raise ValueError(
                    "Model is not loaded."
                )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        con = duckdb.connect()

        s1_path = os.path.join(
            test_dir,
            "test_source1.tsv",
        )

        s2_path = os.path.join(
            test_dir,
            "test_source2.tsv",
        )

        s3_path = os.path.join(
            test_dir,
            "test_source3.tsv",
        )

        # --------------------------------------------------------------
        # LOAD S1
        # --------------------------------------------------------------

        print(
            f"Reading {s1_path}"
        )

        s1_df = self._read_source(
            con,
            s1_path,
        )

        all_s1_order = list(
            s1_df["entity_id"].values
        )

        print(
            f"Total S1 entities: "
            f"{len(all_s1_order):,}"
        )

        countries = list(
            s1_df["country"]
            .dropna()
            .unique()
        )

        print(
            f"Countries: {countries}"
        )

        matching_results = {
            s1_id: []
            for s1_id in all_s1_order
        }

        candidate_pairs = {
            s1_id: []
            for s1_id in all_s1_order
        }

        # --------------------------------------------------------------
        # COUNTRY-BY-COUNTRY
        # --------------------------------------------------------------

        for country in countries:

            country_start = time.time()

            print(
                "\n"
                "----------------------------------------------------"
            )

            print(
                f"COUNTRY: {country}"
            )

            print(
                "----------------------------------------------------"
            )

            s1_country = s1_df[
                s1_df["country"] == country
            ].reset_index(drop=True)

            if len(s1_country) == 0:
                continue

            # ----------------------------------------------------------
            # TARGETS
            # ----------------------------------------------------------

            targets_country = con.execute(
                f"""
                SELECT
                    entity_id,
                    business_name,
                    business_address,
                    country
                FROM read_csv(
                    '{s2_path}',
                    delim='\\t',
                    header=True
                )
                WHERE country = '{country}'

                UNION ALL

                SELECT
                    entity_id,
                    business_name,
                    business_address,
                    country
                FROM read_csv(
                    '{s3_path}',
                    delim='\\t',
                    header=True
                )
                WHERE country = '{country}'
                """
            ).df()

            targets_country = (
                targets_country
                .drop_duplicates(
                    subset=["entity_id"]
                )
                .reset_index(drop=True)
            )

            print(
                f"S1: {len(s1_country):,}"
            )

            print(
                f"S2/S3 targets: "
                f"{len(targets_country):,}"
            )

            # ----------------------------------------------------------
            # INDEX
            # ----------------------------------------------------------

            index_start = time.time()

            generator = CandidateGenerator(
                max_cand_per_key=100,
                default_top_k=self.top_k_candidates,
            )

            generator.fit(
                targets_country
            )

            print(
                f"Index time: "
                f"{time.time() - index_start:.2f}s"
            )

            # ----------------------------------------------------------
            # INFERENCE
            # ----------------------------------------------------------

            BATCH_SIZE = 10000

            n = len(s1_country)

            total_batches = math.ceil(
                n / BATCH_SIZE
            )

            inference_start = time.time()

            for batch_idx in range(
                total_batches
            ):

                start = (
                    batch_idx
                    * BATCH_SIZE
                )

                end = min(
                    start + BATCH_SIZE,
                    n,
                )

                batch = s1_country.iloc[
                    start:end
                ]

                batch_pairs = []

                for _, row in batch.iterrows():

                    s1_id = row["entity_id"]

                    s1_name = row[
                        "business_name"
                    ]

                    s1_addr = row[
                        "business_address"
                    ]

                    candidates = (
                        generator.query(
                            s1_name,
                            s1_addr,
                            top_k=self.top_k_candidates,
                        )
                    )

                    candidate_ids = []

                    for tid_idx in candidates:

                        tgt_id = (
                            generator.t_ids[
                                tid_idx
                            ]
                        )

                        candidate_ids.append(
                            tgt_id
                        )

                        tgt_name = (
                            generator.t_names[
                                tid_idx
                            ]
                        )

                        tgt_addr = (
                            generator.t_addrs[
                                tid_idx
                            ]
                        )

                        features = (
                            extract_pair_features(
                                s1_name,
                                s1_addr,
                                tgt_name,
                                tgt_addr,
                                tgt_id,
                            )
                        )

                        batch_pairs.append(
                            (
                                s1_id,
                                tgt_id,
                                features,
                            )
                        )

                    candidate_pairs[
                        s1_id
                    ] = list(
                        dict.fromkeys(
                            candidate_ids
                        )
                    )

                # ------------------------------------------------------
                # SCORE
                # ------------------------------------------------------

                if batch_pairs:

                    X_batch = np.asarray(
                        [
                            pair[2]
                            for pair in batch_pairs
                        ],
                        dtype=np.float32,
                    )

                    probabilities = (
                        self.model.predict_proba(
                            X_batch
                        )
                    )

                    for pair, probability in zip(
                        batch_pairs,
                        probabilities,
                    ):

                        if (
                            probability
                            >= self.threshold
                        ):

                            matching_results[
                                pair[0]
                            ].append(
                                pair[1]
                            )

                if (
                    (batch_idx + 1) % 10 == 0
                    or batch_idx + 1
                    == total_batches
                ):

                    elapsed = (
                        time.time()
                        - inference_start
                    )

                    rate = (
                        end / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining = (
                        n - end
                    )

                    eta = (
                        remaining / rate
                        if rate > 0
                        else 0
                    )

                    print(
                        f"[{country}] "
                        f"{end:,}/{n:,} "
                        f"({100 * end / n:.1f}%) "
                        f"{rate:.0f} S1/s "
                        f"ETA {eta:.0f}s"
                    )

            print(
                f"Finished {country} in "
                f"{time.time() - country_start:.2f}s"
            )

        # --------------------------------------------------------------
        # CLEAN MATCHES
        # --------------------------------------------------------------

        for s1_id in matching_results:

            matching_results[s1_id] = list(
                dict.fromkeys(
                    matching_results[s1_id]
                )
            )

        # --------------------------------------------------------------
        # WRITE MATCHING
        # --------------------------------------------------------------

        matching_path = os.path.join(
            output_dir,
            "matching_results.tsv",
        )

        candidate_path = os.path.join(
            output_dir,
            "candidate_pairs.tsv",
        )

        print(
            f"\nWriting {matching_path}"
        )

        with open(
            matching_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "source1_entity_id\t"
                "matched_entity_ids\n"
            )

            for s1_id in all_s1_order:

                matches = matching_results[
                    s1_id
                ]

                f.write(
                    f"{s1_id}\t"
                    f"{','.join(matches)}\n"
                )

        # --------------------------------------------------------------
        # WRITE CANDIDATES
        # --------------------------------------------------------------

        print(
            f"Writing {candidate_path}"
        )

        with open(
            candidate_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "source1_entity_id\t"
                "candidate_entity_ids\n"
            )

            for s1_id in all_s1_order:

                candidates = candidate_pairs[
                    s1_id
                ]

                f.write(
                    f"{s1_id}\t"
                    f"{','.join(candidates)}\n"
                )

        elapsed_total = (
            time.time() - t_start
        )

        print(
            "\n===================================================="
        )

        print(
            "PREDICTION COMPLETE"
        )

        print(
            f"Runtime: "
            f"{elapsed_total / 60:.2f} minutes"
        )

        print(
            f"Matching: {matching_path}"
        )

        print(
            f"Candidates: {candidate_path}"
        )

        return (
            matching_path,
            candidate_path,
        )