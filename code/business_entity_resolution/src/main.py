# src/main.py

import argparse
import os
import sys


# Make the project root importable when this file
# is executed directly with:
#
# python code/business_entity_resolution/src/main.py
#

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(
        0,
        PROJECT_DIR,
    )


from src.pipeline import EntityResolutionPipeline


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Business Entity Resolution "
            "Challenge Pipeline"
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Path to student_resource/dataset"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory",
    )

    parser.add_argument(
        "--model-path",
        default="models/entity_matcher.joblib",
        help="Model output path",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help=(
            "Number of candidates per S1. "
            "Recommended starting point: 24"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Initial matching threshold",
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.50,
        help="Minimum validation threshold",
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.99,
        help="Maximum validation threshold",
    )

    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Validation threshold step",
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=60000,
        help="Number of S1 training entities",
    )

    parser.add_argument(
        "--n-val",
        type=int,
        default=10000,
        help="Number of S1 validation entities",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Skip training and load existing model"
        ),
    )

    args = parser.parse_args()

    print(
        "\n"
        "====================================================\n"
        "BUSINESS ENTITY RESOLUTION\n"
        "===================================================="
    )

    print(
        f"Data directory : {args.data_dir}"
    )

    print(
        f"Output         : {args.output_dir}"
    )

    print(
        f"Model          : {args.model_path}"
    )

    print(
        f"Top-K          : {args.top_k}"
    )

    print(
        f"Initial thresh : {args.threshold}"
    )

    pipeline = EntityResolutionPipeline(
        top_k_candidates=args.top_k,
        threshold=args.threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    if args.skip_train:

        if not os.path.exists(
            args.model_path
        ):
            raise FileNotFoundError(
                f"Model not found: "
                f"{args.model_path}"
            )

        print(
            "\nSkipping training."
        )

        from src.model import (
            load_matching_model
        )

        pipeline.model = (
            load_matching_model(
                args.model_path
            )
        )

    else:

        print(
            "\nStarting training..."
        )

        pipeline.train(
            data_dir=args.data_dir,
            model_save_path=args.model_path,
            n_train_samples=args.n_train,
            n_val_samples=args.n_val,
        )

    # --------------------------------------------------------------
    # TEST
    # --------------------------------------------------------------

    print(
        "\nStarting test prediction..."
    )

    pipeline.predict_test(
        test_dir=os.path.join(
            args.data_dir,
            "test",
        ),
        output_dir=args.output_dir,
        model_path=args.model_path,
    )

    print(
        "\nSUCCESS"
    )

    print(
        f"matching_results.tsv: "
        f"{os.path.join(args.output_dir, 'matching_results.tsv')}"
    )

    print(
        f"candidate_pairs.tsv: "
        f"{os.path.join(args.output_dir, 'candidate_pairs.tsv')}"
    )

    print(
        f"Final threshold: "
        f"{pipeline.threshold:.4f}"
    )


if __name__ == "__main__":
    main()