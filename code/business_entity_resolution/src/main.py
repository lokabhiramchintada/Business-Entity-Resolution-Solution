import argparse
import os
import sys

# Ensure parent directory is in sys.path so src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.pipeline import EntityResolutionPipeline

def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default="dataset", help="Path to base dataset directory containing train/ and test/")
    parser.add_argument("--test-dir", default=None, help="Path to test directory (default: <data-dir>/test)")
    parser.add_argument("--output-dir", default="output", help="Directory to save matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--model-path", default="models/lgbm_matcher.joblib", help="Path to save or load LightGBM model")
    parser.add_argument("--top-k", type=int, default=12, help="Top-K candidate set cap per Source 1 entity")
    parser.add_argument("--threshold", type=float, default=0.70, help="Probability threshold for final matches")
    parser.add_argument("--n-train", type=int, default=40000, help="Number of S1 training samples")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if model-path already exists")
    
    args = parser.parse_args()
    
    test_dir = args.test_dir or os.path.join(args.data_dir, "test")
    pipeline = EntityResolutionPipeline(top_k_candidates=args.top_k, threshold=args.threshold)
    
    # 1. Train or load model
    if not args.skip_train or not os.path.exists(args.model_path):
        print(f"Training model using data from {args.data_dir}...")
        pipeline.train(
            data_dir=args.data_dir,
            model_save_path=args.model_path,
            n_train_samples=args.n_train,
            n_val_samples=5000
        )
    else:
        print(f"Skipping training; using existing model at {args.model_path}.")
        
    # 2. Predict on test set
    matching_tsv, candidate_tsv = pipeline.predict_test(
        test_dir=test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    
    print("\nSUCCESS: Generated outputs:")
    print(f"  Matching:  {matching_tsv}")
    print(f"  Candidate: {candidate_tsv}")

if __name__ == "__main__":
    main()
