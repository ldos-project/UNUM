#!/usr/bin/env python3

import os
import pickle
import argparse
import numpy as np

def check_feature_ranges(dataset_path):
    """
    Load a (sample_num, 20, 6) dataset from a pickle file
    and print each feature's min and max.
    """
    if not os.path.isfile(dataset_path):
        print(f"File not found: {dataset_path}")
        return

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)  # shape: (N, 20, 6) if everything is correct

    if len(data.shape) != 3 or data.shape[1] != 20 or data.shape[2] != 6:
        print(f"Unexpected shape: {data.shape} (expected (N, 20, 6))")
        return

    # Reshape to (N*20, 6) to treat each row as a single data point
    reshaped = data.reshape(-1, 6)
    print("sample data points: ", reshaped[0, :])

    print(f"Dataset loaded: shape {data.shape}, reshaped to {reshaped.shape}.")
    for i in range(6):
        feat_vals = reshaped[:, i]
        fmin = np.min(feat_vals)
        fmax = np.max(feat_vals)
        print(f"Feature {i} -> Min: {fmin:.6f}, Max: {fmax:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Check feature ranges from combined RTT dataset")
    parser.add_argument(
        "--dataset-path",
        default="/datastor1/janec/datasets/",
        help="Path to the combined dataset pickle file (shape: (N, 20, 6))"
    )
    args = parser.parse_args()

    for dataset in os.listdir(args.dataset_path):
        if dataset.endswith(".p"):
            print(f"Checking ranges for {dataset}...")
            dataset_path = os.path.join(args.dataset_path, dataset)
            check_feature_ranges(dataset_path)
        else:
            print(f"Skipping non-pickle file: {dataset}")

if __name__ == "__main__":
    main()
