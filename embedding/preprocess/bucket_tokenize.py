#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import ntpath

##########################################
# Constants and paths
##########################################
COMBINED_DIR = "/datastor1/janec/datasets/combined"
TRAIN_DATA_PATH = os.path.join(COMBINED_DIR, "6col-20rtt-train.p")
FEATURE_DIM = 6

##########################################
# Bucket Utility
##########################################
def bucketize_value(value, boundaries):
    """
    Given a float `value` and a sorted list of boundary cutoffs,
    return the bucket index in [0..len(boundaries]].
      - If value < boundaries[0], index=0
      - If boundaries[j-1] <= value < boundaries[j], index=j
      - If value >= boundaries[-1], index=len(boundaries)
    """
    if not boundaries:  # if the feature is being ignored or missing boundaries
        return 0
    idx = np.searchsorted(boundaries, value, side='right')
    return idx

##########################################
# Main Tokenization Logic
##########################################
def main():
    parser = argparse.ArgumentParser(description="Tokenize train data (N,20,6) with single-token or multi-head approach.")
    parser.add_argument("--boundaries-file", required=True,
                        help="Path to the pickle file containing bucket boundaries (feature_idx->list of floats).")
    args = parser.parse_args()

    # 1) Load the train data
    if not os.path.isfile(TRAIN_DATA_PATH):
        print(f"Train data file not found: {TRAIN_DATA_PATH}")
        return

    with open(TRAIN_DATA_PATH, "rb") as f:
        train_data = pickle.load(f)

    # Expect shape: (N, 20, 6)
    if not (isinstance(train_data, np.ndarray) and train_data.ndim == 3 and train_data.shape[2] == FEATURE_DIM):
        print(f"Unexpected shape: {train_data.shape}, expected (N,20,6). Exiting.")
        return

    print(f"Loaded train data: shape = {train_data.shape}")

    # 2) Load boundaries
    if not os.path.isfile(args.boundaries_file):
        print(f"Boundaries file not found: {args.boundaries_file}")
        return

    with open(args.boundaries_file, "rb") as f:
        boundaries_dict = pickle.load(f)

    print(f"Loaded boundaries from {args.boundaries_file}")

    # We'll parse out base RTT (feature 0) separately and keep it as float
    base_rtt_data = train_data[:, :, 0].copy()  # shape (N,20)

    # We will tokenize features 1..5
    # A) Single combo approach => a single vocabulary for (b1,b2,b3,b4,b5)
    # B) Multi-head approach => store 5 separate bucket indices at each time step

    ##########################################
    # PART A: Single-Integer “Mega-Token”
    ##########################################
    combo_dict = {}  # (b1,b2,b3,b4,b5) -> token_id
    combo_list = []  # token_id -> (b1,b2,b3,b4,b5)
    next_combo_id = 0

    def get_or_add_combo_token(combo_tuple):
        nonlocal next_combo_id
        if combo_tuple in combo_dict:
            return combo_dict[combo_tuple]
        else:
            combo_dict[combo_tuple] = next_combo_id
            combo_list.append(combo_tuple)
            next_combo_id += 1
            return next_combo_id - 1

    # Create an int array of shape (N,20) for single-combo tokens
    tokens_single = np.zeros((train_data.shape[0], 20), dtype=np.int32)

    ##########################################
    # PART B: Multi-Classification-Head
    ##########################################
    # We produce a (N, 20, 5) array: each feature has its own bucket index
    #   multi_tokens[i, t, f] => the bucket index for feature (f+1) at sample i, time t.
    # We skip feature 0. So feature (f+1) means actual feature indices (1..5).
    tokens_multi = np.zeros((train_data.shape[0], 20, 5), dtype=np.int32)

    ##########################################
    # 3) Iterate over each sample/time step
    ##########################################
    N = train_data.shape[0]
    for i in range(N):
        for t in range(20):
            # For the single combo approach, we collect feature 1..5’s bucket indices
            combo_indices = []

            # For the multi-head approach, we fill tokens_multi[i,t,*]
            for feat_sub in range(5):
                feat_idx = feat_sub + 1  # actual feature index in [1..5]
                val = train_data[i, t, feat_idx]
                bds = boundaries_dict.get(feat_idx, [])
                b_idx = bucketize_value(val, bds)

                # single combo
                combo_indices.append(b_idx)

                # multi-head
                tokens_multi[i, t, feat_sub] = b_idx

            # single combo => get or add
            combo_tuple = tuple(combo_indices)
            token_id_combo = get_or_add_combo_token(combo_tuple)
            tokens_single[i, t] = token_id_combo

    print("Finished building token arrays.")
    print(f"Single-combo vocabulary size = {len(combo_dict)}")

    ##########################################
    # 4) Save results
    ##########################################
    # We'll name output based on boundary file
    boundary_base = ntpath.basename(args.boundaries_file)
    boundary_base_noext = os.path.splitext(boundary_base)[0]  # e.g., "boundaries-quantile1000"

    # A) Single-integer “mega-token”
    out_single_path = os.path.join(
        COMBINED_DIR, f"{boundary_base_noext}-tokenized-single.pkl"
    )
    single_data_dict = {
        "base_rtt": base_rtt_data,          # shape (N,20) float
        "tokens_single": tokens_single,     # shape (N,20) int
        "combo_dict": combo_dict,           # (b1,b2,b3,b4,b5)->int
        "combo_list": combo_list            # int->(b1,b2,b3,b4,b5)
    }
    with open(out_single_path, "wb") as f:
        pickle.dump(single_data_dict, f)
    print(f"Saved single-combo tokenized data => {out_single_path}")

    # B) Multi-classification-head approach
    # Each of the 5 features has a bucket index in [N,20,5].
    # There's no single "vocab" here. The boundaries themselves define the max bucket for each feature.
    out_multi_path = os.path.join(
        COMBINED_DIR, f"{boundary_base_noext}-tokenized-multi.pkl"
    )
    multi_data_dict = {
        "base_rtt": base_rtt_data,    # shape (N,20)
        "tokens_multi": tokens_multi, # shape (N,20,5) int
        # If you want to store boundaries for each feature or track max indices, do so as well:
        "boundaries_dict": boundaries_dict
    }
    with open(out_multi_path, "wb") as f:
        pickle.dump(multi_data_dict, f)
    print(f"Saved multi-head tokenized data =>", out_multi_path)

if __name__ == "__main__":
    main()
