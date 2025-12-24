#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import ntpath
from multiprocessing import Pool, cpu_count

##########################################
# Paths
##########################################
COMBINED_DIR = "/datastor1/janec/datasets/combined"
TRAIN_DATA_PATH = os.path.join(COMBINED_DIR, "6col_20rtt_train_combined.p")
FEATURE_DIM = 6

##########################################
# Bucket Utility
##########################################
def bucketize_value(value, boundaries):
    if not boundaries:
        return 0
    idx = np.searchsorted(boundaries, value, side='left')
    return idx

##########################################
# 1) Gather combos in parallel (PASS 1)
##########################################
def gather_combos_chunk(args):
    """
    args: (chunk_data, boundaries_dict)
    Return: set of combos for single-integer approach
    Also produce multi_head array for this chunk.
    """
    data_chunk, boundaries_dict = args
    # data_chunk shape = (chunk_size, 20, 6)

    combos_set = set()                  # for single integer approach
    multi_head_out = np.zeros((data_chunk.shape[0], 20, 5), dtype=np.int32)

    for i in range(data_chunk.shape[0]):
        for t in range(20):
            combo_indices = []
            for feat_sub in range(5):
                feat_idx = feat_sub + 1  # real feature in [1..5]
                val = data_chunk[i, t, feat_idx]
                bds = boundaries_dict.get(feat_idx, [])
                b_idx = bucketize_value(val, bds)
                combo_indices.append(b_idx)
                multi_head_out[i, t, feat_sub] = b_idx
            combos_set.add(tuple(combo_indices))

    return combos_set, multi_head_out

##########################################
# 2) Encode combos in parallel (PASS 2)
##########################################
def encode_combos_chunk(args):
    """
    args: (data_chunk, boundaries_dict, combo_dict)
    Return: (tokens_single, tokens_multi)
    """
    data_chunk, boundaries_dict, combo_dict = args
    tokens_single = np.zeros((data_chunk.shape[0], 20), dtype=np.int32)
    tokens_multi = np.zeros((data_chunk.shape[0], 20, 5), dtype=np.int32)

    for i in range(data_chunk.shape[0]):
        for t in range(20):
            combo_indices = []
            for feat_sub in range(5):
                feat_idx = feat_sub + 1
                val = data_chunk[i, t, feat_idx]
                bds = boundaries_dict.get(feat_idx, [])
                b_idx = bucketize_value(val, bds)

                combo_indices.append(b_idx)
                tokens_multi[i, t, feat_sub] = b_idx
            # single token
            combo_tuple = tuple(combo_indices)
            token_id = combo_dict[combo_tuple]  # guaranteed to exist now
            tokens_single[i, t] = token_id

    return tokens_single, tokens_multi

##########################################
# Helper to split data into chunks
##########################################
def chunk_data(array, num_chunks):
    """
    Return a list of sub-arrays splitted from 'array' into 'num_chunks' chunks
    as evenly as possible.
    """
    chunk_size = len(array) // num_chunks
    chunks = []
    start = 0
    for _ in range(num_chunks - 1):
        end = start + chunk_size
        chunks.append(array[start:end])
        start = end
    # last chunk
    chunks.append(array[start:])
    return chunks

##########################################
# Main
##########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries-file", required=True,
                        help="Pickle file with boundaries for each feature index.")
    parser.add_argument("--nprocs", type=int, default=4,
                        help="Number of processes to use.")
    args = parser.parse_args()

    # 1) Load train data
    if not os.path.isfile(TRAIN_DATA_PATH):
        print(f"Train data file not found: {TRAIN_DATA_PATH}")
        return
    with open(TRAIN_DATA_PATH, "rb") as f:
        train_data = pickle.load(f)
    if not (isinstance(train_data, np.ndarray) and train_data.ndim == 3 and train_data.shape[2] == FEATURE_DIM):
        print(f"Unexpected shape: {train_data.shape}. Exiting.")
        return
    print(f"Loaded train data shape = {train_data.shape}")

    # 2) Load boundaries
    if not os.path.isfile(args.boundaries_file):
        print(f"Boundaries file not found: {args.boundaries_file}")
        return
    with open(args.boundaries_file, "rb") as f:
        boundaries_dict = pickle.load(f)
    print(f"Loaded boundaries from {args.boundaries_file}")

    # Base RTT => shape (N,20), keep float
    base_rtt_data = train_data[:, :, 0].copy()

    # We'll parallelize in two passes for single combo approach:

    ###############################
    # PASS 1: gather combos + multi_head data
    #   -> unify combos
    ###############################
    nprocs = min(args.nprocs, cpu_count())
    data_chunks = chunk_data(train_data, nprocs)

    # Build arg list for each process
    pass1_args = [(chunk, boundaries_dict) for chunk in data_chunks]

    with Pool(processes=nprocs) as pool:
        results = pool.map(gather_combos_chunk, pass1_args)

    # results => list of (set_of_combos, multi_head_out)
    all_sets = []
    multi_head_chunks = []
    for combo_set, multi_head_out in results:
        all_sets.append(combo_set)
        multi_head_chunks.append(multi_head_out)

    # unify combos
    global_combos = set()
    for s in all_sets:
        global_combos |= s  # union

    # build combo_dict (combo->ID)
    combo_dict = {}
    combo_list = []
    next_id = 0
    for c in global_combos:
        combo_dict[c] = next_id
        combo_list.append(c)
        next_id += 1
    print(f"Single-combo vocabulary size = {len(combo_dict)}")

    # We already have multi_head_out from pass 1, so we can just vstack them
    tokens_multi_pass1 = np.concatenate(multi_head_chunks, axis=0)  # shape (N,20,5)
    print(f"multi_head shape after pass1 = {tokens_multi_pass1.shape}")

    ###############################
    # PASS 2: build single tokens
    ###############################
    # We do the actual mapping to token_id for each chunk
    pass2_args = []
    for chunk in data_chunks:
        pass2_args.append((chunk, boundaries_dict, combo_dict))

    with Pool(processes=nprocs) as pool:
        pass2_results = pool.map(encode_combos_chunk, pass2_args)

    # pass2_results => list of (tokens_single, tokens_multi) but we don't need tokens_multi from pass2 if we trust pass1
    # However, for consistency, let's gather it. But we can skip pass1 for multi if we want the final approach to be consistent.
    tokens_single_list = []
    tokens_multi_list = []
    for (tsingle, tmulti) in pass2_results:
        tokens_single_list.append(tsingle)
        tokens_multi_list.append(tmulti)

    tokens_single = np.concatenate(tokens_single_list, axis=0)  # shape (N,20)
    tokens_multi = np.concatenate(tokens_multi_list, axis=0)    # shape (N,20,5)

    # If you prefer pass1's multi_head, you can use tokens_multi_pass1. 
    # But let's keep pass2 for consistency:
    print(f"tokens_single shape = {tokens_single.shape}")
    print(f"tokens_multi shape  = {tokens_multi.shape}")

    ###############################
    # Save output
    ###############################
    boundary_base = ntpath.basename(args.boundaries_file)
    boundary_base_noext = os.path.splitext(boundary_base)[0]

    out_single_path = os.path.join(
        COMBINED_DIR, f"{boundary_base_noext}-tokenized-single.pkl"
    )
    single_data_dict = {
        "base_rtt": base_rtt_data,
        "tokens_single": tokens_single,
        "combo_dict": combo_dict,
        "combo_list": combo_list
    }
    with open(out_single_path, "wb") as f:
        pickle.dump(single_data_dict, f)
    print(f"Saved single-combo tokenized data => {out_single_path}")

    out_multi_path = os.path.join(
        COMBINED_DIR, f"{boundary_base_noext}-tokenized-multi.pkl"
    )
    multi_data_dict = {
        "base_rtt": base_rtt_data,
        "tokens_multi": tokens_multi,
        "boundaries_dict": boundaries_dict
    }
    with open(out_multi_path, "wb") as f:
        pickle.dump(multi_data_dict, f)
    print(f"Saved multi-head tokenized data => {out_multi_path}")

if __name__ == "__main__":
    main()
