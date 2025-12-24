import os
import pickle
import numpy as np
from glob import glob

MERGE_THRESHOLD = 0.01
BOUNDARY_DIR = "/datastor1/janec/datasets/boundaries/"

def merge_boundaries(boundaries, threshold=MERGE_THRESHOLD):
    merged = {}
    for i, edges in boundaries.items():
        if not edges:
            merged[i] = []
            continue
        sorted_edges = sorted(edges)
        new_edges = [sorted_edges[0]]
        for val in sorted_edges[1:]:
            if abs(val - new_edges[-1]) >= threshold:
                new_edges.append(val)
        merged[i] = new_edges
    return merged

def print_comparison(original, merged):
    for i in range(6):
        orig_len = len(original.get(i, []))
        merged_len = len(merged.get(i, []))
        print(f"Feature {i}: {orig_len} → {merged_len} boundaries")

if __name__ == "__main__":
    boundary_files = sorted(glob(os.path.join(BOUNDARY_DIR, "boundaries-*.pkl")))

    for file_path in boundary_files:
        if "merged" in file_path:
            continue
        print(f"\nProcessing: {os.path.basename(file_path)}")

        # Load original boundaries
        with open(file_path, "rb") as f:
            original = pickle.load(f)

        # Merge boundaries
        merged = merge_boundaries(original)

        # Print before/after counts
        print_comparison(original, merged)

        # Save merged version
        base = os.path.basename(file_path).replace(".pkl", "-merged.pkl")
        out_path = os.path.join(BOUNDARY_DIR, base)
        with open(out_path, "wb") as f:
            pickle.dump(merged, f)
        print(f"Saved merged boundaries to: {out_path}")
