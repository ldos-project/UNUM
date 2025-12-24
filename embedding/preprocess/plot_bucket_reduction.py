import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

BOUNDARY_DIR = "/datastor1/janec/datasets/boundaries/"
FEATURES_TO_PLOT = [1, 2, 3, 4, 5]  # Skip feature 0

def load_boundaries(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def count_boundaries(boundaries):
    return [len(boundaries.get(i, [])) for i in range(max(FEATURES_TO_PLOT)+1)]

def plot_bucket_reduction(file_names, orig_counts, merged_counts):
    num_files = len(file_names)
    num_features = len(FEATURES_TO_PLOT)

    fig, axs = plt.subplots(1, num_features, figsize=(5 * num_features, 6))
    # fig.suptitle("Bucket Reduction per Feature", fontsize=16)

    for feature in FEATURES_TO_PLOT:
        orig = [counts[feature] for counts in orig_counts]
        merged = [counts[feature] for counts in merged_counts]
        x = np.arange(len(file_names))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - 0.2, orig, width=0.4, label='Original')
        ax.bar(x + 0.2, merged, width=0.4, label='Merged')

        # ax.set_title(f"Feature {feature} Bucket Reduction", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(file_names, rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylabel("Num Buckets", fontsize=18)

        if feature == 5:
            ax.set_yscale("log")

        ax.legend(fontsize=14)
        plt.tight_layout()

        out_path = f"fig/bucket_reduction_feature_{feature}.png"
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        plt.close()
if __name__ == "__main__":
    original_files = sorted(glob(os.path.join(BOUNDARY_DIR, "boundaries-*.pkl")))
    original_files = [f for f in original_files if "-merged" not in f]

    file_names = []
    original_counts = []
    merged_counts = []

    for orig_path in original_files:
        base_name = os.path.basename(orig_path).replace(".pkl", "")
        merged_path = os.path.join(BOUNDARY_DIR, f"{base_name}-merged.pkl")
        if not os.path.exists(merged_path):
            continue

        orig = load_boundaries(orig_path)
        merged = load_boundaries(merged_path)

        file_names.append(base_name.replace("boundaries-", ""))
        original_counts.append(count_boundaries(orig))
        merged_counts.append(count_boundaries(merged))

    plot_bucket_reduction(file_names, original_counts, merged_counts)
