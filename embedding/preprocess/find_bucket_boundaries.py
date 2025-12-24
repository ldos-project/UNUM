import os
import pickle
import numpy as np
from glob import glob
import argparse

try:
    from cuml.cluster import KMeans as GPUKMeans
    import cupy as cp
    cp.cuda.Device(1).use()
    CUMl_AVAILABLE = True
except ImportError:
    CUMl_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

DATASET_DIR = "/datastor1/janec/datasets"
COMBINED_DIR = "/datastor1/janec/datasets/combined"
BOUNDARY_DIR = "/datastor1/janec/datasets/boundaries"

TRAIN_DATA_PATH = os.path.join(COMBINED_DIR, "6col_20rtt_train_combined.p")

FEATURE_DIM = 6
MAX_SAMPLES_PER_FEATURE = 2000000

###############################################
#             Collect Feature Values
###############################################
def collect_feature_values(files):
    """
    Collect data for each sample, each feature.
    Returns: list of length FEATURE_DIM; each entry is a Python list of values.
    """
    feature_values = [[] for _ in range(FEATURE_DIM)]
    for f in files:
        with open(f, 'rb') as handle:
            data = pickle.load(handle)  # shape: (N, 20, 6) expected
            if not isinstance(data, np.ndarray) or len(data.shape) != 3 or data.shape[2] != FEATURE_DIM:
                continue

            # Flatten all 20 RTTs for each sample
            for i in range(FEATURE_DIM):
                vals = data[:, :, i].flatten()
                feature_values[i].extend(vals)
    return feature_values

###############################################
#           Binning Helper Functions
###############################################
def get_quantile_boundaries(arr, num_buckets):
    """Quantile-based bucket edges, removing duplicates."""
    percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]
    bucket_edges = np.percentile(arr, percentiles)
    
    # Convert to a list & remove duplicates
    unique_edges = sorted(set(bucket_edges))
    return unique_edges

def get_freedman_diaconis_boundaries(arr):
    """
    Freedman–Diaconis formula for bin width:
      bin_width = 2 * IQR / cbrt(N)
    => num_bins = (max - min) / bin_width
    This can create a variable number of bins.
    """
    arr = np.sort(arr)
    n = len(arr)
    if n < 2:
        return []

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    bin_width = 2.0 * iqr / (n ** (1.0/3.0))
    if bin_width <= 0:
        return []

    data_min, data_max = arr[0], arr[-1]
    num_bins = int(np.ceil((data_max - data_min) / bin_width))
    if num_bins < 1:
        num_bins = 1

    boundaries = []
    for b in range(1, num_bins):
        boundary = data_min + b * bin_width
        if boundary < data_max:
            boundaries.append(boundary)
        else:
            break
    return boundaries


def get_kmeans_boundaries_gpu(arr, n_clusters):
    data_gpu = cp.asarray(arr.reshape(-1, 1))
    kmeans = GPUKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_gpu)

    centers = cp.asnumpy(kmeans.cluster_centers_).flatten()
    centers = np.sort(centers)

    boundaries = []
    for c1, c2 in zip(centers[:-1], centers[1:]):
        midpoint = (c1 + c2) / 2.0
        boundaries.append(midpoint)
    return boundaries

###############################################
#            Compute Bucket Boundaries
###############################################
def compute_bucket_boundaries(feature_values, method, num_buckets):
    """
    For each feature i, produce a list of boundary values based on chosen method.
    If method == 'kmeans', we pick best silhouette-based cluster count for each feature 
    from find_optimal_k_elbow_silhouette.
    """
    boundaries = {}
    for i, values in enumerate(feature_values):
        # If your Feature 1 is constant => optionally skip if you want:
        if i == 0:
        #     boundaries[i] = []
            continue

        arr = np.array(values)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            boundaries[i] = []
            continue
        if arr.shape[0] > MAX_SAMPLES_PER_FEATURE:
            arr = np.random.choice(arr, size=MAX_SAMPLES_PER_FEATURE, replace=False)

        if method == "quantile":
            edges = get_quantile_boundaries(arr, num_buckets)
        elif method == "histogram":
            MAX_HIST_SAMPLES = 10_000
            if arr.shape[0] > MAX_HIST_SAMPLES:
                arr = np.random.choice(arr, size=MAX_HIST_SAMPLES, replace=False)
            edges = get_freedman_diaconis_boundaries(arr)
        elif method == "kmeans":
            edges = get_kmeans_boundaries_gpu(arr, num_buckets)
        else:
            raise ValueError(f"Unknown bucket method: {method}")

        boundaries[i] = edges

    return boundaries

###############################################
#                     Main
###############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find bucket boundaries for (N, 20, 6) data, with train/test split")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Path to directory of .p files")
    parser.add_argument("--method", choices=["quantile","histogram","kmeans"], default="quantile",
                        help="Bucketization method to use")
    parser.add_argument("--num-buckets", type=int, default=1000,
                        help="Max number of buckets (quantile) or max_k (kmeans)")

    args = parser.parse_args()

    # 1) Load or create train/test .p files
    if os.path.exists(TRAIN_DATA_PATH):
        print(f"Train/test files found:\n  {TRAIN_DATA_PATH}")
    else:
        print(f"Train/test files not found: {TRAIN_DATA_PATH}")
        exit(1)

    # 3) Collect feature values
    feature_values = collect_feature_values([TRAIN_DATA_PATH])

    # 4) Compute boundaries
    bucket_boundaries = compute_bucket_boundaries(feature_values, args.method, args.num_buckets)

    # 5) Save boundaries under /datastore1/janec/datasets/boundaries
    # os.makedirs(BOUNDARY_DIR, exist_ok=True)

    # e.g. boundaries-quantile1000.pkl or boundaries-kmeans.pkl
    if args.method == "quantile" or args.method == "kmeans":
        boundaries_filename = f"boundaries-{args.method}{args.num_buckets}.pkl"
    else:
        boundaries_filename = f"boundaries-{args.method}.pkl"
    final_save_path = os.path.join(BOUNDARY_DIR, boundaries_filename)

    with open(final_save_path, "wb") as f:
        pickle.dump(bucket_boundaries, f)

    print(f"Bucket boundaries saved to: {final_save_path}")
    for i, b in bucket_boundaries.items():
        print(f"Feature {i} boundaries: {b}")
