import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import silhouette_score

try:
    from cuml.cluster import KMeans as GPUKMeans
    import cupy as cp
    cp.cuda.Device(1).use()
except ImportError:
    raise RuntimeError("cuML is not available.")

DATASET_DIR = "/datastor1/janec/datasets/raw"
COMBINED_DIR = "/datastor1/janec/datasets/combined"

FEATURE_DIM = 6

def find_optimal_k_elbow_silhouette_gpu(feature_idx, data, max_k=10):
    data_gpu = cp.asarray(data)

    inertia_values = []
    silhouette_values = []
    ks = range(2, max_k + 1)

    best_k_sil = 2
    best_sil_score = -1

    for k in ks:
        kmeans = GPUKMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_gpu)
        inertia_values.append(float(kmeans.inertia_))

        labels = kmeans.labels_.get()
        sample_data = cp.asnumpy(data_gpu[:10000]) if data_gpu.shape[0] > 10000 else cp.asnumpy(data_gpu)
        sample_labels = labels[:sample_data.shape[0]]

        if len(np.unique(sample_labels)) > 1:
            sil = silhouette_score(sample_data, sample_labels)
        else:
            sil = -1
        silhouette_values.append(sil)

        if sil > best_sil_score:
            best_sil_score = sil
            best_k_sil = k

    diffs = np.diff(inertia_values)
    elbow_idx = np.argmax(diffs)
    best_k_elbow = ks[elbow_idx]

    # Save plots
    plt.figure()
    plt.plot(ks, inertia_values, marker='o')
    plt.xlabel("Number of Clusters", fontsize=20)
    plt.ylabel("Inertia", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"fig/{feature_idx}_elbow_method.png")
    plt.close()

    plt.figure()
    plt.plot(ks, silhouette_values, marker='o')
    plt.xlabel("Number of Clusters", fontsize=20)
    plt.ylabel("Silhouette Score", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f"fig/{feature_idx}_silhouette_scores.png")
    plt.tight_layout()
    plt.close()

    return best_k_elbow, best_k_sil


def load_or_create_train_test():
    """
    For a given scenario:
    - Filter matching files
    - Split each file into 80% train, 20% test
    - Combine all train splits into one dataset
    - Combine all test splits into one test dataset
    - Save both into COMBINED_DIR

    Returns:
        combined_train_data (np.ndarray),
        combined_test_data (np.ndarray)
    """
    scenario_filters = {
        "cc-synthetic": lambda fname: "20_synthetic" in fname,
        "cc-real": lambda fname: "20_real" in fname,
        "pensieve-synthetic": lambda fname: "pensieve" in fname and "synthetic" in fname,
        "pensieve-real": lambda fname: "pensieve" in fname and "real" in fname,
    }

    os.makedirs(COMBINED_DIR, exist_ok=True)

    all_pickle_files = glob(os.path.join(DATASET_DIR, "*.p"))
    matched_files = {}
    for f in all_pickle_files:
        for scenario in scenario_filters:
            if scenario_filters[scenario](os.path.basename(f)):
                if scenario not in matched_files:
                    matched_files[scenario] = []
                matched_files[scenario].append(f)
                print(f"Found matching file for scenario '{scenario}': {f}")

    if not matched_files:
        print(f"No matching files found for scenario: {scenario}")
        return None, None

    
    train_splits = []

    for scenario, files in matched_files.items():
        test_splits = []
        for f in files:
            with open(f, 'rb') as handle:
                data = pickle.load(handle)

            if not (isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[2] == FEATURE_DIM):
                print(f"Skipping invalid file: {f}")
                continue

            np.random.shuffle(data)
            cutoff = int(0.8 * len(data))
            train_split = data[:cutoff]
            test_split = data[cutoff:]

            train_splits.append(train_split)
            test_splits.append(test_split)
        test_path = os.path.join(COMBINED_DIR, f"6col_20rtt_{scenario}_test_combined.p")
        with open(test_path, "wb") as tf:
            pickle.dump(test_splits, tf)
            print(f"Saved test data of length {len(test_splits)} for scenario '{scenario}' to {test_path}")
        combined_train = np.concatenate(train_splits, axis=0)
    train_path = os.path.join(COMBINED_DIR, f"6col_20rtt_train_combined.p")
    with open(train_path, "wb") as tf:
        pickle.dump(combined_train, tf)
        print(f"Saved train data of shape {combined_train.shape} for scenario '{scenario}' to {train_path}")
    
    return combined_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal K from training dataset by scenario")
    parser.add_argument("--max-k", type=int, default=100, help="Maximum number of clusters to test")

    args = parser.parse_args()
    train_path = os.path.join(COMBINED_DIR, f"6col_20rtt_train_combined.p")

    if os.path.exists(train_path):
        print(f"Train/test files found:\n  {train_path}")
        train_data = pickle.load(open(train_path, "rb"))
    else:
        train_data = load_or_create_train_test()
    if train_data is None:
        exit(1)

    for feature_idx in range(FEATURE_DIM):
        
        if feature_idx == 0:
            continue

        values = train_data[:, :, feature_idx].flatten()
        values = values[~np.isnan(values)]

        MAX_KMEANS_SAMPLES = 2_000_000
        if values.shape[0] > MAX_KMEANS_SAMPLES:
            print(f"Subsampling from {values.shape[0]} to {MAX_KMEANS_SAMPLES} samples for feature {feature_idx}")
            values = np.random.choice(values, size=MAX_KMEANS_SAMPLES, replace=False)


        if len(values) == 0:
            print("No valid values for clustering.")
            exit(1)
        print(f"Running KMeans clustering on {len(values)} values for feature {feature_idx}...")
        best_k_elbow, best_k_sil = find_optimal_k_elbow_silhouette_gpu(feature_idx, values.reshape(-1, 1), max_k=args.max_k)

        print(f"Feature {feature_idx}:")
        print(f"  Best k (elbow):      {best_k_elbow}")
        print(f"  Best k (silhouette): {best_k_sil}")
