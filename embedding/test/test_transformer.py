import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from embedding.utils.models import Seq2SeqWithEmbeddingmodClassMultiHead
from embedding.utils.utils import create_mask
from tqdm import tqdm
import argparse

def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.set_size_inches(5,2)
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename,format="png",dpi=600, bbox_inches=bbox)

parser = argparse.ArgumentParser()
parser.add_argument('--load_existing', action='store_true', help='Load results from existing .pkl files')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to test')
args = parser.parse_args()

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PAD_IDX = 2
PREDICTION_LENGTH = 10
NUM_FEATURES = 5  # exclude base RTT

TEST_SETS = [
    "6col_20rtt_cc-real_test_combined.p",
    "6col_20rtt_cc-synthetic_test_combined.p",
    "6col_20rtt_pensieve-real_test_combined.p",
    "6col_20rtt_pensieve-synthetic_test_combined.p",
]
TEST_DIR = "/datastor1/janec/datasets/combined"
BOUNDARY_DIR = "/datastor1/janec/datasets/combined"
MODEL_DIR = "/datastor1/janec/complete-models"

BOUNDARY_FILES = {
    "quantile30": "boundaries-quantile30-merged-tokenized-multi.pkl",
    "quantile50": "boundaries-quantile50-merged-tokenized-multi.pkl",
    "quantile80": "boundaries-quantile80-merged-tokenized-multi.pkl",
    "quantile100": "boundaries-quantile100-merged-tokenized-multi.pkl",
    "kmeans30": "boundaries-kmeans30-merged-tokenized-multi.pkl",
    "kmeans50": "boundaries-kmeans50-merged-tokenized-multi.pkl",
    "kmeans80": "boundaries-kmeans80-merged-tokenized-multi.pkl",
    "kmeans100": "boundaries-kmeans100-merged-tokenized-multi.pkl",
    "histogram": "boundaries-histogram-merged-tokenized-multi.pkl"
}

MODELS_TO_TEST = [
    os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".p")
]

def get_clean_label(model_name):
    if "kmeans30" in model_name:
        return "KMeans-30"
    elif "kmeans50" in model_name:
        return "KMeans-50"
    elif "kmeans80" in model_name:
        return "KMeans-80"
    elif "kmeans100" in model_name:
        return "KMeans-100"
    elif "quantile30" in model_name:
        return "Quantile-30"
    elif "quantile50" in model_name:
        return "Quantile-50"
    elif "quantile80" in model_name:
        return "Quantile-80"
    elif "quantile100" in model_name:
        return "Quantile-100"
    elif "histogram" in model_name:
        return "Histogram"
    else:
        return model_name  # fallback to full name if unknown

def compute_cdf(distances):
    sorted_vals = np.sort(distances)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, cdf

def get_true_bucket_index(value, boundaries):
    if not boundaries:
        return 0
    idx = np.searchsorted(boundaries, value, side='left')
    return idx

def make_bucket_midpoints(boundaries):
    if not boundaries:
        return []

    midpoints = []

    # Clamp gap if extremely large or suspicious
    left_gap = boundaries[1] - boundaries[0] if len(boundaries) > 1 else 1
    if abs(left_gap) > 10:
        left_gap = 0  # prevent wild extrapolation

    # Leftmost bucket midpoint (extrapolated)
    leftmost = boundaries[0] - left_gap / 2
    midpoints.append(leftmost)

    # Mid-bucket midpoints
    for i in range(len(boundaries) - 1):
        midpoints.append((boundaries[i] + boundaries[i + 1]) / 2)

    right_gap = boundaries[-1] - boundaries[-2] if len(boundaries) > 1 else 1
    if abs(right_gap) > 10:
        right_gap = 0

    # Rightmost bucket midpoint (extrapolated)
    rightmost = boundaries[-1] + right_gap / 2
    midpoints.append(rightmost)

    return midpoints



for test_set in TEST_SETS:
    tag = test_set.replace(".p", "")

    if args.load_existing:
        cdf_val_results = {feat: pickle.load(open(f"log/transformer-distance/cdf_val_results_{tag}_feat{feat}.pkl", "rb")) for feat in range(1, NUM_FEATURES + 1)}
        cdf_bucket_dist_results = {feat: pickle.load(open(f"log/transformer-distance/cdf_bucket_dist_results_{tag}_feat{feat}.pkl", "rb")) for feat in range(1, NUM_FEATURES + 1)}
        print(f"=== Loaded existing results for {test_set} ===")
        for k in cdf_val_results[3].keys():
            print(f"Model: {k} -> {get_clean_label(k)}")
        # print(f"Value distance results for feature 3: {cdf_val_results[3].keys()}")
    else:
    
        print(f"=== Running on test set: {test_set} ===")
        with open(os.path.join(TEST_DIR, test_set), "rb") as f:
            test_data_np = pickle.load(f)
            test_data_np = np.concatenate(test_data_np, axis=0)

        # Randomly sample 30,000 rows without replacement
        sample_indices = np.random.choice(test_data_np.shape[0], size=args.num_samples, replace=False)
        test_data_np = test_data_np[sample_indices]

        # Convert to tensor
        test_data = torch.from_numpy(test_data_np).float().to(DEVICE)
        
        cdf_val_results = {feat: {} for feat in range(1, NUM_FEATURES + 1)}
        cdf_bucket_dist_results = {feat: {} for feat in range(1, NUM_FEATURES + 1)}

        for model_path in MODELS_TO_TEST:
            model_name = os.path.basename(model_path)

            matched_boundary_key = None
            for key in BOUNDARY_FILES:
                if key in model_name:
                    matched_boundary_key = key
                    break
            if matched_boundary_key is None:
                print(f"Skipping {model_name} (no matching boundary key)")
                continue

            boundaries_path = os.path.join(BOUNDARY_DIR, BOUNDARY_FILES[matched_boundary_key])
            with open(boundaries_path, "rb") as f:
                boundaries_dict = pickle.load(f)["boundaries_dict"]
            print(f"Boundaries loaded from {boundaries_path}")
            bucket_midpoints = {
                feat: make_bucket_midpoints(boundaries)
                for feat, boundaries in boundaries_dict.items()
            }
            model = torch.load(model_path, map_location=DEVICE)
            model.eval()

            val_distances_by_feat = {feat: [] for feat in range(1, NUM_FEATURES + 1)}
            bucket_distances_by_feat = {feat: [] for feat in range(1, NUM_FEATURES + 1)}


            with torch.no_grad():
                for i in tqdm(range(test_data.shape[0]), desc=f"{test_set}-{model_name}"):
                    sample = test_data[i:i+1]
                    enc_input = sample[:, :-PREDICTION_LENGTH, :]
                    dec_input = 1.5 * torch.ones((1, PREDICTION_LENGTH, 6)).to(DEVICE)
                    expected_output = sample[:, -PREDICTION_LENGTH:, :]

                    src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, PAD_IDX, DEVICE)
                    pred = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)

                    for t in range(PREDICTION_LENGTH):
                        for feat in range(NUM_FEATURES):
                            feat_idx = feat + 1
                            bucket_idx = pred[0, t, feat].argmax().item()
                            midpoints = bucket_midpoints[feat_idx]
                            if not midpoints:
                                continue  # skip this feature if midpoints list is empty

                            bucket_idx = min(bucket_idx, len(midpoints) - 1)
                            predicted_value = midpoints[bucket_idx]
                            true_value = expected_output[0, t, feat_idx].item()
                            true_bucket = get_true_bucket_index(true_value, boundaries_dict[feat_idx])
                            bucket_dist = abs(bucket_idx - true_bucket)
                            
                            # --- Clamp for Feature 3 ---
                            if feat_idx == 3:
                                max_reasonable_value = 10.0
                                true_value = min(true_value, max_reasonable_value)
                                predicted_value = min(predicted_value, max_reasonable_value)

                            val_dist = abs(predicted_value - true_value)
                            
                            if val_dist > 1000 and feat_idx == 3:
                                print("model_name:", model_name)
                                print(f"Feature {feat_idx} - Sample {i} - Bucket index: {bucket_idx}, True bucket: {true_bucket}")
                                print(f"Mid point value 0: {midpoints[0]}, Mid point value 1: {midpoints[1]}")
                                print(f"Mid point value -2: {midpoints[-2]}, Mid point value -1: {midpoints[-1]}")
                                print(f"Value distance too large: {val_dist} (predicted: {predicted_value}, true: {true_value})")
                                print(f"Boundaries: {boundaries_dict[feat_idx]}")
                                exit(1)

                            val_distances_by_feat[feat_idx].append(val_dist)
                            bucket_distances_by_feat[feat_idx].append(bucket_dist)
            
            for feat in range(1, NUM_FEATURES + 1):
                cdf_val_results[feat][model_name] = compute_cdf(val_distances_by_feat[feat])
                cdf_bucket_dist_results[feat][model_name] = compute_cdf(bucket_distances_by_feat[feat])


        for feat in range(1, NUM_FEATURES + 1):
            with open(f"cdf_val_results_{tag}_feat{feat}.pkl", "wb") as f:
                pickle.dump(cdf_val_results[feat], f)
            with open(f"cdf_bucket_dist_results_{tag}_feat{feat}.pkl", "wb") as f:
                pickle.dump(cdf_bucket_dist_results[feat], f)

    for feat in range(1, NUM_FEATURES + 1):
        # Value distance CDF
        plt.figure(figsize=(10, 6))
        for model_name, (distances, cdf) in cdf_val_results[feat].items():
            label = get_clean_label(model_name)
            # if not any(k in model_name for k in ["kmeans30", "kmeans100", "quantile100"]):
            #     continue
            distances = np.clip(distances, 1e-3, None)
            plt.plot(distances, cdf, label=label)
        #plt.title(f"Feature {feat}: CDF of Absolute Value Distance ({tag})")
        plt.xlabel("Absolute Value Distance", fontsize=28, weight="normal")
        plt.ylabel("CDF", fontsize=28, weight="normal")
        plt.grid(True)
        plt.legend(prop={'size':20,'weight':'book'})
        plt.xscale("log")
        plt.xticks([1e-3, 1, 10, 100, 1000])
        plt.gca().set_xticklabels(['0', '1', '10', '100', '1000'])
        
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label1.set_fontsize(28)
            tick.label1.set_fontweight("normal")
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label1.set_fontsize(28)
            tick.label1.set_fontweight("normal")
        plt.tight_layout()
        plt.savefig(f"fig/transformer-distance/model_cdf_value_distance_{tag}_feat{feat}.png")
        # Bucket index distance CDF
        plt.figure(figsize=(10, 6))
        # sorted_keys = sorted(cdf_bucket_dist_results[feat].keys())
        # for model_name in sorted_keys:
        #     # distances, cdf = cdf_bucket_dist_results[feat][model_name]
        #     label = get_clean_label(model_name)
        #     # if not any(k in model_name for k in ["kmeans30", "kmeans100", "quantile100"]):
        #     #     continue
        #     plt.plot(distances, cdf, label=label)
        for model_name, (distances, cdf) in cdf_bucket_dist_results[feat].items():
            label = get_clean_label(model_name)
            # if not any(k in model_name for k in ["kmeans30", "kmeans100", "quantile100"]):
            #     continue
            plt.plot(distances, cdf, label=label)
        #plt.title(f"Feature {feat}: CDF of Bucket Index Distance ({tag})")
        plt.xlabel("Bucket Index Distance",fontsize=18, weight="normal")
        plt.ylabel("CDF", fontsize=22, weight="normal")
        plt.grid(True)
        plt.legend(ncol=1, prop={'size':20,'weight':'book'}, loc='center left', bbox_to_anchor=(0.445, 0.5))
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight("normal")
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label1.set_fontsize(22)
            tick.label1.set_fontweight("normal")
        plt.tight_layout()
        plt.savefig(f"fig/transformer-distance/model_cdf_bucket_distance_{tag}_feat{feat}.png")
        
        # handles, labels = plt.gca().get_legend_handles_labels()
        #import pdb; pdb.set_trace()
        # fig = plt.figure(frameon=False)
        # axe = plt.Axes(fig, [0., 0., 1., 1.])
        # axe.set_axis_off()
        # fig.add_axes(axe)
        # legend = axe.legend(handles, labels, loc='upper left', ncol=3, prop={'size':22,'weight':'book'},
        #                      frameon=True, borderaxespad=0.)
        # plt.margins(0,0)
        # axe.axes.xaxis.set_visible(False)
        # axe.axes.yaxis.set_visible(False)
        # axe.axes.xaxis.set_ticks([])
        # axe.axes.yaxis.set_ticks([])
        #export_legend(legend, filename=f"fig/transformer-distance/new_legend.png")
        #plt.close('all')

