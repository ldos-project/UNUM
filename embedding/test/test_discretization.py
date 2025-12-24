import torch
import pickle
from embedding.utils.models import create_mask
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
# from runvocabmlp import MLP
import sys
import numpy as np

PAD_IDX = 2
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(DEVICE)
ITER = sys.argv[1]

bucket_boundaries_combined = {
    1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.29, 1.63, 2.12, 3.64, 4.02, 5.74, 8, 12, 14],
    2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5, 205, 395, 1206],
    3: [0.01, 0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
    4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
    5: [0.75, 1, 1.001, 1.003, 1.012, 1.25, 3.52, 4.7, 5.39, 6.26]
}

# Define fmin and fmax arrays based on the given min and max values
fmin = np.array([0.11613000184297562, 0.0010000000474974513, 0.0, 0.0, 0.0])  # Feature 1-5 min values
fmax = np.array([438.7537925, 75126.192, 4.144327163696289, 21.996290116813032, 3546.0])  # Feature 1-5 max values

def compute_l1_distances(true_vectors, predicted_vectors):
    """
    Compute the distance for each sample as the sum of absolute differences 
    between true and predicted indices across the 5 features.

    Parameters
    ----------
    true_vectors : ndarray of shape (N, 5)
    predicted_vectors : ndarray of shape (N, 5)

    Returns
    -------
    distances : ndarray of shape (N,)
        distances[i] = sum of abs differences for sample i
    """
    # Assuming both arrays have the same shape: (N, 5)
    distances = np.average(np.abs(true_vectors - predicted_vectors), axis=1)
    return distances

def compute_l1_distances_per_column(true_vectors, predicted_vectors):
    distances_dict = {}
    feature_name = ["sRTT", "Avg Tput", "Loss Rate"]
    for i in range(true_vectors.shape[1]):
        true_column = true_vectors[:, i]
        predicted_column = predicted_vectors[:, i]
        distances = np.average(np.abs(true_column - predicted_column))
        print(f"Feature {i} distances: {distances}")
        distances_dict[feature_name[i]] = distances
    return distances_dict

def bucketize_values(values, boundaries):
    """
    Bucketize values based on boundaries for each feature.

    Args:
    - values: ndarray of shape (N, 5), true or predicted values.
    - boundaries: Dictionary where each feature index maps to its bucket boundaries.

    Returns:
    - bucket_indices: ndarray of shape (N, 5), bucket indices for each value.
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()  # Ensure values is a NumPy array
    
    bucket_indices = np.zeros_like(values, dtype=int)
    for feature_idx, feature_boundaries in boundaries.items():
        feature_idx_offset = feature_idx - 1  # Feature indices in boundaries are 1-based
        bucket_indices[:, feature_idx_offset] = np.searchsorted(
            feature_boundaries, values[:, feature_idx_offset], side="right"
        )
    return bucket_indices


def compute_bucket_distances(true_vectors, predicted_vectors, boundaries):
    """
    Compute the bucket distance for each true-predicted pair.

    Args:
    - true_vectors: ndarray of shape (N, 5), true values.
    - predicted_vectors: ndarray of shape (N, 5), predicted values.
    - boundaries: Dictionary where each feature index maps to its bucket boundaries.

    Returns:
    - distances: ndarray of shape (N,), average bucket index difference for each pair.
    """
    # Ensure input is converted to NumPy arrays if necessary
    if isinstance(true_vectors, torch.Tensor):
        true_vectors = true_vectors.cpu().numpy()
    if isinstance(predicted_vectors, torch.Tensor):
        predicted_vectors = predicted_vectors.cpu().numpy()

    # print("True vectors: ", true_vectors)
    # print("Predicted vectors: ", predicted_vectors)

    # Bucketize both true and predicted values
    true_buckets = bucketize_values(true_vectors, boundaries)
    predicted_buckets = bucketize_values(predicted_vectors, boundaries)
    # print("True buckets: ", true_buckets)
    # print("Predicted buckets: ", predicted_buckets)

    # Calculate the index difference for each feature
    index_differences = np.abs(true_buckets - predicted_buckets)

    # Average the differences across all features
    distances = np.mean(index_differences, axis=1)

    return distances

    
def test_and_plot_distribution_multi_model(model_list, model_name_list, is_transformer_list, datasets, vocab_dict, prediction_len, batch_size=32, iteration=""):
    """
    Test multiple models and plot class distributions for true and predicted labels.
    
    Args:
    - model_list: List of trained models for testing.
    - model_name_list: List of model names for labeling.
    - is_transformer_list: List of booleans indicating whether the model is a transformer.
    - dataset: Dataset to test on.
    - vocab_dict: Dictionary mapping 44-size vectors to class indices.
    - prediction_len: Length of the prediction.
    - batch_size: Batch size.
    - vocab_size: Number of classes in the vocabulary.
    - iteration: Iteration number for saving files.
    """
    model_results = {}
    for model_name in model_name_list:
        if model_name == "Classification Transformer":
            model_results[model_name] = {'true_classes': [], 'predicted_classes': []}
        elif model_name == "Regression Transformer":
            model_results[model_name] = {'bucket_distances': []}

    num_batches = datasets[0].shape[0] // batch_size
    with torch.no_grad():
        for i, model in enumerate(model_list):
            dataset = torch.from_numpy(datasets[i])
            for batch in range(num_batches):
                input_batch = dataset[batch * batch_size:(batch + 1) * batch_size, :, :].clone()
                enc_input = input_batch[:, :-prediction_len, :].to(DEVICE)
                dec_input = (1.5 * torch.ones((batch_size, prediction_len, input_batch.shape[2]))).to(DEVICE)
                
                src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=DEVICE)
                expected_output = input_batch[:, -prediction_len:, :].to(DEVICE)

                batch_true_classes = []
                # Convert expected_output to class indices (ground truth)
                for j in range(batch_size):
                    for t in range(prediction_len):
                        if model_name_list[i] == 'Classification Transformer':
                            if expected_output.shape[-1] == 6:
                                vector = tuple(expected_output[j, t, 1:].tolist())
                            else:
                                vector = tuple(expected_output[j, t, :].tolist())
                            class_idx = vocab_dict.get(vector, -1)
                            if class_idx == -1:
                                print(f"Vocab dict key example: {list(vocab_dict.keys())[0]}")
                                raise ValueError(f"Vector not found in vocab_dict: {vector}")
                            batch_true_classes.append(class_idx)

                # Convert to tensor and move to device
                if model_name_list[i] == 'Classification Transformer':
                    batch_true_classes = torch.tensor(batch_true_classes, dtype=torch.long).to(DEVICE)
                    batch_true_classes = batch_true_classes.view(batch_size, prediction_len)

                # valid_mask = (expected_output[:, :, :].sum(dim=-1) != -1).to(DEVICE)

                # print("i:", i)
                # print("model_list: ", model_name_list)
                model_name = model_name_list[i]

                if is_transformer_list[i]:
                    # Forward pass through the transformer model to get the predicted probabilities
                    model_output = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
                else:
                    # Forward pass through the MLP model
                    enc_input_flat = enc_input.reshape(enc_input.shape[0], enc_input.shape[1] * enc_input.shape[2])
                    model_output = model(enc_input_flat)
                    model_output = model_output.view(batch_size, prediction_len, -1)  # Reshape to [batch_size, prediction_len, num_classes]

                
                if model_name_list[i] == 'Classification Transformer':
                        # Collect true and predicted classes
                    for t in range(prediction_len):
                        predicted_probs = model_output[:, t, :]  # Predicted probability for each time step
                        predicted_classes = torch.argmax(predicted_probs, dim=-1)  # Get the index of the max probability
                        
                        model_results[model_name]['true_classes'].extend(batch_true_classes[:, t].tolist())
                        model_results[model_name]['predicted_classes'].extend(predicted_classes.tolist())
                if model_name_list[i] == 'Regression Transformer':
                    expected_output = expected_output[:, :, 1:]
                    true_values = expected_output.cpu().numpy().reshape(-1, expected_output.shape[-1])
                    model_output = model_output[:, :, 1:]
                    predicted_values = model_output.cpu().numpy().reshape(-1, model_output.shape[-1])
                    # Apply the denormalization
                    true_values_denormalized = true_values * (fmax - fmin) + fmin  # Element-wise operation
                    predicted_values_denormalized = predicted_values * (fmax - fmin) + fmin  # Element-wise operation

                    # Calculate bucket distances for regression results
                    bucket_distances = compute_bucket_distances(true_values_denormalized, predicted_values_denormalized, bucket_boundaries_combined)

                    # Store results
                    # model_results[model_name]['true_values'].extend(true_values)
                    # model_results[model_name]['predicted_values'].extend(predicted_values)
                    model_results[model_name]['bucket_distances'].extend(bucket_distances)

    with open(str(ITER)+'iter-model-discretization-comparison-results.p', 'wb') as f_results:
        pickle.dump(model_results, f_results, pickle.HIGHEST_PROTOCOL)
    
    # with open(str(ITER)+'iter-model-discretization-comparison-results.p', 'rb') as f_results:
    #     model_results = pickle.load(f_results)
    
    # Plot the distributions for both models
    with open('NEWDatasets/combined-dataset-preprocessed/6col-VocabBackDict.p', 'rb') as f_vocab_back:
        vocab_back = pickle.load(f_vocab_back)

    plt.figure(figsize=(10, 6))
    cdf_dict = {}
    per_col_distances_dict = {}

    percentile_thresholds = [50, 75, 90, 95, 99, 99.9]
    
    # Plot PDF and CDF for each model
    for i, model_name in enumerate(model_name_list):
        if model_name == 'Classification Transformer':
            print(f"Processing bucket distances for {model_name}...")
        
            true_classes = model_results[model_name]['true_classes']
            predicted_classes = model_results[model_name]['predicted_classes']
            
            true_vectors = np.array([vocab_back[class_idx] for class_idx in true_classes])
            predicted_vectors = np.array([vocab_back[class_idx] for class_idx in predicted_classes])

            # total_distances, feature_distances = calculate_bucket_distances(true_vectors, predicted_vectors, bucket_boundaries_ccbench)
            if true_vectors.shape[-1] == 5:
                total_distances = compute_l1_distances(true_vectors[:, [0, 2, 3]], predicted_vectors[:, [0, 2, 3]])
                per_col_distances_dict[model_name] = compute_l1_distances_per_column(true_vectors[:, [0, 2, 3]], predicted_vectors[:, [0, 2, 3]])
            else:
                total_distances = compute_l1_distances(true_vectors, predicted_vectors)
                per_col_distances_dict[model_name] = compute_l1_distances_per_column(true_vectors, predicted_vectors)

        elif model_name == 'Regression Transformer':
            total_distances = model_results[model_name]['bucket_distances']

        # Plot the overall distribution for each model (PDF)
        unique_total_distances, total_counts = np.unique(total_distances, return_counts=True)
        total_distribution = total_counts / total_counts.sum() * 100
        # Cumulative sum to get CDF
        cdf = np.cumsum(total_counts / total_counts.sum() * 100)
        cdf_dict[model_name] = (unique_total_distances, cdf)

        for percentile in percentile_thresholds:
            # Find the distance where the CDF first reaches or exceeds the given percentile
            threshold_index = np.searchsorted(cdf, percentile, side="left")
            if threshold_index < len(unique_total_distances):
                distance_at_percentile = unique_total_distances[threshold_index]
            else:
                distance_at_percentile = unique_total_distances[-1]  # Use the maximum distance

            print(f"{percentile}th Percentile Distance for model {model_name}: {distance_at_percentile:.3f}")
        
        # PDF Plot
        plt.plot(unique_total_distances, total_distribution, label=model_name)  # Line plot for PDF
        print("Total distribution (PDF):", total_distribution)

    # Improved labels, legend, and formatting
    plt.xlabel('Average Bucket Distance', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', fontsize=14, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Reduce whitespace and save the plot
    plt.tight_layout(pad=1.0)
    plt.savefig(f'discretization_bucket_distance_comparison_PDF.png', dpi=300, bbox_inches='tight')
    plt.show()

    # CDF Plot
    plt.figure(figsize=(8, 6))

    for model_name in model_name_list:
        unique_total_distances, cdf = cdf_dict[model_name]
        # CDF Plot with thicker lines
        plt.plot(unique_total_distances, cdf, label=model_name, linewidth=3)
        print(f"Total CDF for {model_name}: {cdf}")

    # Improved labels, legend, and formatting
    plt.xlabel('Average Bucket Index Distance', fontsize=20)
    plt.ylabel('CDF', fontsize=20)
    # plt.xlim([0, 4])
    # plt.xlim([2, 9])
    # plt.ylim([99.4, 100])
    # plt.xticks([0, 1, 2, 3, 4], fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right', fontsize=18, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Reduce whitespace and save the plot
    plt.tight_layout(pad=1.0)
    plt.savefig(f'discretization_bucket_distance_comparison_CDF_tail_zoom.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Per col Bar Plot
    # plt.figure(figsize=(8, 6))
    
    # feature_names = list(per_col_distances_dict.values())[0].keys()
    # print("Feature names: ", feature_names)
    # x = np.arange(len(feature_names))  # X-axis positions for the features
    # width = 0.35  # Width of the bars

    # for model_name in model_name_list:
    #     per_col_distances = [per_col_distances_dict[model_name][feature] for feature in feature_names]
    #     print(f"Per column distances for {model_name}: {per_col_distances}")
    #     if model_name == 'Transformer with 3 Features':
    #         plt.bar(x - width/2, per_col_distances, width, label="3 Features")
    #     else:
    #         plt.bar(x + width/2, per_col_distances, width, label="6 Features")

    # # Add labels and title
    # plt.xlabel('Feature', fontsize=20)
    # plt.ylabel('Average Bucket Index Distance', fontsize=20)
    # plt.xticks(x, feature_names, fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.legend(fontsize=18)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # Save and show the plot
    # plt.tight_layout(pad=1.0)
    # plt.savefig(f'discretization_per_column_distances_comparison.png')
    # plt.show()

def process_and_calculate_distances_multiple_models(bucket_boundaries):
    """
    Calculate and plot bucket distances for multiple models.

    Args:
    - true_vectors: The true vectors (ground truth).
    - predicted_vectors_dict: A dictionary where each key is a model name, and the value is the predicted vectors by that model.
    - bucket_boundaries: Dictionary of bucket boundaries for each feature.
    - model_names: List of model names for plotting.
    """
    # Dictionary to store total and feature distances for each model
    model_distances = {}
    with open('./NEWDatasets/'+str(ITER)+'iter-model-results.p', 'rb') as f_results:
        model_results = pickle.load(f_results)
    
    # Plot the distributions for both models
    with open('NEWDatasets/FullDataset1x-filtered1-bucketized-VocabBackDict.p', 'rb') as f_vocab_back:
        vocab_back_dict = pickle.load(f_vocab_back)

    plt.figure(figsize=(10, 6))
    for model_name in model_results.keys():
        print(f"Processing bucket distances for {model_name}...")
        true_classes = model_results[model_name]['true_classes']
        predicted_classes = model_results[model_name]['predicted_classes']
        true_vectors = np.array([vocab_back_dict[class_idx] for class_idx in true_classes])
        predicted_vectors = np.array([vocab_back_dict[class_idx] for class_idx in predicted_classes])

        # Calculate distances
        # total_distances, feature_distances = calculate_bucket_distances(true_vectors, predicted_vectors, bucket_boundaries_1x)
        total_distances = compute_l1_distances(true_vectors, predicted_vectors)
        # model_distances[model_name] = {'total': total_distances , 'features': feature_distances}
    
    # Plot feature-wise bucket distance distribution for each model
    num_features = len(bucket_boundaries)
    
    for feature_idx in range(num_features):
        plt.figure(figsize=(10, 6))
        for model_name in model_distances.keys():
            distances = model_distances[model_name]['features'][feature_idx]
            unique_distances, counts = np.unique(distances, return_counts=True)
            distribution = counts / counts.sum() * 100
            
            # Plot distribution for this model
            plt.bar(unique_distances, distribution, alpha=0.6, label=model_name)  # Use alpha for transparency
            
            zero_distance_percentage = distribution[unique_distances == 0].sum()
            print(f"  {model_name} feature {feature_idx} accuracy: {zero_distance_percentage:.2f}% ")
        
        plt.xlabel('Distance from True Bucket')
        plt.ylabel('Percentage')
        plt.title(f'Distribution of Bucket Distances for Feature {feature_idx}')
        plt.grid(True)
        plt.legend(loc='upper right')  # Add legend for model names
        plt.savefig(f'{ITER}iter-bucket-distance-feature-{feature_idx}.png')
        plt.close()  # Close the figure to save memory


# Test the model
with open('/u/janechen/Documents/Transformer-training/NEWDatasets/combined-dataset-preprocessed/6col-VocabDict.p', 'rb') as f_vocab:
    classification_vocab_dict = pickle.load(f_vocab)

regression_model_name = 'Regression Transformer'
regression_model = torch.load('/datastor1/janec/complete-models/Checkpoint-Regression_Transformer3_256_10_10_32_4_lr_0.0001-999iter.p', map_location=DEVICE)
classification_model_name = 'Classification Transformer'
classification_model = torch.load('/datastor1/janec/complete-models/Checkpoint-Large_Combined_10RTT_6col_Transformer3_256_8_8_64_8_lr_1e-05-999iter.p', map_location=DEVICE)
model_list = [regression_model, classification_model]
model_name_list = [regression_model_name, classification_model_name]
is_transformer_list = [True, True]
with open('NEWDatasets/combined-dataset-preprocessed/6col-rtt-based-test.p', 'rb') as f:
    classification_dataset = pickle.load(f)
with open('/u/janechen/Documents/Transformer-training-regression/NEWDatasets/combined-dataset-preprocessed-regression/6col-rtt-based-test.p', 'rb') as f:
    regression_dataset = pickle.load(f)
prediction_len = 10
test_and_plot_distribution_multi_model(model_list, model_name_list, is_transformer_list, [regression_dataset, classification_dataset], classification_vocab_dict, prediction_len, batch_size=32, iteration=ITER)

