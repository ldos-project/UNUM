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
    print("True vector: ", true_vectors)
    print("Predicted vector: ", predicted_vectors)
    print("Shape of true vector: ", true_vectors.shape)
    print("Shape of predicted vector: ", predicted_vectors.shape)
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
    
def test_and_plot_distribution_multi_model(model_list, model_name_list, is_transformer_list, datasets, vocab_dict_list, prediction_len, batch_size=32, iteration=""):
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
    # model_results = {model_name: {'true_classes': [], 'predicted_classes': []} for model_name in model_name_list}

    # num_batches = datasets[0].shape[0] // batch_size
    # with torch.no_grad():
    #     for i, model in enumerate(model_list):
    #         dataset = torch.from_numpy(datasets[i])
    #         for batch in range(num_batches):
    #             input_batch = dataset[batch * batch_size:(batch + 1) * batch_size, :, :].clone()
    #             enc_input = input_batch[:, :-prediction_len, :].to(DEVICE)
    #             dec_input = (1.5 * torch.ones((batch_size, prediction_len, input_batch.shape[2]))).to(DEVICE)
                
    #             src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=DEVICE)
    #             expected_output = input_batch[:, -prediction_len:, :].to(DEVICE)

    #             batch_true_classes = []
    #             # Convert expected_output to class indices (ground truth)
    #             for j in range(batch_size):
    #                 for t in range(prediction_len):
    #                     if expected_output.shape[-1] == 6:
    #                         vector = tuple(expected_output[j, t, 1:].tolist())
    #                     else:
    #                         vector = tuple(expected_output[j, t, :].tolist())
    #                     class_idx = vocab_dict_list[i].get(vector, -1)
    #                     if class_idx == -1:
    #                         print(f"Vocab dict key example: {list(vocab_dict_list[i].keys())[0]}")
    #                         raise ValueError(f"Vector not found in vocab_dict: {vector}")
    #                     batch_true_classes.append(class_idx)

    #             # Convert to tensor and move to device
    #             batch_true_classes = torch.tensor(batch_true_classes, dtype=torch.long).to(DEVICE)
    #             batch_true_classes = batch_true_classes.view(batch_size, prediction_len)

    #             # valid_mask = (expected_output[:, :, :].sum(dim=-1) != -1).to(DEVICE)

    #             # print("i:", i)
    #             # print("model_list: ", model_name_list)
    #             model_name = model_name_list[i]

    #             if is_transformer_list[i]:
    #                 # Forward pass through the transformer model to get the predicted probabilities
    #                 model_output = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
    #             else:
    #                 # Forward pass through the MLP model
    #                 enc_input_flat = enc_input.reshape(enc_input.shape[0], enc_input.shape[1] * enc_input.shape[2])
    #                 model_output = model(enc_input_flat)
    #                 model_output = model_output.view(batch_size, prediction_len, -1)  # Reshape to [batch_size, prediction_len, num_classes]

    #             # Collect true and predicted classes
    #             for t in range(prediction_len):
    #                 predicted_probs = model_output[:, t, :]  # Predicted probability for each time step
    #                 predicted_classes = torch.argmax(predicted_probs, dim=-1)  # Get the index of the max probability
                    
    #                 model_results[model_name]['true_classes'].extend(batch_true_classes[:, t].tolist())
    #                 model_results[model_name]['predicted_classes'].extend(predicted_classes.tolist())

    # with open(str(ITER)+'iter-model-tokenization-comparison-results.p', 'wb') as f_results:
    #     pickle.dump(model_results, f_results, pickle.HIGHEST_PROTOCOL)
    
    with open(str(ITER)+'iter-model-tokenization-comparison-results.p', 'rb') as f_results:
        model_results = pickle.load(f_results)
    
    # Plot the distributions for both models
    with open('NEWDatasets/combined-dataset-preprocessed/6col-VocabBackDict.p', 'rb') as f_vocab_back:
        rtt_vocab_back_dict = pickle.load(f_vocab_back)
    with open('NEWDatasets/ccbench-dataset-preprocessed/6col-VocabBackDict.p', 'rb') as f_vocab_back:
        time_vocab_back_dict = pickle.load(f_vocab_back)
    vocab_back_list = [time_vocab_back_dict, rtt_vocab_back_dict]

    plt.figure(figsize=(10, 6))
    cdf_dict = {}
    per_col_distances_dict = {}

    percentile_thresholds = [50, 75, 90, 95, 99, 99.9]
    
    # Plot PDF and CDF for each model
    for i, model_name in enumerate(model_name_list):
        print(f"Processing bucket distances for {model_name}...")
        true_classes = model_results[model_name]['true_classes']
        predicted_classes = model_results[model_name]['predicted_classes']
        
        true_vectors = np.array([vocab_back_list[i][class_idx] for class_idx in true_classes])
        predicted_vectors = np.array([vocab_back_list[i][class_idx] for class_idx in predicted_classes])

        # total_distances, feature_distances = calculate_bucket_distances(true_vectors, predicted_vectors, bucket_boundaries_ccbench)
        if true_vectors.shape[-1] == 5:
            total_distances = compute_l1_distances(true_vectors[:, [0, 2, 3]], predicted_vectors[:, [0, 2, 3]])
            per_col_distances_dict[model_name] = compute_l1_distances_per_column(true_vectors[:, [0, 2, 3]], predicted_vectors[:, [0, 2, 3]])
        else:
            total_distances = compute_l1_distances(true_vectors, predicted_vectors)
            per_col_distances_dict[model_name] = compute_l1_distances_per_column(true_vectors, predicted_vectors)

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
    plt.savefig(f'tokenization_bucket_distance_comparison_PDF.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'tokenization_bucket_distance_comparison_CDF_tail_zoom.png', dpi=300, bbox_inches='tight')
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
    # plt.savefig(f'tokenization_per_column_distances_comparison.png')
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
    rtt_vocab_dict = pickle.load(f_vocab)
with open('/u/janechen/Documents/Transformer-training/NEWDatasets/ccbench-dataset-preprocessed/6col-VocabDict.p', 'rb') as f_vocab:
    time_vocab_dict = pickle.load(f_vocab)

time_model_name = 'Time-based Transformer'
time_model = torch.load('/datastor1/janec/complete-models/Time-Checkpoint-BaseTransformer3_64_5_5_16_4_lr_1e-05_vocab-809iter.p', map_location=DEVICE)
rtt_model_name = 'RTT-based Transformer'
rtt_model = torch.load('/datastor1/janec/complete-models/Checkpoint-Combined_10RTT_6col_Transformer3_64_5_5_16_4_lr_1e-05-999iter.p', map_location=DEVICE)
model_list = [time_model, rtt_model]
model_name_list = [time_model_name, rtt_model_name]
is_transformer_list = [True, True]
with open('NEWDatasets/combined-dataset-preprocessed/6col-rtt-based-test.p', 'rb') as f:
    rtt_dataset = pickle.load(f)
with open('NEWDatasets/ccbench-dataset-preprocessed/6col-time-based-test.p', 'rb') as f:
    time_dataset = pickle.load(f)
prediction_len = 10
test_and_plot_distribution_multi_model(model_list, model_name_list, is_transformer_list, [time_dataset, rtt_dataset], [time_vocab_dict, rtt_vocab_dict], prediction_len, batch_size=32, iteration=ITER)

