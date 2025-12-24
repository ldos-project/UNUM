import numpy as np
import pickle
import matplotlib.pyplot as plt

# 1. Load your data
genet_data1 = np.load('NEWDatasets/genet-dataset-raw/tcp_metrics.npy')
genet_data2 = np.load('NEWDatasets/genet-dataset-raw/tcp_metrics_trace_file_2.npy')

# with open('NEWDatasets/ccbench-dataset-raw/6col-rtt-based.p', "rb") as f:
#     ccbench_data1 = pickle.load(f)

print("genet_data1 shape:", genet_data1.shape)
print("genet_data2 shape:", genet_data2.shape)
# print("ccbench_data1 shape:", ccbench_data1.shape)

# 2. Flatten ccbench_data1 from (520601, 20, 6) to (520601 * 20, 6)
# ccbench_data1_flat = ccbench_data1.reshape(-1, 6)
# print("ccbench_data1_flat shape:", ccbench_data1_flat.shape)

# 3. Combine all datasets vertically (i.e., row-wise)
combined_data = np.vstack([genet_data1, genet_data2])#, ccbench_data1_flat])
print("combined_data shape:", combined_data.shape)

# 4. Plot distributions for the 6 features in the combined dataset
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
# x_lim_list = [None, None, None, None, None, None]
x_lim_list = [None, (0, 50), (0, 5000), None, None, None]

for i in range(6):
    ax = axes[i]
    feature_data = combined_data[:, i]

    # 5. Calculate and print the 10th, ... 90th percentiles
    p0 = np.percentile(feature_data, 0)
    p10 = np.percentile(feature_data, 10)
    p20 = np.percentile(feature_data, 20)
    p30 = np.percentile(feature_data, 30)
    p40 = np.percentile(feature_data, 40)
    p50 = np.percentile(feature_data, 50)
    p60 = np.percentile(feature_data, 60)
    p70 = np.percentile(feature_data, 70)
    p80 = np.percentile(feature_data, 80)
    p90 = np.percentile(feature_data, 90)
    p100 = np.percentile(feature_data, 100)
    print(f"Feature {i+1}: 0th pct: {p0:.2f}, 10th pct: {p10:.2f}, 20th pct: {p20:.2f}, 30th pct: {p30:.2f}, 40th pct: {p40:.2f}, 50th pct: {p50:.2f}, 60th pct: {p60:.2f}, 70th pct: {p70:.2f}, 80th pct: {p80:.2f}, 90th pct: {p90:.2f}, 100th pct: {p100:.2f}")

    # Sort the data for CDF
    sorted_data = np.sort(feature_data)
    # Compute the CDF values
    cdf = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)  # from 0 to 1

    # Plot the CDF
    ax.plot(sorted_data, cdf, label='CDF', color='blue')

    # Draw vertical lines for p10 and p90
    ax.axvline(p10, color='red', linestyle='--', label='10th pct')
    ax.axvline(p90, color='green', linestyle='--', label='90th pct')

    ax.set_title(f'Feature {i+1}')
    ax.set_xlabel('Value')
    ax.set_ylabel('CDF')
    if x_lim_list[i]:
        ax.set_xlim(x_lim_list[i])
    ax.legend()

fig.suptitle('CDF of Features (Combined Dataset)', fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig('combined_dataset_distributions.png')  # Save the plot as an image file
