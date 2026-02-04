
# UNUM

Unum is a new framework powered by a unified network state embedder leveraging Transformers' self-attention mechanism and diverse training datasets to learn rich, latent state representations. A core design goal of Unum is to decouple state estimation into a standalone, first-class entity in network control, and improve the state estimation quality. 

This repository provides code for Unum embedding training (`embedding/`) and sample integrations with two representative downstream controllers: congestion control (`controller-examples/cc/`) and adaptive bitrate selection (`controller-examples/abr/`). Additional details and evaluation results can be found in our NSDI'26 paper `UNUM: A New Framework for Network Control`.

## Artifact Evaluation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/ldos-project/UNUM.git
   cd UNUM
   
   # Or if already cloned, update submodules:
   git submodule update --init --recursive
   ```

2. **Choose your use case:**

   **Option A: Congestion Control Integration**
   - See `controller-examples/cc/README.md` for detailed setup
   - Requires 16-node CloudLab cluster
   
   **Option B: ABR Integration**
   - See `controller-examples/abr/README.md` for detailed setup  
   - Supports single-server quick evaluation mode

## Unum Network State Embedding


### Step 1: Collect Network Traces

Collect network traces under a variety of network environments and conditions.  
The [KernMLOps](https://github.com/ldos-project/KernMLOps) project provides an easy-to-use toolchain for collecting kernel-level network telemetry on Linux.

---

### Step 2: Preprocess the Dataset (`embedding/preprocess/`)

Preprocessing consists of bucketization followed by tokenization.

**Bucketization**

```sh
scripts/run_all_bucketization.sh
```
This script generates bucket boundary files using Quantile, KMeans and Histogram.

**Tokenization**

```sh
scripts/run_all_tokenization.sh
```
This step converts raw datasets into tokenized datasets according to the generated bucket boundaries.

### Step 3: Train Network State Prediction Models (`embedding/train/`)

**Unum Embedder**
```
python runvocab.py -GPU {} -DFF {} -NEL {} -NDL {} -NH {} -ES {} -W {} -LR {} -M {} -BF {} -TT {}
```
Arguments:
- `-GPU, --GPUNumber` — GPU index to use (0-based). Uses CPU if CUDA is unavailable.
- `-DFF, --DimFeedForward` — Transformer feed-forward layer dimension (e.g., 256).
- `-NEL, --NumEncoderLayers` — Number of encoder layers.
- `-NDL, --NumDecoderLayers` — Number of decoder layers.
- `-NH, --NHead` — Number of attention heads.
- `-ES, --EmbSize` — Model embedding size (d_model).
- `-W, --Weighted` — Whether to use weighted loss (true/false).
- `-LR, --LearningRate` — Learning rate (e.g., 1e-4).
- `-M, --ModelName` — A name prefix for checkpoints and logs.
- `-BF, --boundaries-file` — Path to bucket boundary file used for tokenization (e.g., `.../boundaries-quantile100.pkl`). Determines which tokenized dataset is loaded.
- `-TT, --token-type` — Tokenization mode: `single` (combo token classification) or `multi` (multi-head tokens per feature).

Optional tuning arguments:
- `-AB1, --AdamBeta1`, `-AB2, --AdamBeta2` — Adam optimizer betas.
- `-SI, --SelectedIndices` — Comma-separated feature indices for feature selection.
- `-LW, --LossWeight` — Comma-separated loss weights per target.
- `-A, --Alpha` — Alpha for reweighted loss.
- `-ME, --MoreEmbedding` — Toggle additional embedding layers.

Example command:
```
python runvocab.py -GPU 2 -DFF 256 -NEL 4 -NDL 4 -NH 2 -ES 16 -W False -LR 1e-4 -M Combined_10RTT_6col -BF /datastor1/janec/datasets/boundaries/boundaries-quantile50.pkl -TT multi
```

**MLP**
```
python runvocabmlp.py --hidden_dim {}
```
Arguments:
- `--hidden_dim` — Hidden layer width for the MLP classifier (defaults to 102).
- `--checkpoint_path` — Optional path to save/load checkpoints.
- `--resume_from_epoch` — Optional epoch number to resume training from.

**CNN**
```
python runvocabcnn.py --num_channels {}
```
Arguments:
- `--num_channels` — Number of output channels for convolution layers (defaults to 256).
- `--checkpoint_path` — Optional path to save/load checkpoints.
- `--resume_from_epoch` — Optional epoch number to resume training from.

**LSTM**
```
python runvocablstm.py --hidden_dim {}
```
Arguments:
- `--hidden_dim` — Hidden dimension size for the LSTM (defaults to 128).
- `--checkpoint_path` — Optional path to save/load checkpoints.
- `--resume_from_epoch` — Optional epoch number to resume training from.

### Step 4: Testing (`embedding/test/`)
Evaluate trained models using `python test_transformer.py`, which reports bucket index prediction accuracy as reported in the paper.
