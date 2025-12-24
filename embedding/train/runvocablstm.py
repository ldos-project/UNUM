import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import PathVariables
import torch
import torch.nn as nn
import pickle
import time
import argparse
from embedding.utils.utils import test_model_batched, test_model, weighted_mse

#CONSTANTS
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(DEVICE)
PAD_IDX = 2
BATCH_SIZE = 1024
NUM_EPOCHS = 1000
CONTEXT_LENGTH = 10
PREDICTION_LENGTH = 10

def train_lstm(model, dataset, optimizer, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None, vocab_dict=None, checkpoint_path=None, resume_from_epoch=0):

    loss_func = nn.CrossEntropyLoss()
    loss_traj = []
    model.train()
    num_batch = dataset.shape[0] // batch_size

    if checkpoint_path and resume_from_epoch > 0:
        checkpoint_file = os.path.join(checkpoint_path, "LSTM-"+checkpoint_suffix+"-epoch"+str(resume_from_epoch)+".pth")
        if os.path.exists(checkpoint_file):
            print(f"[info] Loading checkpoint from: {checkpoint_file}")
            model = torch.load(checkpoint_file, map_location=device)
            print(f"[info] Resumed training from epoch {resume_from_epoch}")
        else:
            print(f"[warning] Checkpoint file not found: {checkpoint_file}. Starting from scratch.")

    # Training loop
    for epoch in range(resume_from_epoch, num_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch in range(num_batch):
            # Prepare batch data
            input = dataset[batch * batch_size:(batch + 1) * batch_size, :, :].clone()
            enc_input = input[:, :-prediction_len, :].to(device).float()  # Shape: [batch_size, 10, 6]
            expected_output = input[:, -prediction_len:, :].to(device)  # Shape: [batch_size, 10, 6]  

            # Forward pass
            model_out = model(enc_input)  # Shape: [batch_size, 10, 6]
            optimizer.zero_grad()

            # Process categorical outputs
            batch_classes = []
            for i in range(batch_size):
                for t in range(prediction_len):
                    discrete_features = tuple(expected_output[i, t, 1:].tolist())
                    class_idx = vocab_dict.get(discrete_features, -1)
                    if class_idx == -1:
                        raise ValueError(f"Vector not found in vocab_dict: {discrete_features}")
                    batch_classes.append(class_idx)

            # Convert batch classes to tensor
            batch_classes = torch.tensor(batch_classes, dtype=torch.long).to(device)
            batch_classes = batch_classes.view(batch_size, prediction_len)  # Reshape for batch

            # Ensure all class indices are valid
            if (batch_classes < 0).any() or (batch_classes >= num_classes).any():
                raise ValueError(f"Invalid class indices found: {batch_classes}")
            
            # Ensure there are no negative values in batch_classes
            assert (batch_classes >= 0).all(), "Negative class indices found in batch_classes."

            # Ensure that the indices do not exceed the number of classes
            assert (batch_classes < num_classes).all(), "Class indices exceed number of classes."

            model_out = model_out.view(batch_size, prediction_len, -1)  # Reshape to [batch_size, prediction_len, num_classes]

            loss = 0
            for i in range(prediction_len):
                logits = model_out[:, i, :]  # Model output for time step i
                loss += loss_func(logits, batch_classes[:, i])  # Cross-entropy loss for step i

            # Backpropagation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log epoch stats
        epoch_time = time.time() - t0
        epoch_loss /= num_batch
        loss_traj.append(epoch_loss)

        print(f"[LSTM] Epoch {epoch} | Time: {epoch_time:.2f}s | Loss: {epoch_loss:.6f}")

        # Save model checkpoint every 100 epochs
        if checkpoint_suffix is not None and (epoch + 1) % 100 == 0:
            torch.save(model, f'LSTM-{checkpoint_suffix}-epoch{epoch + 1}.pth')

        # Shuffle dataset
        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]

    return model, loss_traj



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM output
        out = self.fc(out)  # Apply FC layer to all time steps
        out = self.relu(out)  # Optional: ReLU activation
        out = self.softmax(out)  # Final softmax for classification
        return out  # Shape: [batch_size, seq_len, output_dim]

parser = argparse.ArgumentParser(description="Train an LSTM model with a configurable hidden dimension.")
parser.add_argument('--hidden_dim', type=int, default=128, 
                    help='Hidden dimension size for the LSTM. Defaults to 102.')
parser.add_argument('--checkpoint_path', type=str, help="Path to save/load model checkpoints")
parser.add_argument('--resume_from_epoch', type=int, default=0, help="Epoch to resume training from (if applicable)")
args = parser.parse_args()

with open('NEWDatasets/combined-dataset-preprocessed/6col-VocabDict.p', 'rb') as f_vocab:
        vocab_dict = pickle.load(f_vocab)
        num_classes = len(vocab_dict)
        print("vocab dict size: ", num_classes)

with open('NEWDatasets/combined-dataset-preprocessed/6col-rtt-based-train.p', 'rb') as f:
    train_dataset_np = pickle.load(f)
    train_dataset = torch.from_numpy(train_dataset_np)
model = LSTM(input_dim=train_dataset.shape[-1], output_dim=num_classes, hidden_dim=args.hidden_dim).to(DEVICE)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
train_dataset = train_dataset.to(DEVICE)
save_name = 'LSTM-{}dim-noweighting-vocab'.format(args.hidden_dim)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
trained_model, loss_traj = train_lstm(model, train_dataset, opt, PREDICTION_LENGTH, DEVICE, 1000, BATCH_SIZE, checkpoint_suffix=str(args.hidden_dim), vocab_dict=vocab_dict, checkpoint_path=args.checkpoint_path, resume_from_epoch=args.resume_from_epoch)

torch.save(trained_model, '/datastor1/janec/Models/{}-1000iter.p'.format(save_name))