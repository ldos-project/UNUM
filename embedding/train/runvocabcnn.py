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

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(DEVICE)
BATCH_SIZE = 1024
NUM_EPOCHS = 1000
CONTEXT_LENGTH = 10
PREDICTION_LENGTH = 10

class CNN(nn.Module):
    def __init__(self, input_dim, num_channels, output_dim, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(num_channels, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, seq_len]
        x = self.relu(self.conv1(x))  # First convolution
        x = self.relu(self.conv2(x))  # Second convolution
        x = x.permute(0, 2, 1)  # Change back to [batch_size, seq_len, num_channels]
        x = self.fc(x)  # Fully connected layer for each time step
        x = self.softmax(x)  # Softmax over output dimension
        return x  # Shape: [batch_size, seq_len, output_dim]

def train_cnn(model, dataset, optimizer, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None, vocab_dict=None, checkpoint_path=None, resume_from_epoch=0):
    loss_func = nn.CrossEntropyLoss()
    loss_traj = []
    model.train()
    num_batch = dataset.shape[0] // batch_size

    if checkpoint_path and resume_from_epoch > 0:
        checkpoint_file = os.path.join(checkpoint_path, "CNN-"+checkpoint_suffix+"-epoch"+str(resume_from_epoch)+".pth")
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
            model_out = model(enc_input)  # Shape: [batch_size, 10, num_classes]
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

            # Loss calculation
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

        print(f"[CNN] Epoch {epoch} | Time: {epoch_time:.2f}s | Loss: {epoch_loss:.6f}")

        # Save model checkpoint every 100 epochs
        if checkpoint_suffix is not None and (epoch + 1) % 100 == 0:
            torch.save(model, f'CNN-{checkpoint_suffix}-epoch{epoch + 1}.pth')

        # Shuffle dataset
        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]

    return model, loss_traj

parser = argparse.ArgumentParser(description="Train a CNN model with configurable parameters.")
parser.add_argument('--num_channels', type=int, default=256, 
                    help='Number of output channels for convolution layers. Defaults to 128.')
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

model = CNN(input_dim=train_dataset.shape[-1], num_channels=args.num_channels, output_dim=num_classes).to(DEVICE)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
train_dataset = train_dataset.to(DEVICE)
save_name = 'CNN-{}channels-noweighting-vocab'.format(args.num_channels)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
trained_model, loss_traj = train_cnn(model, train_dataset, opt, PREDICTION_LENGTH, DEVICE, 1000, BATCH_SIZE, checkpoint_suffix=str(args.num_channels), vocab_dict=vocab_dict, checkpoint_path=args.checkpoint_path, resume_from_epoch=args.resume_from_epoch)

torch.save(trained_model, '/datastor1/janec/Models/{}-1000iter.p'.format(save_name))
