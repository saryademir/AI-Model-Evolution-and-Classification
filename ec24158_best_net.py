import os
import torch
import librosa
import numpy as np
import scipy.io
from scipy.signal.windows import hann
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split  

BATCH_SIZE = 8 #for testing

# You only need to define your best model here. The rest of the code will work as is
class Net(nn.Module):
    def __init__(self, D=3, K=4):
        super().__init__()
        layers = []
        self.network = nn.Sequential(
            nn.Linear(D, 32),    
            nn.ReLU(),         
            nn.Linear(32, K)    
        )


    # you shouldn't have to modify this method
    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
        - x (Tensor): The input data tensor.

        Returns:
        - Tensor: The output of the network after processing the input tensor through all the layers defined
          in the `network` attribute.
        """
        return self.network(x)


def load_and_process_val_pcvc_data(directory='.', train_size=0.8, random_seed=42):
    
    # List all .mat files in the specified directory
    all_mats = [file for file in os.listdir(directory) if file.endswith('.mat')]
    raw_data = []
    num_vowels = 6
    ndatapoints_per_vowel = 299
    labels = []

    for idx, mat_file in enumerate(all_mats):
        mat_path = os.path.join(directory, mat_file)
        mat_data = np.squeeze(scipy.io.loadmat(mat_path)['x'])
        raw_data.append(mat_data)
        labels.append(np.repeat(np.arange(num_vowels)[np.newaxis], mat_data.shape[0], axis=0))

    # Concatenate and reshape all data
    raw_data, labels = np.concatenate(raw_data, axis=1), np.concatenate(labels, axis=1)
    nreps, nvow, nsamps = raw_data.shape
    raw_data = np.reshape(raw_data, (nreps * nvow, nsamps), order='F')
    labels = np.reshape(labels, (nreps * nvow), order = 'F')

    # Split data into training and validation sets
    tr_data, vl_data, tr_labels, vl_labels = train_test_split(
        raw_data, labels, train_size=train_size, random_state=random_seed, stratify=labels)
    
    # Define window size and function
    window_size = 10000
    window = hann(window_size)
    
    # Process Validation Data with fixed slicing
    vl_data = vl_data[:, 5000:15000] * window
    vl_data = np.array([librosa.resample(d, orig_sr=48000, target_sr=16000) for d in vl_data])

    # One-hot encode labels    
    vl_labels = np.eye(num_vowels)[vl_labels]

    return vl_data, vl_labels.astype('float')

# Load and process the PCVC dataset
X_val, labels_val = load_and_process_val_pcvc_data()
Nsamps, Nclasses = X_val.shape[-1], labels_val.shape[-1]
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(labels_val)
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# Load the model
best_model = Net(D=Nsamps, K=Nclasses)  # Ensure the architecture matches
best_model.load_state_dict(torch.load('best_net.pth', weights_only=True))
print("Loaded the best model from 'best_net.pth'")

# Evaluate the model on the validation set
best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in val_loader:
        output = best_model(data)
        pred = output.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

validation_accuracy = correct / total
print(f'Evaluation on validation set complete. Accuracy: {validation_accuracy:.4f} ({correct}/{total} correct)')