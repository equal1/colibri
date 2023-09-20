import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def prepare_dataloader(features, labels, batch_size=64, shuffle=True, seed=None):
    """
    Prepare DataLoader for training and validation.

    :param features: NumPy array or PyTorch tensor of features
    :param labels: NumPy array or PyTorch tensor of labels
    :param batch_size: Batch size
    :param shuffle: Whether to shuffle the data
    :param seed: Seed for reproducibility
    :return: train_loader, val_loader
    """
    
    if not torch.is_tensor(features):
        features = torch.tensor(features, dtype=torch.float32)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    dataset = TensorDataset(features, labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


