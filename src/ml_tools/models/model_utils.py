import os
import shutil
import sys

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary as torch_summary
import torchvision.transforms as transforms


def prepare_dataloader(features, labels, type, batch_size=64, shuffle=True, seed=None, num_workers=24):
    """
    Prepare DataLoader for training and validation.

    :param features: NumPy array or PyTorch tensor of features
    :param labels: NumPy array or PyTorch tensor of labels
    :param batch_size: Batch size
    :param shuffle: Whether to shuffle the data
    :param seed: Seed for reproducibility
    :return: train_loader, val_loader
    """

    if type == 'val':
        shuffle = False
    
    if not torch.is_tensor(features):
        features = torch.tensor(features, dtype=torch.float32)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    dataset = TensorDataset(features, labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    
    return loader

# Helper Functions
def accuracy(output, target):
    """
    Compute the accuracy of the predicted outputs with respect to the targets.
    
    Args:
    - output (torch.Tensor): The predicted outputs from the model.
    - target (torch.Tensor): The true labels for the data.
    
    Returns:
    - float: The accuracy percentage.
    """
    batch_size = target.size(0)
    target_labels = torch.argmax(target, 1)
    pred = torch.argmax(output, 1)
    correct = (pred==target_labels).sum().float()
    res = correct.mul_(100.0 / batch_size)
    return res


def count_parameters_in_MB(model):
    """
    Count the number of parameters in the model (excluding auxiliary parameters) in megabytes (MB).
    
    Args:
    - model (torch.nn.Module): The model whose parameters are to be counted.
    
    Returns:
    - float: The number of parameters in MB.
    """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save(model, model_path):
    """
    Save the model state to the specified path.
    
    Args:
    - model (torch.nn.Module): The model to be saved.
    - model_path (str): The path where the model state will be saved.
    """
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    """
    Load the model state from the specified path.
    
    Args:
    - model (torch.nn.Module): The model whose state is to be loaded.
    - model_path (str): The path from where the model state will be loaded.
    """
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    """
    Drop path regularization for convolutional architectures. 
    Drops out a random set of activations in a tensor by setting them to zero.
    
    Args:
    - x (torch.Tensor): The input tensor.
    - drop_prob (float): The probability of an element to be zeroed.
    
    Returns:
    - torch.Tensor: The tensor after applying drop path.
    """
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def write_loss(filename,  loss):
    """
    Append loss values to a file, separated by commas.
    
    Args:
    - filename (str): The name of the file to write the losses to.
    - loss (list): The list of loss values.
    """
    with open(filename, "a") as f:
        for l in loss:
            f.write(str(l))
            f.write(",")
        f.write("\n")


def write_acc(filename,  loss):
    """
    Append accuracy values to a file, separated by commas.
    
    Args:
    - filename (str): The name of the file to write the accuracies to.
    - loss (list): The list of accuracy values.
    """
    with open(filename, "a") as f:
        for l in loss:
            f.write(str(l))
            f.write(",")
        f.write("\n")


def save_checkpoint(state, is_best, save):
    """
    Save the checkpoint of the training. If the checkpoint is the best one so far, 
    also saves it under a different filename.
    
    Args:
    - state (dict): Dictionary containing the current state of training.
    - is_best (bool): True if the current checkpoint is the best one so far.
    - save (str): The directory where the checkpoint will be saved.
    """
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def model_summary(model, input_size, device="cuda"):
    """
    Print a summary of the given model.

    Args:
    - model (torch.nn.Module): The model whose summary is to be printed.
    - input_size (tuple): The size of the input tensor.
    - device (str): The device type, "cuda" or "cpu".

    Returns:
    - str: A formatted string containing the summary of the model.
    """
    try:
        summary_str = torch_summary(model, input_size, device=device)
        return summary_str
    except Exception as e:
        return str(e)