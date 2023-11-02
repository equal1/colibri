import itertools
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ml_tools import qflow_interface
from ml_tools.models.state_estimator import StateEstimatorWrapper
from ml_tools.models.model_utils import infer


def get_saved_models(saved_model_dir, model_filename , extension='.pkl'):
    model_files = [f for f in os.listdir(saved_model_dir) if f.startswith('state_estimator_1_') and f.endswith('.pkl')]
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return model_files

def plot_accuracy_and_loss(epochs, accuracies, losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.title('Test Accuracy and Loss vs. Epoch')
    ax1 = plt.gca()
    ax1.plot(epochs, accuracies, '-o', color='b', label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(epochs, losses, '-o', color='r', label='Test Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()



def plot_model_test_performance_per_epoch(saved_model_dir, model_filename, save_plot_path, test_queue, extension='.pkl'):
    criterion = nn.CrossEntropyLoss().cuda()
    model_files = get_saved_models(saved_model_dir, model_filename, extension)
    
    accuracies = []
    losses = []
    for model_file in model_files:
        trained_model = torch.load(os.path.join(saved_model_dir, model_file))
        acc, loss, _, _ = infer(trained_model, criterion, test_queue)
        accuracies.append(acc)
        losses.append(loss)
        
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    plot_accuracy_and_loss(epochs, accuracies, losses, save_plot_path)

