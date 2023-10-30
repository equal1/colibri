import itertools
import sys
import os
import time
import warnings
from ml_tools import qflow_interface


warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import numpy as np
import torch
import torch.nn as nn
from search import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ml_tools.models.state_estimator import StateEstimatorWrapper


optimal_model_config = {
    'in_channels': 1,
    'cnn_blocks': 6,
    'kernel_size': 5,
    'base_filters': 64,
    'fc_blocks': 2,
    'num_classes': 5,
    'activation': 'tanh', 
    'cnn_pooling': 'max_pool', 
    'global_pooling': 'adaptive_avg',
    'conv_drop_rate': 0.1496526893126031,
    'fc_drop_rate': 0.13400162558848863,
    'drop_rate_decay': 0.2400133983649985
}

optimal_optimizer_config = {
        'learning_rate': 0.006834557945902603,
        'momentum': 0.9790420970484083,
        'weight_decay': 0.0012367741418186516,
    }

def initialize_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion.cuda()


def infer(data_queue, model, criterion):
    model.eval()
    y_true = []
    y_pred = []
    total = 0.0
    total_loss = 0.0
    for step, (input, target) in enumerate(data_queue):
        n = input.size(0)
        total += n
        input = input.cuda()
        target = target.cuda()
        target_labels = torch.argmax(target, 1)
        logits = model(input)
        pred = torch.argmax(logits, 1)
        y_true.extend(target_labels.cpu().data.numpy())
        y_pred.extend(pred.cpu().data.numpy())
        loss = criterion(logits, target)
        total_loss += n * loss.cpu().data.numpy()

    correct = (np.array(y_pred) == np.array(y_true)).sum()
    acc = correct * (100.0 / len(y_pred))
    return acc, total_loss / total, y_pred, y_true


def get_saved_models(saved_model_dir):
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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, name="confusion.png"):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if float(num) > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name)


def evaluate_model(model_path, batch_size=36, classes=None):
    if classes is None:
        classes = ["0", "1", "2", "3", "4"]
    
    criterion = initialize_criterion()
    test_queue = qflow_interface.read_qflow_test_data(batch_size=batch_size)

    trained_model = torch.load(model_path)
    trained_model.cuda()

    # Perform inference and get accuracy, loss, and labels
    acc, loss, y_pred, y_true = infer(test_queue, trained_model, criterion)

    print("test acc:", acc)
    print("test loss:", loss)

    # Compute the confusion matrix
    t = confusion_matrix(y_true, y_pred)
    print("confusion matrix", t)

    # Plot the normalized confusion matrix
    plot_confusion_matrix(t, classes, name="figures/layer_5/confusion_st_dynamic.png", normalize=True)





if __name__ == "__main__":

    criterion = initialize_criterion()
    test_queue = qflow_interface.read_qflow_test_data(batch_size=36)


    saved_model_dir = "/local/rbals/playground/architecture_search/search/saved_model/"

    model_files = get_saved_models(saved_model_dir)
    accuracies = []
    losses = []
    
    for model_file in model_files:
        trained_model = torch.load(os.path.join(saved_model_dir, model_file))
        print(f"Evaluating model: {model_file}")
        acc, loss, _, _ = infer(test_queue, trained_model, criterion)
        accuracies.append(acc)
        losses.append(loss)
        
   
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    plot_accuracy_and_loss(epochs, accuracies, losses, "/local/rbals/playground/architecture_search/search/figures/state_estimator/st_1_models_evaluation.png")

    # saved_model_dir = "/local/rbals/playground/architecture_search/search/saved_model/state_estimator_dynamic.pkl"
    # evaluate_model(saved_model_dir)





# # Evaluate multiple saved models

#     saved_model_dir = "/local/rbals/playground/architecture_search/search/checkpoints/"

#     checkpoint = torch.load(os.path.join(saved_model_dir, "epoch=99-val_accuracy=0.9311.ckpt"))

#     wrapper_model = StateEstimatorWrapper(optimal_model_config, optimal_optimizer_config)

#     wrapper_model.load_state_dict(checkpoint['state_dict'])

#     model = wrapper_model.model

#     # Save the model
#     torch.save(model, os.path.join(saved_model_dir, "state_estimator_dynamic.pkl"))