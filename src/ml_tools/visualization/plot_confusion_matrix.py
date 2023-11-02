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

def plot_confusion_matrix(model_path, test_queue, classes=None, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, name="confusion.png"):
    if classes is None:
        classes = ["0", "1", "2", "3", "4"]
    
    criterion = nn.CrossEntropyLoss().cuda()

    trained_model = torch.load(model_path)
    trained_model.cuda()

    # Perform inference and get accuracy, loss, and labels
    acc, loss, y_pred, y_true = infer(trained_model, criterion, test_queue)
    print("test acc:", acc)
    print("test loss:", loss)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix", cm)

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
    plt.show()


