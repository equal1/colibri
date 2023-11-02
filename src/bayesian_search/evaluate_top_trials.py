import sys
import os
import json
import warnings
import pickle
import io
import re
from ml_tools import qflow_interface
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary as torch_summary
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from search import utils
from ml_tools.models.state_estimator import StateEstimatorWrapper
from ml_tools.models.model_utils import infer
from parsers.parse_trials import parse_trial_log_file

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))



def compute_metrics(y_true, y_pred_prob, num_classes):
    y_pred = np.argmax(y_pred_prob, axis=1)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1_score': f1_score(y_true, y_pred, average='macro'),
        'top_5_accuracy': sum([1 for i in range(len(y_true)) if y_true[i] in (-np.sort(-y_pred_prob[i]))[:5].argsort()[:5]]) / len(y_true),
        'top_1_accuracy': sum([1 for i in range(len(y_true)) if y_true[i] == np.argmax(y_pred_prob[i])]) / len(y_true),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    # Calculate AUC, ROC curve and PR curve for each class
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1], pos_label=1)
    metrics['AUC'] = auc(fpr, tpr)
    metrics['roc_curve'] = (fpr.tolist(), tpr.tolist())

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1], pos_label=1)
    metrics['pr_curve'] = (precision.tolist(), recall.tolist())

    return metrics


def load_model_and_compute_metrics(trial_folder, trial, models_path, test_queue, criterion):
    model_config = trial['Params']
    
    optimizer_config = {
        'learning_rate': model_config.pop('learning_rate'),
        'momentum': model_config.pop('momentum'),
        'weight_decay': model_config.pop('weight_decay'),
    }
    trial_path = os.path.join(models_path, trial_folder)
    model_checkpoint = [f for f in os.listdir(trial_path) if f.endswith('.ckpt')][0]
    checkpoint = torch.load(os.path.join(trial_path, model_checkpoint))
    model = StateEstimatorWrapper(model_config, optimizer_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model_size = utils.count_parameters_in_MB(model)
    y_pred_prob, y_true = infer(test_queue, model, criterion)
    metrics = compute_metrics(y_true, np.array(y_pred_prob), 5)
    
    return {
        'trial_number': int(trial_folder.split('_')[-1]),
        'val_accuracy': float(model_checkpoint.split('=')[-1].split('.ckpt')[0]),
        'model_size_MB': model_size,
        'model_config': model_config,
        'optimizer_config' :optimizer_config,
        'metrics': metrics
    }

def evaluate_top_trials(models_path, log_filepath, report_filepath):
    output_data = []
    filtered_trials = parse_trial_log_file(log_filepath)
    criterion = nn.CrossEntropyLoss().cuda()
    test_queue = qflow_interface.read_qflow_test_data(batch_size=36)
    sorted_trial_folders = sorted(os.listdir(models_path), key=lambda x: int(x.split('_')[-1]))
    sorted_filtered_trials = sorted(filtered_trials, key=lambda x: x['Trial'])

    for trial_folder, trial in zip(sorted_trial_folders, sorted_filtered_trials):
        model_data = load_model_and_compute_metrics(trial_folder, trial, models_path, test_queue, criterion)
        output_data.append(model_data)

    # Save the output data to a JSON file
    with open(report_filepath, 'w') as f:
        json.dump(output_data, f, indent=4)

