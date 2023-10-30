# Import necessary libraries and modules
import argparse
import sys

from ml_tools import qflow_interface

print("start ")
import os
import time
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the parent directory to the system path to allow importing modules from there
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

# Import more libraries and modules
import numpy as np
import torch
import torch.nn as nn
import search.darts_search.genotypes as genotypes
from search import utils
import copy

from ml_tools.models.state_estimator import StateEstimatorWrapper

optimal_model_config = {
            "cnn_blocks": 5,
            "kernel_size": 5,
            "base_filters": 64,
            "fc_blocks": 2,
            "activation": "tanh",
            "cnn_pooling": "max_pool",
            "global_pooling": "adaptive_avg",
            "conv_drop_rate": 0.17862690416565716,
            "fc_drop_rate": 0.12421171385095253,
            "drop_rate_decay": 0.2342628855529526
        }

optimal_optimizer_config = {
            "learning_rate": 0.0099310492507264,
            "momentum": 0.9632906408162738,
            "weight_decay": 0.0037540429319538426
        },

# Define batch sizes for training and testing
train_batch_size = 512
test_batch_size = 512

# Load data for training and validation
train_queue, valid_queue = qflow_interface.read_qflow_data(batch_size=train_batch_size, is_prepared=True, fast_search=False)

# Load test data
test_queue = qflow_interface.read_qflow_test_data(batch_size=test_batch_size)

# Load model
model_checkpoint = "/local/rbals/playground/architecture_search/search/top_models/trial_208/epoch=99-val_accuracy=0.9203.ckpt"
checkpoint = torch.load(model_checkpoint)

# Evaluate model on test data and compute metrics
model = StateEstimatorWrapper(optimal_model_config, optimal_optimizer_config)
model.load_state_dict(checkpoint['state_dict'])

model.cuda()

# Deep copy the state dictionary of the trained model
trained_dict = copy.deepcopy(model.state_dict())

# Define the loss function (criterion) as CrossEntropyLoss and move it to GPU
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

# Define the inference function
def infer(data_queue, model, criterion):
    # Set the model to evaluation mode
    model.eval()
    y_true = []
    y_pred = []
    total = 0.0
    total_loss = 0.0
    # Loop through the data
    for step, (input, target) in enumerate(data_queue):
        n = input.size(0)
        total += n
        # Move input and target to GPU
        input = input.cuda()
        target = target.cuda()
        # Get the true labels
        target_labels = torch.argmax(target, 1).cuda()
        # Get the model's predictions
        logits = model(input)
        pred = torch.argmax(logits, 1)
        # Append true and predicted labels to lists
        for val in target_labels.cpu().data.numpy():
            y_true.append(val)
        for val in pred.cpu().data.numpy():
            y_pred.append(val)
        # Calculate the loss
        loss = criterion(logits, target_labels)
        loss_val = loss.cpu().data.numpy()
        total_loss += n * loss_val
    # Calculate accuracy
    correct = (np.array(y_pred) == np.array(y_true)).sum()
    acc = correct * (100.0 / len(y_pred))
    return acc, total_loss / total

# Create a mapping from index to key for the trained model's state dictionary
idx = 0
index_to_key_map = {}
for k, _ in trained_dict.items():
    index_to_key_map[idx] = k
    idx += 1
print("all edges count: ", len(index_to_key_map))

# Initialize pruning parameters
cut_count = 0
max_to_cut = len(index_to_key_map)
logging.basicConfig(filename="logs/pruned_process_trail_model.log", filemode="w", level=logging.INFO)
print("pruned process 5 layer")
acc = 0
to_del_order =  []
cutted_set = set(to_del_order)
print("start time: ", time.asctime(time.localtime(time.time())))

# Start the pruning process
while cut_count < max_to_cut:
    result = []
    cur_dict = copy.deepcopy(trained_dict)
    # Zero out the weights for the edges to be pruned
    for todel in to_del_order:
        cur_dict[index_to_key_map[todel]] = torch.zeros_like(cur_dict[index_to_key_map[todel]])
    # Evaluate the model's performance after pruning each edge
    for target in range(len(trained_dict)):
        if "weight" not in index_to_key_map[target] and "bias" not in index_to_key_map[target]:
            continue
        inner_dict = copy.deepcopy(cur_dict)
        if target in cutted_set:
            continue
        inner_dict[index_to_key_map[target]] = torch.zeros_like(inner_dict[index_to_key_map[target]])
        model.load_state_dict(inner_dict)
        acc, test_loss = infer(valid_queue, model, criterion)
        result.append([target, acc])
        if acc==100.00:
            print("early stop")
            break
        logging.info('cut edge idx:%d loss:%f acc:%f ' % (target, test_loss, acc))
        print("cut edge idx=", target, "acc=", acc, "loss=", test_loss)

    # Sort the results based on accuracy and prune the edge that results in the highest accuracy
    result.sort(key=lambda x: -x[1])
    j = 0  
    cutted_set.add(result[j][0])
    to_del_order.append(result[j][0])
    test_dict = copy.deepcopy(cur_dict)
    test_dict[index_to_key_map[result[j][0]]] = torch.zeros_like(test_dict[index_to_key_map[result[j][0]]])
    model.load_state_dict(test_dict)
    torch.save(model, ('./pruned_model/weights_cut_trial    _%d.pkl' % cut_count))
    test_acc, test_loss = infer(test_queue, model, criterion)
    torch.cuda.empty_cache()
    cut_count += 1
    logging.info("OUTSIDE cut_count:%d to_del_order:%s after_del_acc:%f test_acc:%f test_loss:%f" % (
    cut_count, str(to_del_order), result[j][1], test_acc, test_loss))
    print("cut_count:", cut_count, "to_del_order=", to_del_order, "after del acc", result[j][1])
    print("test acc: ", test_acc, "  test loss:   ", test_loss)
    print(time.asctime(time.localtime(time.time())))

print("to_del_order", to_del_order)
