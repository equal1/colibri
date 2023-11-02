import json
import os
import sys
import logging
from ml_tools import qflow_interface
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import darts_search.genotypes as genotypes
from darts_search.model import Network
from darts_search.architect import Architect
from ml_tools.models.model_utils import model_utils

from ml_tools.models.state_estimator import StateEstimator

warnings.filterwarnings("ignore")

# Load configuration from 'config.json'
def get_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

config = get_config()
logging.basicConfig(filename="logs/train_%d.log" % config['layers'], filemode="w", level=logging.INFO)

OUT_CLASSES = 5

lambda_train_regularizer = config['lambda_train_regularizer']
lambda_valid_regularizer = config['lambda_valid_regularizer']

result_gentotype = []

# Main function for training and architecture search
def dart_search(is_search=True):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    # Set random seeds for reproducibility
    np.random.seed(config['seed'])
    cudnn.benchmark = True
    torch.manual_seed(config['seed'])
    cudnn.enabled = True
    torch.cuda.manual_seed(config['seed'])

    # Initialize the loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Initialize the model based on the search mode
    if is_search:
        model = Network(config['init_channels'], OUT_CLASSES, config['layers'], criterion)
    else:
        genotype = eval("genotypes.%s" % config['arch'])
        model = Network(config['init_channels'], OUT_CLASSES, config['layers'], auxiliary=config['auxiliary'], genotype=genotype)

    model.cuda()
    
    #Separate architecture parameters from weight parameters
    if is_search:
        arch_parameters = model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        parameters = model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params, parameters)
    else:
        weight_params = model.parameters()

    # Initialize the optimizer for model weights
    optimizer = torch.optim.SGD(
        weight_params,
        config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Load data for training and validation
    train_queue, valid_queue = qflow_interface.read_qflow_data(batch_size=config['batch_size'], label_key_name='state', is_prepared=True, fast_search=False)

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(config['epochs']), eta_min=config['learning_rate_min'])

    # Initialize the architect for architecture search
    if is_search:
        architect = Architect(model, criterion, config)
        
    # Main training loop
    for epoch in range(config['epochs']):
        if is_search:
            lr = scheduler.get_lr()[0]
            train(epoch, train_queue, valid_queue, model, criterion, optimizer, architect, lr)
        else:
            # model.drop_path_prob = config['drop_path_prob'] * epoch / config['epochs']
            train(epoch, train_queue, valid_queue, model, criterion, optimizer)

        # Validate the model
        with torch.no_grad():
            infer(epoch, valid_queue, model, criterion)

        scheduler.step()

        # Save the model periodically
        if is_search or epoch == config['epochs'] - 1:
            torch.save(model, "./saved_model/layer_%d_weights_%d.pkl" % (config['layers'], epoch))
        else:
            if epoch % 5 == 0:
                torch.save(model, "./saved_model/layer_%d_weights_%d.pkl" % (config['layers'], epoch))

        # Save the discovered architectures during search
        if is_search:
            genotype, normal_cnn_count, reduce_cnn_count = model.get_genotype()
            result_gentotype.append(genotype)
            print("epoch:", epoch, "gentotype", genotype)


    # Save the final list of discovered architectures
    if is_search:
        with open('logs/types%d.txt' % config['layers'], 'w') as f:
            for t in result_gentotype:
                f.write(str(t))
                f.write("\n")

# Training function
def train(epoch, train_queue, valid_queue, model, criterion, optimizer, architect=None, lr=None):
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # Update architecture parameters using the architect
        if architect:
            if config['unrolled'] == 1:
                order_option = True
            elif config['unrolled'] == 2:
                order_option = False

            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda()

            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=order_option)

        # Forward and backward passes for model weights
        optimizer.zero_grad()
        logits, _ = model(input)
        prec = model_utils.accuracy(logits, target)
        loss = criterion(logits, target)
        loss_val = loss.cpu().data.numpy()
        loss.backward()

        if architect:
            parameters = model.arch_parameters()
        else:
            parameters = model.parameters()

        nn.utils.clip_grad_norm_(parameters, config['grad_clip'])
        optimizer.step()

        logging.info('train_epoch:%d step:%d loss:%f acc:%f' % (epoch, step, loss_val, prec))
        print('train_epoch:%d step:%d loss:%f acc:%f' % (epoch, step, loss_val, prec))

# Validation function
def infer(epoch, valid_queue, model, criterion):
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()
        logits, _ = model(input)
        prec = model_utils.accuracy(logits, target)
        loss = criterion(logits, target)
        loss_val = loss.cpu().data.numpy()

        logging.info('valid_epoch:%d step:%d loss:%f acc:%f' % (epoch, step, loss_val, prec))
        print('valid_epoch:%d step:%d loss:%f acc:%f' % (epoch, step, loss_val, prec))










