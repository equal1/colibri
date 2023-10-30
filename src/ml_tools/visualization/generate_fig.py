# Required libraries are imported.
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Modifying the Python path to include the parent directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

def load_data(filename, is_pruned_process=False):
    # Initializing dictionaries to store cumulative accuracy and loss values for both training and validation.
    train_acc_cumulative = {}
    valid_acc_cumulative = {}
    train_loss_cumulative = {}
    valid_loss_cumulative = {}
    
    train_steps_per_epoch = {}
    valid_steps_per_epoch = {}

    # Opening the given filename in read mode.
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            # If it's a pruned process and "OUTSIDE" is not in the line, skip this line.
            if is_pruned_process and "OUTSIDE" not in line:
                continue
            
            # Parsing the epoch number, values of accuracy and loss from the line.
            epoch_num = int(line.split("train_epoch:")[1].split(" ")[0]) if "train" in line else int(line.split("valid_epoch:")[1].split(" ")[0])
            data = line.split(" ")
            loss = float(data[-2].split(":")[-1])
            acc = float(data[-1].split(":")[-1])

            # Storing parsed values into corresponding dictionaries.
            if "valid" in line:
                valid_acc_cumulative[epoch_num] = valid_acc_cumulative.get(epoch_num, 0) + acc
                valid_loss_cumulative[epoch_num] = valid_loss_cumulative.get(epoch_num, 0) + loss
                valid_steps_per_epoch[epoch_num] = valid_steps_per_epoch.get(epoch_num, 0) + 1
            else:
                train_acc_cumulative[epoch_num] = train_acc_cumulative.get(epoch_num, 0) + acc
                train_loss_cumulative[epoch_num] = train_loss_cumulative.get(epoch_num, 0) + loss
                train_steps_per_epoch[epoch_num] = train_steps_per_epoch.get(epoch_num, 0) + 1
    
    # Calculating the average accuracy and loss for each epoch.
    train_acc = [train_acc_cumulative[epoch] / train_steps_per_epoch[epoch] for epoch in sorted(train_acc_cumulative.keys())]
    valid_acc = [valid_acc_cumulative[epoch] / valid_steps_per_epoch[epoch] for epoch in sorted(valid_acc_cumulative.keys())]
    train_loss = [train_loss_cumulative[epoch] / train_steps_per_epoch[epoch] for epoch in sorted(train_loss_cumulative.keys())]
    valid_loss = [valid_loss_cumulative[epoch] / valid_steps_per_epoch[epoch] for epoch in sorted(valid_loss_cumulative.keys())]

    return train_acc, valid_acc, train_loss, valid_loss


# Function to generate a plot for accuracy and loss.
def generate_fig(acc, loss, name="xx.png", x_label='epochs'):
    # Initializing a new figure.
    fig = plt.figure()

    # Creating the primary y-axis for accuracy.
    ax1 = fig.add_subplot(111)
    
    x = range(len(acc))
    acc_list = acc
    loss_list = loss

    # Plotting accuracy values.
    lns1 = ax1.plot(x, acc_list, label='acc')
    ax1.set_ylabel('accuracy')
    
    # Creating a secondary y-axis for loss and plotting loss values.
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x, loss_list, 'r', label='loss')
    ax2.set_ylabel('loss')
    
    # Setting the x-axis label.
    ax1.set_xlabel(x_label)

    # Combining legends from both axes and displaying them.
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0, bbox_to_anchor=(1,0.9))
    
    # Saving the figure to the given filename and displaying it.
    plt.savefig(name)


# Load data
train_acc, valid_acc, train_loss, valid_loss = load_data("logs/train_2.log")


# # Generate figures
generate_fig(train_acc, train_loss, "./figures/state_estimator/train_st_512_1.png")
generate_fig(valid_acc, valid_loss, "./figures/state_estimator/valid_st_512_1.png")

# Loading data from a file corresponding to pruned processes.
# train_acc, valid_acc, train_loss, valid_loss = load_data("logs/pruned_process_5.log", is_pruned_process=True)

# Generating a figure for the loaded data.
# generate_fig(train_loss, train_acc, "./figures/5_layers_prune.png", x_label='removed connection count')
