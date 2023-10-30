import matplotlib.pyplot as plt
import numpy as np
import json

# Assuming data is already loaded
data = json.load(open('model_performance.json', 'r'))

model_metrics = [(entry["trial_number"],entry["metrics"]["recall"],entry["metrics"]["confusion_matrix"]) for entry in data]

# Sort by accuracy
model_metrics.sort(key=lambda x: x[1], reverse=True)

# Normalize the confusion matrix
for i in range(len(model_metrics)):
    model_metrics[i] = (model_metrics[i][0],model_metrics[i][1],np.array(model_metrics[i][2]) / np.sum(model_metrics[i][2], axis=1, keepdims=True))

# Print the models
for i in range(len(model_metrics)):
    print(f"Model {i+1}: Trial {model_metrics[i][0]}")
    print(f"Recall: {model_metrics[i][1]}")
    print(f"Confusion matrix: \n{model_metrics[i][2]}")
    print("\n")




