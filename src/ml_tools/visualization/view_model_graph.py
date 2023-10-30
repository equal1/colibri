import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from ml_tools.models.state_estimator import DynamicStateEstimator

config = {
        'in_channels': 1,
        'cnn_blocks': 6,
        'kernel_size': 7,
        'base_filters': 16,
        'fc_blocks': 2,
        'num_classes': 5,
        'activation': 'relu',
        'cnn_pooling': 'max_pool',
        'global_pooling': 'adaptive_avg',
        'conv_drop_rate': 0.66,
        'fc_drop_rate': 0.5,
        'drop_rate_decay': 0.1
    }

from torchview import draw_graph
# from search import data_utils, utils

model = DynamicStateEstimator(**config)
# print(utils.model_summary(model,(1, 32, 32)))


# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_size=(512,1,32,32), device='meta',save_graph=True)

    



