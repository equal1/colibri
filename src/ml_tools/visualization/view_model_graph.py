import os
import sys

from torchview import draw_graph

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from ml_tools.models.state_estimator import StateEstimatorWrapper

def visualize_model(model, input_size=(512,1,32,32), save_graph=False):
    # device='meta' -> no memory is consumed for visualization
    model_graph = draw_graph(model, input_size=input_size, device='meta',save_graph=save_graph)
    return model_graph


    



