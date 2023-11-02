import os
import time
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
import copy
from ml_tools import qflow_interface
from ml_tools.models.state_estimator import StateEstimatorWrapper
from ml_tools.models.model_utils import infer


class ModelPruner:
    def __init__(self, model_config, optimizer_config, train_batch_size, test_batch_size, path_to_model_checkpoint):
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.train_queue, self.valid_queue = self.load_data()
        self.test_queue = self.load_test_data()
        
        self.model_checkpoint = self.load_model_checkpoint(path_to_model_checkpoint)
        self.model = self.load_model()

        self.trained_dict = copy.deepcopy(self.model.state_dict())
        self.idx_to_key_map = {idx: k for idx, (k, _) in enumerate(self.trained_dict.items())}

    def load_data(self):
        return qflow_interface.read_qflow_data(batch_size=self.train_batch_size, is_prepared=True, fast_search=False)

    def load_test_data(self):
        return qflow_interface.read_qflow_test_data(batch_size=self.test_batch_size)
    
    def load_model_checkpoint(self, path_to_model_checkpoint):
        checkpoint = torch.load(path_to_model_checkpoint)
        return checkpoint

    def load_model(self):
        checkpoint = torch.load(self.model_checkpoint)
        model = StateEstimatorWrapper(self.model_config, self.optimizer_config)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        return model

    def prune(self, max_to_cut, logging_path="resources/metrics/prune_logs/", path_to_pruned_model="models/intermediary_models/pruned_models"):
        cut_count = 0
        to_del_order = []
        cutted_set = set(to_del_order)
        
        logging.basicConfig(filename=logging_path, filemode="w", level=logging.INFO)

        while cut_count < max_to_cut:
            result = []
            cur_dict = copy.deepcopy(self.trained_dict)

            for todel in to_del_order:
                cur_dict[self.idx_to_key_map[todel]] = torch.zeros_like(cur_dict[self.idx_to_key_map[todel]])

            for target in range(len(self.trained_dict)):
                if "weight" not in self.idx_to_key_map[target] and "bias" not in self.idx_to_key_map[target]:
                    continue
                inner_dict = copy.deepcopy(cur_dict)
                if target in cutted_set:
                    continue

                inner_dict[self.idx_to_key_map[target]] = torch.zeros_like(inner_dict[self.idx_to_key_map[target]])
                self.model.load_state_dict(inner_dict)
                acc, test_loss = infer(self.model, self.criterion, self.test_queue)
                result.append([target, acc])

                if acc == 100.00:
                    print("early stop")
                    break
                logging.info('cut edge idx:%d loss:%f acc:%f ' % (target, test_loss, acc))

            result.sort(key=lambda x: -x[1])
            j = 0  

            cutted_set.add(result[j][0])
            to_del_order.append(result[j][0])

            test_dict = copy.deepcopy(cur_dict)
            test_dict[self.idx_to_key_map[result[j][0]]] = torch.zeros_like(test_dict[self.idx_to_key_map[result[j][0]]])

            self.model.load_state_dict(test_dict)
            torch.save(self.model, (path_to_pruned_model + '/weights_cut_trial_%d.pkl' % cut_count))

            test_acc, test_loss = self.infer(self.model, self.criterion, self.test_queue)
            torch.cuda.empty_cache()
            
            cut_count += 1
            logging.info("OUTSIDE cut_count:%d to_del_order:%s after_del_acc:%f test_acc:%f test_loss:%f" % (
            cut_count, str(to_del_order), result[j][1], test_acc, test_loss))

        return to_del_order
