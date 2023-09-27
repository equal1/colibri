import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import ml_tools.config as config

class StateEstimator(pl.LightningModule):
    def __init__(self, model_opt='noise_opt'):
        super(StateEstimator, self).__init__()
        
        # model options
        if model_opt == 'noise_opt':
            # best values from manual hyperparameter search so far
            self.lr = 1.21e-3
            n_filters = [[22, 22], [35, 35]]
            self.drop_rates = [[0.655, 0.655], [0.194, 0.194]]
            self.n_cnn = 2
            self.cnn_stack = 2
            self.layer_norm = False
            kernel_size = 7
            stride = 2
            padding = (kernel_size - 1) // 2
        # it might be useful in the future to only train on the noiseless data
        elif model_opt == 'noiseless_opt':
            # not yet optimized
            self.lr = 3.45e-3
            n_filters = [[23], [7], [18]]
            self.drop_rates = [[0.12], [0.28], [0.30]]
            self.n_cnn = 3
            self.cnn_stack = 1
            self.layer_norm = True
            kernel_size = 5
            stride = 2
            padding = (kernel_size - 1) // 2
        else:
            raise ValueError('model_opt not recognized')
        
        layers = []
        for i in range(self.n_cnn):
            for j in range(self.cnn_stack):
                in_channels = 1 if i == 0 and j == 0 else (n_filters[i-1][-1] if j == 0 else n_filters[i][j-1])
                out_channels = n_filters[i][j]
                
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
                nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='relu')

                if self.layer_norm:
                    layers.append(nn.LayerNorm(out_channels))
                else:
                    layers.append(nn.BatchNorm2d(out_channels))
            
                layers.append(nn.Dropout(self.drop_rates[i][j]))

                layers.append(nn.ReLU())

        
        self.features = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_filters[-1][-1], config.NUM_STATES)


    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        y_class_indices = torch.argmax(y, dim=1)
        accuracy = (torch.argmax(y_hat, dim=1) == y_class_indices).float().mean()

        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_class_indices = torch.argmax(y, dim=1)
        accuracy = (torch.argmax(y_hat, dim=1) == y_class_indices).float().mean()
        
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
