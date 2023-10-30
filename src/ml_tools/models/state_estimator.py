import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pytorch_lightning as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from ml_tools import qflow_interface

class StateEstimator(nn.Module):
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
            self.normalization = True
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
            self.normalization = True
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

                if self.normalization:
                    layers.append(nn.BatchNorm2d(out_channels))
                       
                layers.append(nn.Dropout(self.drop_rates[i][j]))

                layers.append(nn.ReLU())

        
        self.features = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_filters[-1][-1], config.NUM_STATES)


    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


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
        
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
    
# -----------------------------------------------------------------------------
MIN_DROP_RATE = 0.1

class DynamicStateEstimator(nn.Module):
    def __init__(self,
                 in_channels=1, 
                 cnn_blocks=4,
                 kernel_size=7, 
                 base_filters=32, 
                 fc_blocks=2, 
                 num_classes=5, 
                 activation='relu', 
                 cnn_pooling='max_pool',
                 global_pooling='adaptive_avg', 
                 conv_drop_rate=0.66, 
                 fc_drop_rate=0.5, 
                 drop_rate_decay=0.1
                 ):
        
        super(DynamicStateEstimator, self).__init__()

        
        self.padding = (kernel_size - 1) // 2
        final_filters = base_filters * 2 ** (cnn_blocks - 1)

        if final_filters >= 256:
            final_filters = 256
            
        self.activation = self._get_activation(activation)

        self.cnn_pooling = self._get_pooling(cnn_pooling)
        self.global_pooling = self._get_pooling(global_pooling)

        self.features = self._build_cnn_layers(base_filters, conv_drop_rate, drop_rate_decay, cnn_blocks, kernel_size, in_channels)

        self.flatten = nn.Flatten()

        self.classifier = self._build_classifier(final_filters, num_classes, fc_blocks, fc_drop_rate,)

        self._initialize_weights(activation)

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        assert name in activations, f"Activation {name} not supported. Choose from {list(activations.keys())}."
        return activations[name]
    
    def _get_pooling(self, type):
        poolings = {
            'adaptive_avg': nn.AdaptiveAvgPool2d((1, 1)),
            'adaptive_max': nn.AdaptiveMaxPool2d((1, 1)),
            'avg_pool':  nn.AvgPool2d(2, stride=2),
            'max_pool': nn.MaxPool2d(2, stride=2)
        }
        assert type in poolings, f"Pooling {type} not supported. Choose from {list(poolings.keys())}."
        return poolings[type]

    def _build_classifier(self, fc_hidden, num_classes, fc_stack, fc_drop_rate):
        layers = []
        in_features = fc_hidden
        out_features = fc_hidden // 2

        for _ in range(fc_stack-1):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(fc_drop_rate))
            layers.append(self.activation)

            in_features = out_features
            out_features //= 2 

        layers.append(nn.Linear(in_features, num_classes))  

        return nn.Sequential(*layers)


    def _build_cnn_layers(self, base_filters, drop_rate, drop_rate_decay, cnn_blocks, kernel_size, input_channels):
        layers = []
        in_channels = input_channels
        out_channels = base_filters

        for block in range(cnn_blocks):
                
            conv_block = self._conv_block(in_channels, out_channels, kernel_size, drop_rate)
            residual_block = ResidualBlock(in_channels, out_channels, conv_block)

            layers.append(residual_block)
            
            if block != cnn_blocks - 1:
                layers.append(self.cnn_pooling)

            in_channels = out_channels

            if out_channels >= 256:
                out_channels = 256
            else:
                out_channels *= 2

            drop_rate = max(drop_rate - drop_rate_decay, MIN_DROP_RATE)

        return nn.Sequential(*layers)

    def _conv_block(self, in_channels, out_channels, kernel_size, drop_rate):
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=self.padding, groups=in_channels), # Depthwise Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1), # Pointwise Convolution
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Dropout(drop_rate) 
        ]
        return nn.Sequential(*layers)
    
    
    def _initialize_weights(self, activation_name):
        if activation_name in ['relu', 'leaky_relu']:
            he_init = True
            xavier_init = False
        elif activation_name in ['tanh', 'sigmoid']:
            xavier_init = True
            he_init = False
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if xavier_init:
                    nn.init.xavier_normal_(m.weight)
                elif he_init:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_name)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pooling(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers):
        super(ResidualBlock, self).__init__()
        self.sublayers = sublayers
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.sublayers(x) + self.shortcut(x)

# -----------------------------------------------------------------------------


# Constants
OUT_CLASSES = 5
BATCH_SIZE = 512
EPOCHS = 100
MIN_LR = 0.001

class StateEstimatorWrapper(pl.LightningModule):
    def __init__(self, model_config, optimizer_config):
        super(StateEstimatorWrapper, self).__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.model = DynamicStateEstimator(**model_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_class_indices = torch.argmax(y, dim=1)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y_class_indices).float().mean()
        self.log('val_accuracy', accuracy, on_epoch=True, logger=True)
        self.log('val_loss', loss, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_class_indices = torch.argmax(y, dim=1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y_class_indices).float().mean()
        self.log('train_accuracy', accuracy, on_epoch=True, logger=True)
        self.log('train_loss', loss, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.optimizer_config['learning_rate'],
            momentum=self.optimizer_config['momentum'],
            weight_decay=self.optimizer_config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=MIN_LR)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_queue, _ = qflow_interface.read_qflow_data(batch_size=BATCH_SIZE, label_key_name='state', is_prepared=True, fast_search=False)
        return train_queue

    def val_dataloader(self):
        _, valid_queue = qflow_interface.read_qflow_data(batch_size=BATCH_SIZE, label_key_name='state', is_prepared=True, fast_search=False)
        return valid_queue


