import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import ml_tools.config as config 


# Constants
OUT_CLASSES = 5
BATCH_SIZE = 512
EPOCHS = 100
MIN_LR = 0.001

class QualityControl(pl.LightningModule):
    def __init__(self):
        super(QualityControl, self).__init__()

        self.lr = 2.65e-4
        k_size = [[7, 3]]
        cnn_maxpool = True
        cnn_stack = 2
        n_cnn = 1

        n_filters = [[184, 249]]
        drop_rates = [[0.05, 0.0]]
        self.normalization = True
        ave_pool = True
        self.activation_fn = nn.SiLU()

        dense_n = 1
        self.dense_dropout = [0.6]
        dense_units = [161]

        if cnn_maxpool:
            self.cnn_stride = 1
        else:
            self.cnn_stride = 2

        layers = []
        for i in range(n_cnn):
           for j in range(cnn_stack):
                stride = self.cnn_stride if j == cnn_stack - 1 else 1
                in_channels = 1 if i == 0 and j == 0 else n_filters[i][j - 1]
                out_channels = n_filters[i][j]
                kernel_size = k_size[i][j]
                padding = (kernel_size - 1) // 2
    
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    
                if self.normalization:
                    layers.append(nn.BatchNorm2d(out_channels))
                
                layers.append(self.activation_fn)
                layers.append(nn.Dropout(drop_rates[i][j]))


           if cnn_maxpool:
               layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        
        fc_layers = []
        for i in range(dense_n):
            in_features = n_filters[-1][-1] if i == 0 else dense_units[i - 1]
            out_features = dense_units[i]
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(self.activation_fn)
            fc_layers.append(nn.Dropout(self.dense_dropout[i]))

        

        self.features = nn.Sequential(*layers)
        if ave_pool:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_avg_pool = None
        self.fc_layers = nn.Sequential(*fc_layers)
        self.fc_final = nn.Linear(dense_units[-1], config.NUM_QUALITY_CLASSES)

    def forward(self, x):
        x = self.features(x)
        if self.global_avg_pool:
            x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = self.fc_final(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=MIN_LR)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_class_indices = torch.argmax(y, dim=1)
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        accuracy = (torch.argmax(y_hat, dim=1) == y_class_indices).float().mean()

        self.log('train_accuracy', accuracy, on_epoch=True, logger=True)
        self.log('train_loss', loss, on_epoch=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_class_indices = torch.argmax(y, dim=1)
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        accuracy = (torch.argmax(y_hat, dim=1) == y_class_indices).float().mean()

        self.log('val_accuracy', accuracy, on_epoch=True, logger=True)
        self.log('val_loss', loss, on_epoch=True, logger=True)
