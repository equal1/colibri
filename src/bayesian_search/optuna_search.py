import os
import sys
import pickle

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger  

import optuna
from optuna.integration import PyTorchLightningPruningCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from ml_tools.models.state_estimator import StateEstimatorWrapper

optimal_model_config = {
    'in_channels': 1,
    'cnn_blocks': 6,
    'kernel_size': 5,
    'base_filters': 64,
    'fc_blocks': 2,
    'num_classes': 5,
    'activation': 'tanh', # (relu,leaky_relu,tanh)
    'cnn_pooling': 'max_pool', # (max_pool,avg_pool)
    'global_pooling': 'adaptive_avg', # (adaptive_avg,adaptive_max)
    'conv_drop_rate': 0.1496526893126031,
    'fc_drop_rate': 0.13400162558848863,
    'drop_rate_decay': 0.2400133983649985
}

optimal_optimizer_config = {
        'learning_rate': 0.006834557945902603,
        'momentum': 0.9790420970484083,
        'weight_decay': 0.0012367741418186516,
    }

def objective(trial: optuna.trial.Trial) -> float:

    # Suggested hyperparameters
    model_config = {
        'in_channels': 1,
        'cnn_blocks': trial.suggest_int('cnn_blocks', 1, 6),
        'kernel_size': trial.suggest_int('kernel_size', 3, 7, step=2),
        'base_filters': trial.suggest_categorical('base_filters', [16, 32, 64]),
        'fc_blocks': trial.suggest_int('fc_blocks', 1, 3),
        'num_classes': 5,
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh']),
        'cnn_pooling': trial.suggest_categorical('cnn_pooling', ['max_pool', 'avg_pool']),
        'global_pooling': trial.suggest_categorical('global_pooling', ['adaptive_avg', 'adaptive_max']),
        'conv_drop_rate': trial.suggest_float('conv_drop_rate', 0.1, 0.75),
        'fc_drop_rate': trial.suggest_float('fc_drop_rate', 0.1, 0.75),
        'drop_rate_decay': trial.suggest_float('drop_rate_decay', 0.1, 0.5),
    }

    optimizer_config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'momentum': trial.suggest_float('momentum', 0.5, 1),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
    }

    model = StateEstimatorWrapper(model_config, optimizer_config)

    # PyTorch Lightning's pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_accuracy")

    # Use MLFlowLogger
    logger = MLFlowLogger(
        experiment_name="Optuna_Search",
        save_dir="mlruns/" 
    )

    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=100, 
        accelerator="auto",
        callbacks=[pruning_callback],
    )

    trainer.fit(model)
    
    return trainer.callback_metrics["val_accuracy"].item()

# Create a study object and specify the direction is "minimize".
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)  # You can change n_trials to any number you like

# save the study to a file
with open("logs/study_state_estimator.pkl", "wb") as f:
    pickle.dump(study, f)

