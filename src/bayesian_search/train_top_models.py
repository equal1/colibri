import os
import sys
import pickle

import pytorch_lightning as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from ml_tools.models.state_estimator import StateEstimatorWrapper
from search.parsers import parse_trial_log_file, extract_information_from_trials


log_filepath = "logs/extract_trials_log.txt"
study_path = '/local/rbals/playground/architecture_search/search/logs/study_state_estimator.pkl'
extract_information_from_trials(study_path,log_file=log_filepath, order_by='value')
filtered_trials = parse_trial_log_file(log_filepath)


for idx, trial in enumerate(filtered_trials):
    model_config = trial['Params']
    
    # Remove optimizer params from the model_config for clarity
    optimizer_config = {
        'learning_rate': model_config.pop('learning_rate'),
        'momentum': model_config.pop('momentum'),
        'weight_decay': model_config.pop('weight_decay'),
    }
    
    model = StateEstimatorWrapper(model_config, optimizer_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"top_models/trial_{trial['Trial']}", 
        filename="{epoch:02d}-{val_accuracy:.4f}",
        save_top_k=1,
        monitor="val_accuracy",
        mode='max'
    )

    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)
