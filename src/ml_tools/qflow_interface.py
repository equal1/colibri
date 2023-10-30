import os
import glob
import sys

import numpy as np

# Ensure the module path is included in the system path(this is done to allow the import of modules from the parent directory)
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import necessary modules
from ml_tools.data.subimage_generator import SubImageGenerator
from ml_tools.preprocessing.dataset_preparator import DatasetPreparator
from ml_tools.preprocessing.subimage_preprocessor import SubImagePreprocessor
from ml_tools.models.model_utils import prepare_dataloader

# Diferent locations for where to work with the data
# TODO: Maybe i should make a class for this
quality_control_paths = {
    "raw": '../data/raw/data_qflow_v2/simulated/sim_uniform/',
    "interim": '../data/interim/data_qflow_v2/simulated/sim_uniform/',
    "interim_save": "../data/interim/data_qflow_v2/simulated/sim_uniform/",
    "unprocessed_dataset": '../data/interim/data_qflow_v2/simulated/sim_uniform/unprocessed_dataset/',
    "processed": '../data/processed/data_qflow_v2/simulated/sim_uniform/',
    "processed_dataset": '../data/processed/data_qflow_v2/simulated/sim_uniform/processed_dataset/',
}

state_estimator_paths = {
    "raw": '../data/raw/data_qflow_v2/simulated/sim_normal/',
    "interim": '../data/interim/data_qflow_v2/simulated/sim_normal/',
    "interim_save": "../data/interim/data_qflow_v2/simulated/sim_normal/",
    "unprocessed_dataset": '../data/interim/data_qflow_v2/simulated/sim_normal/unprocessed_dataset/',
    "processed": '../data/processed/data_qflow_v2/simulated/sim_normal/',
    "processed_dataset": '../data/processed/data_qflow_v2/simulated/sim_normal/processed_dataset/',
}


def generate_sub_images(data_path, file_ext='*.hdf5', border_margin=20):
    """
    Generate random sub-images of 30x30 from the raw data.

    Parameters:
    - data_path: Path to the raw data.
    - file_ext: File extension of the raw data files. Default is '*.hdf5'.
    - border_margin: Margin around the borders to ignore. Default is 20.
    """

    data_files = glob.glob(os.path.join(data_path, file_ext))
    for raw_data_file in data_files:
        generator = SubImageGenerator(border_margin=border_margin)
        generator.generate_multiple_sub_images(raw_data_file, save_images=True, return_images=False)


def prepare_dataset(subimage_folder_path, save_path, train_validation_split=0.75, label_key_name='state'):
    """
    Prepare a dataset from sub-images.

    Parameters:
    - subimage_folder_path: Path to the folder containing sub-images.
    - save_path: Path to save the prepared dataset.
    - train_validation_split: Ratio to split the training and validation data. Default is 0.75.
    - label_key_name: Key name to retrieve labels from the dataset. Default is 'state'.
    """
    
    preparator = DatasetPreparator()
    return preparator.prepare_dataset(
        path_or_dict=subimage_folder_path,
        save_path=save_path,
        train_validation_split=train_validation_split, 
        label_key_name=label_key_name, 
    )


def load_npz_data(dataset_path, data_type, processed=False):
    """
    Load data and labels from .npz files.

    Parameters:
    - dataset_path: Path to the dataset.
    - data_type: Type of data ('training' or 'validation').
    - processed: Flag to determine if the data is processed. Default is False.

    Returns:
    - data: Loaded data from the .npz file.
    - labels: Loaded labels from the .npz file.
    """

    if processed:
        data_path = os.path.join(dataset_path, data_type, f"processed_{data_type}_data.npz")
        labels_path = os.path.join(dataset_path, data_type, f"processed_{data_type}_labels.npz")
    else:
        data_path = os.path.join(dataset_path, data_type, f"{data_type}_data.npz")
        labels_path = os.path.join(dataset_path, data_type, f"{data_type}_labels.npz")


    data = np.load(data_path)["data"]
    labels = np.load(labels_path)["labels"]
    
    return data, labels



def preprocess_subimages(data, labels, train_or_validation, save_path,
                         auto_invert=False, noise_reduction=None, clip_value=None, cutoff_value=None):

    """
    Preprocess sub-images by applying various filters and transformations.

    Parameters:
    - data: Sub-images data.
    - labels: Sub-images labels.
    - train_or_validation: Type of data ('training' or 'validation').
    - save_path: Path to save the preprocessed data.
    - auto_invert, noise_reduction, clip_value, cutoff_value: Preprocessing parameters.

    Returns:
    - Processed data and labels.
    """
    
    noise_reduction = noise_reduction if noise_reduction is not None else []
    
    preprocessor = SubImagePreprocessor(
        auto_invert=auto_invert,
        noise_reduction=noise_reduction,
        clip_value=clip_value,
        cutoff_value=cutoff_value
    )
    return preprocessor.preprocess_subimages(
        data,
        labels,
        train_or_validation=train_or_validation,
        save_path=save_path
    )


def load_and_preprocess_data(data_paths, label_key_name='state'):
    """
    The complete data loading and preprocessing pipeline.

    Parameters:
    - data_paths: Paths dictionary to access different stages of the data.
    - label_key_name: Key name to retrieve labels from the dataset. Default is 'state'.

    Returns:
    - Training and validation data and labels.
    """

    # Generate sub-images
    generate_sub_images(data_paths['raw'])

    # Prepare dataset
    train_data, train_labels, validation_data, validation_labels = prepare_dataset(
        data_paths['interim'], data_paths['interim_save'], label_key_name=label_key_name
    )

    # Load and preprocess training data
    train_data, train_labels = load_npz_data(data_paths['unprocessed_dataset'], 'training', processed=False)
    validation_data, validation_labels = load_npz_data(data_paths['unprocessed_dataset'], 'validation', processed=False)

    # Preprocess subimages
    processed_train_data, processed_train_labels = preprocess_subimages(
        train_data, train_labels, 'training', data_paths['processed']
    )
    processed_validation_data, processed_validation_labels = preprocess_subimages(
        validation_data, validation_labels, 'validation', data_paths['processed']
    )

    # Load processed data
    train_data, train_labels = load_npz_data(data_paths['processed_dataset'], 'training', processed=True)
    validation_data, validation_labels = load_npz_data(data_paths['processed_dataset'], 'validation', processed=True)

    return train_data, train_labels, validation_data, validation_labels


def prepare_test_data(test_data_path):
    """
    Load, preprocess and transpose the test data and then return it.

    Parameters:
    - test_data_path: Path to the test data.

    Returns:
    - Processed test data and labels.
    """

    data_files = glob.glob(test_data_path + 'dataset_*/' + '*.npy')    
    exp_data = []; exp_labels = []
    for f in data_files:
        d = np.load(f, allow_pickle=True).item()
        exp_data.append(d['sensor'])
        exp_labels.append(d['label'])
    exp_labels = np.array(exp_labels)
    exp_prepper = SubImagePreprocessor(
        auto_invert=True,
        noise_reduction=[],
        clip_value=None,
        cutoff_value=None
    )
    proc_exp_data, exp_labels = exp_prepper.preprocess_subimages(exp_data, exp_labels)
    proc_exp_data_transpose = np.transpose(proc_exp_data, (0, 3, 1, 2))
    return proc_exp_data_transpose, exp_labels

def read_qflow_test_data(batch_size=64):
    """
    Read the QFlow test data and create a DataLoader.

    Parameters:
    - batch_size: Batch size for the DataLoader. Default is 64.

    Returns:
    - DataLoader with test data.
    """
        
    test_data_path = '../data/raw/data_qflow_v2/experimental/exp_small/'
    test_data, test_labels = prepare_test_data(test_data_path)
    test_dataloader = prepare_dataloader(test_data, test_labels, type='val', batch_size=batch_size)
    return test_dataloader

def read_qflow_data(batch_size=64, label_key_name='state', is_prepared=False, fast_search=False):
    """
    Read the QFlow data.

    Parameters:
    - batch_size: Batch size for the DataLoader. Default is 64.
    - label_key_name: Key name to retrieve labels from the dataset. Default is 'state'.
    - is_prepared: Flag to check if data is already prepared. Default is False.
    - fast_search: Flag for a quicker, smaller dataset preparation. Default is False.

    Returns:
    - DataLoader for training and validation data.
    """

    if label_key_name == 'state':
        simulated_data_type = 'sim_normal'
        dataset_paths = state_estimator_paths
    else:
        simulated_data_type = 'sim_uniform'
        dataset_paths = quality_control_paths

    if is_prepared:
        dataset_path = f'../data/processed/data_qflow_v2/simulated/{simulated_data_type}/processed_dataset/'
        train_data, train_labels = load_npz_data(dataset_path, 'training', processed=True)
        validation_data, validation_labels = load_npz_data(dataset_path, 'validation', processed=True)
    else:
        train_data, train_labels, validation_data, validation_labels = load_and_preprocess_data(dataset_paths, label_key_name=label_key_name)
    
    # Transpose the data
    train_data_transpose = np.transpose(train_data, (0, 3, 1, 2))
    validation_data_transpose = np.transpose(validation_data, (0, 3, 1, 2))

    if fast_search:
        train_subset_indices = np.random.choice(train_data_transpose.shape[0], train_data_transpose.shape[0]//100, replace=False)
        train_data_transpose = train_data_transpose[train_subset_indices]
        train_labels = train_labels[train_subset_indices]

        val_subset_indices = np.random.choice(validation_data_transpose.shape[0], validation_data_transpose.shape[0]//100, replace=False)
        validation_data_transpose = validation_data_transpose[val_subset_indices]
        validation_labels = validation_labels[val_subset_indices]

    train_dataloader = prepare_dataloader(train_data_transpose, train_labels, type='train', batch_size=batch_size, shuffle=True)
    val_dataloader = prepare_dataloader(validation_data_transpose, validation_labels, type='val', batch_size=batch_size, shuffle=True)

    return train_dataloader,val_dataloader

    

