from typing import Tuple, Union, Optional
import random, glob, os, json

import concurrent.futures
import numpy as np

import ml_tools.config as config

class DatasetPreparator:

    def __init__(self, seed: Optional[int] = None):
        """
        Instantiate an object for dataset preparation.

        Args:
            seed: Optional seed value for random operations.
        """
        self.seed = seed

    def calculate_minimum_class_count(self, label_array: np.ndarray) -> int:
        """
        Finds the minimum count of samples per class in the dataset.

        Args:
            label_array: Array of labels, each specifying the class of a data sample.

        Returns:
            The smallest count of samples among all classes.
        """
        class_max_indices = np.argmax(label_array, axis=-1)
        min_class_count = len(class_max_indices)
        
        for i in range(label_array.shape[-1]):
            current_count = np.sum(class_max_indices == i)
            if current_count < min_class_count:
                min_class_count = current_count

        return min_class_count

    def equalize_class_distribution(self, features: np.ndarray, labels: np.ndarray, 
                                    extra_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rebalances the dataset by resampling it based on class with the smallest count.

        Args:
            features: Array of feature vectors.
            labels: Array of corresponding class labels.
            extra_labels: Optional additional labels.

        Returns:
            Tuple of resampled feature and label arrays.
        """
        random_instance = np.random.default_rng(self.seed)
        smallest_class_size = self.calculate_minimum_class_count(labels)
        
        resampled_features = []
        resampled_labels = []
        resampled_extra_labels = []

        for index in range(labels.shape[-1]):
            selected_data_indices = (labels.argmax(axis=-1) == index)
            
            class_specific_features = features[selected_data_indices]
            class_specific_labels = labels[selected_data_indices]
            
            if extra_labels is not None:
                class_specific_extra_labels = extra_labels[selected_data_indices]

            random_indices = list(range(class_specific_features.shape[0]))
            random_instance.shuffle(random_indices)
            
            resampled_features.append(class_specific_features[random_indices[:smallest_class_size]])
            resampled_labels.append(class_specific_labels[random_indices[:smallest_class_size]])

            if extra_labels is not None:
                resampled_extra_labels.append(class_specific_extra_labels[random_indices[:smallest_class_size]])

        combined_features = np.concatenate(resampled_features, axis=0)
        combined_labels = np.concatenate(resampled_labels, axis=0)

        if extra_labels is not None:
            combined_extra_labels = np.concatenate(resampled_extra_labels, axis=0)

        shuffle_order = list(range(combined_features.shape[0]))
        random_instance.shuffle(shuffle_order)
        
        if extra_labels is not None:
            return combined_features[shuffle_order], combined_extra_labels[shuffle_order]
        else:
            return combined_features[shuffle_order], combined_labels[shuffle_order]

    def map_noise_to_classes(self, state_labels: np.ndarray, noise_magnitudes: np.ndarray,
                             lower_bounds: Optional[np.ndarray] = None,
                             upper_bounds: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Maps the noise magnitudes for each example to a noise class(low, medium, high) 
        based on the state of the example and the noise thresholds for each state.
        The default values for the lower and upper bounds are taken from the paper.

        Args:
            state_labels (np.ndarray): Array containing the state labels for each example.
                                       Shape is assumed to be (num_examples, num_states).
            noise_magnitudes (np.ndarray): Array containing the noise magnitudes for each example.
                                     Shape is assumed to be (num_examples, ).
            lower_bounds (Optional[np.ndarray]): Array specifying the lower noise thresholds for each state.
                                                   Shape is (num_states,).
            upper_bounds (Optional[np.ndarray]): Array specifying the upper noise thresholds for each state.
                                                    Shape is (num_states,).

        Returns:
            np.ndarray: Array containing the noise classes per example. Shape is (num_examples, num_quality_classes).
        """
        num_quality_classes = config.NUM_QUALITY_CLASSES
        num_states = config.NUM_STATES

        if upper_bounds is None:
            upper_bounds = [1.22, 1.00, 1.21, 0.68, 2.00]
        if lower_bounds is None:
            lower_bounds = [0.31, 0.32, 0.41, 0.05, 0.47]

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        quality_classes_per_example = np.zeros(noise_magnitudes.shape + (num_quality_classes,))

        num_states = state_labels.shape[-1]

        quality_classes_per_state = np.zeros(noise_magnitudes.shape + (num_quality_classes,) + (num_states,))

        for index in range(num_states):
            quality_classes_per_state[noise_magnitudes <= lower_bounds[index], 0, index] = 1
            quality_classes_per_state[(noise_magnitudes > lower_bounds[index]) & \
                                   (noise_magnitudes <= upper_bounds[index]), 1, index] = 1
            quality_classes_per_state[noise_magnitudes > upper_bounds[index], 2, index] = 1

        quality_classes_per_example = np.einsum('ijk,ik->ij', quality_classes_per_state, state_labels)

        return quality_classes_per_example

    def save_dataset(
            self,
            training_data: np.ndarray,
            training_labels: np.ndarray,
            validation_data: np.ndarray,
            validation_labels: np.ndarray, 
            save_path: str
            ) -> None:
        

            unprocessed_dataset_path = os.path.join(save_path, 'unprocessed_dataset')
            training_folder_path = os.path.join(unprocessed_dataset_path, 'training')
            validation_folder_path = os.path.join(unprocessed_dataset_path, 'validation')

            if not os.path.exists(unprocessed_dataset_path):
                os.makedirs(unprocessed_dataset_path)    

            if not os.path.exists(training_folder_path):
                os.makedirs(training_folder_path)
            if not os.path.exists(validation_folder_path):
                os.makedirs(validation_folder_path)

            np.savez(os.path.join(training_folder_path, 'training_data.npz'), training_data=training_data)
            np.savez(os.path.join(training_folder_path, 'training_labels.npz'), training_labels=training_labels)
            np.savez(os.path.join(validation_folder_path, 'validation_data.npz'), validation_data=validation_data)
            np.savez(os.path.join(validation_folder_path, 'validation_labels.npz'), validation_labels=validation_labels)
    
    def load_npz_file(self, file):
        file_data = np.load(file, allow_pickle=True)
        return {key: file_data[key] for key in file_data.keys()}
    
    def load_data(self, path_or_dict):
        try:
            is_dict = False
            if path_or_dict.endswith('.npz'):
                data_files = [path_or_dict]
            else:
                data_files = glob.glob(path_or_dict + '*.npz')

            data_collection = {}
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_file = {executor.submit(self.load_npz_file, file): file for file in data_files}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print(f'{file} generated an exception: {exc}')
                    else:
                        data_collection.update(data)
        except TypeError:
            is_dict = True
            data_collection = path_or_dict
        
        return data_collection, is_dict
    
    def prepare_dataset(self, 
                        path_or_dict : Union[str, dict],
                        save_path: Optional[str] = None,
                        train_validation_split: float = 0.8, 
                        sensor_data_key: str = 'measurement',
                        label_key_name: str = 'state', 
                        should_resample: bool = True, 
                        min_noise_limits: Optional[np.ndarray] = None, 
                        max_noise_limits: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Prepares and partitions the dataset for model training and validation.

            Parameters:
                path_or_dict : Either the path to a .npz file or a dictionary containing the dataset.
                train_validation_split: Percentage of the dataset to use for training.
                sensor_data_key: The key corresponding to sensor data in the dataset.
                label_key_name: The key for label data in the dataset.
                should_resample: Flag to indicate whether resampling should be applied.
                min_noise_limits: Lower thresholds for noise classes.
                max_noise_limits: Upper thresholds for noise classes.

            Returns:
                A tuple containing Training Data, Training Labels, validation Data, and validation Labels.
            """

            data_collection, is_dict = self.load_data(path_or_dict)

            data_keys = list(data_collection.keys())
            random.Random(self.seed).shuffle(data_keys)
            sensor_readings = []
            state_labels = []

            if label_key_name != 'state':
                additional_labels = []
            else:
                additional_labels = None
                training_labels = None
                validation_labels = None

            if label_key_name == 'data_quality':
                is_data_quality = True
                label_key_name = 'noise_level '
            else:
                is_data_quality = False

            for key in data_keys:

                if is_dict:
                    individual_data = data_collection[key]
                else:
                    individual_data = data_collection[key].item()

                sensor_data = individual_data[sensor_data_key]
                sensor_readings.append(sensor_data.reshape(config.SUB_IMG_DIM, config.SUB_IMG_DIM, 1))
                state_labels.append(individual_data['state'])

                if additional_labels is not None:
                    additional_labels.append(individual_data[label_key_name])

            sensor_readings = np.array(sensor_readings)
            state_labels = np.array(state_labels)

            if additional_labels is not None:
                additional_labels = np.array(additional_labels)

            total_samples = sensor_readings.shape[0]
            print("Total number of samples :", total_samples)
            num_training_samples = int(train_validation_split * total_samples)

            training_data = sensor_readings[:num_training_samples]
            print("Training data info:", training_data.shape)
            training_states = state_labels[:num_training_samples]

            if additional_labels is not None:
                training_labels = additional_labels[:num_training_samples]

            validation_data = sensor_readings[num_training_samples:]
            print("Validation data info:", validation_data.shape)
            validation_states = state_labels[num_training_samples:]

            if additional_labels is not None:
                validation_labels = additional_labels[num_training_samples:]

            if is_data_quality:
                training_labels = self.map_noise_to_classes(
                    training_states, training_labels,
                    lower_bounds=min_noise_limits,
                    upper_bounds=max_noise_limits,
                )
                validation_labels = self.map_noise_to_classes(
                    validation_states, validation_labels,
                    lower_bounds=min_noise_limits,
                    upper_bounds=max_noise_limits,
                )
       
            if should_resample:
                training_data, training_labels = self.equalize_class_distribution(
                    training_data, training_states, training_labels)
                validation_data, validation_labels = self.equalize_class_distribution(
                    validation_data, validation_states, validation_labels)
            elif not should_resample and label_key_name == 'state':
                training_labels = training_states
                validation_labels = validation_states

            if additional_labels is not None and len(training_labels.shape) == 1:
                np.expand_dims(training_labels, 1)
            if additional_labels is not None and len(validation_labels.shape) == 1:
                np.expand_dims(validation_labels, 1)

            
            if save_path is not None:
                self.save_dataset(
                    training_data,
                    training_labels,
                    validation_data,
                    validation_labels,
                    save_path
                )


            return training_data, training_labels, validation_data, validation_labels





