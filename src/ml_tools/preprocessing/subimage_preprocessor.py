from typing import List, Optional, Union

import os

import numpy as np
from skimage.transform import resize as img_resize

import ml_tools.preprocessing.preprocessing_functions as data_utils
import ml_tools.config as config

class SubImagePreprocessor:
    def __init__(self, auto_invert: bool = False, noise_reduction: List[str] = [], 
                 clip_value: Optional[float] = None, cutoff_value: Optional[float] = None) -> None:
        """
        Constructor for the SubImagePreprocessor class.

        Parameters:
            auto_invert: Flag to decide if auto inversion should be applied.
            noise_reduction: List of strings specifying which noise reduction methods should be used.
            clip_value: Value to clip, used if 'clip' is in noise_reduction.
            cutoff_value: Value for thresholding, used if 'threshold' is in noise_reduction.
        """
        self.auto_invert = auto_invert

        allowed_noise_reduction = ['threshold', 'clip']
        if not set(noise_reduction).issubset(allowed_noise_reduction):
            raise ValueError(f'Invalid noise reduction methods {noise_reduction}. Allowed values: {allowed_noise_reduction}')
        self.noise_reduction = noise_reduction
        self.clip_value = clip_value
        self.cutoff_value = cutoff_value

    def save_dataset(
            self,
            proccessed_img_batch: np.ndarray,
            train_or_test: str,
            save_path: str
            ) -> None:
        
            processed_dataset_path = os.path.join(save_path, 'processed_dataset')

            if train_or_test == 'train':
                train_or_test = 'training'
                train_or_test_path = os.path.join(processed_dataset_path, train_or_test)
            else:
                train_or_test = 'testing'
                train_or_test_path = os.path.join(processed_dataset_path, train_or_test)

            if not os.path.exists(train_or_test_path):
                os.makedirs(train_or_test_path)

            numpy_file_name = 'processed_' + train_or_test + '_data.npz'

            np.savez(os.path.join(train_or_test_path, numpy_file_name), numpy_file_name=proccessed_img_batch)


    def preprocess_subimage(self, img: np.ndarray) -> np.ndarray:
        """
        Enhances a single subimage.

        Parameters:
            img: Input array that represents a subimage.

        Output:
            Preprocessed subimage.
            
        """

        # Calculate derivative
        img = data_utils.compute_gradient(img)

        # Apply threshold
        if 'threshold' in self.noise_reduction:
            img = data_utils.apply_threshold(img, self.cutoff_value)

        # Apply clipping
        if 'clip' in self.noise_reduction:
            img = data_utils.apply_clipping(img, self.clip_value)

        # Normalize
        img = data_utils.normalize_zscore(img)

        # Auto-inversion(it helps with detecting patterns in the image)
        if self.auto_invert:
            img = data_utils.correct_skewness(img)

        desired_shape = (config.SUB_IMG_DIM, config.SUB_IMG_DIM, 1)
        if img.shape != desired_shape:
            img = img_resize(img, desired_shape)

        return img

    def preprocess_subimages(self, 
                            img_batch: np.ndarray,
                            train_or_test: str = "",
                            save_path: str = None
                            ) -> np.ndarray:
        """
        Preprocesses a batch of subimages.

        Parameters:
            img_set: Input array of subimages.

        Output:
            An array of preprocessed subimages.
        """
        proccessed_img_batch = np.array([self.preprocess_subimage(img) for img in img_batch])
        if save_path is not None:
            self.save_dataset(proccessed_img_batch, train_or_test, save_path)
        return proccessed_img_batch
