from typing import Dict, Optional
import random, os

import h5py
import numpy as np

# Class responsible for generating training sub-images from larger scans.
class SubImageGenerator:

    SUB_IMG_DIM = 64  # Dimension of the sub-image

    def __init__(self, border_margin: int):
        """ 
        Initializes object to crop sub-images.

        Args:
            border_margin: Pixels to skip at the border of the large image.
        """
        self.border_margin = border_margin

    @staticmethod
    def calc_probability_vector(entry: Dict) -> np.ndarray:
        """
        Computes a normalized distribution of labels.

        Args:
            entry: Dictionary with label information.
        """
        histogram = np.histogram(2 * entry['state'], bins=[0, 1, 2, 3, 4, 5])[0]
        return histogram / np.sum(histogram)

    @staticmethod
    def generate_labels(entry: Dict) -> Dict:
        """
        Populates the data dictionary with labels.

        Args:
            entry: Dictionary with label data.
        """
        entry['state'] = SubImageGenerator.calc_probability_vector(entry)
        return entry

    def random_crop_origin(self, axis_length: int) -> int:
        """ 
        Chooses a random origin point for cropping.

        Args:
            axis_length: Length of the axis to crop from.
        """
        return random.choice(range(int(self.border_margin), int(axis_length - self.SUB_IMG_DIM - self.border_margin)))

    def generate_sub_image(self, sensor_vals: np.ndarray, state_vals: np.ndarray, 
                           x_axis: np.ndarray, y_axis: np.ndarray, 
                           noise_scale: float) -> Dict:
        """
        Produces a cropped data dictionary from larger arrays.

        Args:
            sensor_vals: 2D array of noisy sensor measurements.
            state_vals: 2D array of state data.
            x_axis: 1D array of x-axis data.
            y_axis: 1D array of y-axis data.
            noise_level: Magnitude of noise present in data.
        """
        origin_x, origin_y = self.random_crop_origin(len(x_axis)), self.random_crop_origin(len(y_axis))
        
        sub_img_data = {
            'measurement': sensor_vals[origin_y:origin_y + self.SUB_IMG_DIM, origin_x:origin_x + self.SUB_IMG_DIM],
            'state': state_vals[origin_y:origin_y + self.SUB_IMG_DIM, origin_x:origin_x + self.SUB_IMG_DIM],
            'noise_level': noise_scale,
            'V1': np.linspace(x_axis[origin_x], x_axis[origin_x + self.SUB_IMG_DIM], self.SUB_IMG_DIM),
            'V2': np.linspace(y_axis[origin_y], y_axis[origin_y + self.SUB_IMG_DIM], self.SUB_IMG_DIM)
        }

        return self.generate_labels(sub_img_data)

    def generate_multiple_sub_images(self, h5_file_path: str, data_key: str = 'sensor',
                                     x_key: str = 'V_P1_vec', y_key: str = 'V_P2_vec',
                                     num_crops: int = 10, save_images: bool = True,
                                     return_images: bool = False) -> Optional[Dict]:
        """
        Processes an entire HDF5 file to generate sub-images.

        Args:
            h5_file_path: Path to HDF5 file.
            data_key: Key to z-axis data in HDF5.
            x_key: Key to x-axis data in HDF5.
            y_key: Key to y-axis data in HDF5.
            num_crops: Number of crops per large image.
            save_images: Whether to save the data.
            return_images: Whether to return the cropped data.
        """
        if not save_images and not return_images:
            raise ValueError('Either save_images or return_images should be True.')

        cropped_dataset = {}

        with h5py.File(h5_file_path, 'r') as h5_file:
            for idx, dataset in h5_file.items():
                for i in range(num_crops):
                    cropped_data = self.generate_sub_image(
                        dataset['output'][data_key][:], dataset['output']['state'][:],
                        dataset[x_key][:], dataset[y_key][:], dataset['noise_level'][()]
                    )
                    cropped_dataset[f"{idx}_{i}"] = cropped_data

        print(f"Generated {len(cropped_dataset.keys())} sub-images.")

        if save_images:
            # It saves the data in the same folder as the raw data, but in the interim folder as npz file.
            save_images_path = h5_file_path.replace('raw', 'interim').replace('hdf5', 'npz')
            os.makedirs(os.path.dirname(save_images_path), exist_ok=True)
            np.savez_compressed(save_images_path, **cropped_dataset)

        if return_images:
            return cropped_dataset
