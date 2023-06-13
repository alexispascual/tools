import os
import re
import cv2
import sys
import yaml
import math
import random
import numpy as np
import platform
import csv

from pprint import pprint as pp
from pathlib import Path
from typing import List, Tuple, Union, Any


def load_config(default_config: str, silent=False):
    """
    Multi-use function. Exposes command-line to provide alternative configurations to training scripts.
    Also works as a stand-alone configuration importer in notebooks and scripts.
    """
    if len(sys.argv) == 1:
        # Then the file being called is the only argument so return the default configuration
        config_file = default_config
    elif len(sys.argv) == 2:
        # Then a specific configuration file has been used so load it
        config_file = str(sys.argv[1])
    elif all([len(sys.argv) == 3, sys.argv[1] == '-f']):
        config_file = default_config
    else:
        print(sys.argv)
        raise ValueError('CLI only accepts 0 args (default) or 1 arg (path/to/config).')

    with open(str(config_file), 'r') as f:
        y = yaml.full_load(f)
        if not silent:
            print('Experimental parameters\n------')
            pp(y)
        return y


def check_directory(directory: str, create=False):
    if not os.path.isdir(directory):

        if create:
            print(f"Creating directory {directory}")
            os.makedirs(directory)

            return

        print(f"{directory} does not exist")
        sys.exit(1)


def check_file(*args: Union[str, Path]) -> bool:
    ret = True

    for arg in args:
        if not os.path.isfile(arg):
            print(f"{arg} does not exist!")
            ret = False
        else:
            print(f"Found {arg}.")

    return ret


def show_image(image, image_2=None, bgr=False):
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_2 is not None:
        if bgr:
            image_2 = cv2.cvtColor(image_2, cv2.BGR2RGB)
        if image.shape[0] > image_2.shape[0]:
            image = cv2.resize(image, (image_2.shape[1], image_2.shape[0]))
        elif image.shape[0] < image_2.shape[0]:
            image_2 = cv2.resize(image_2, (image.shape[1], image.shape[0]))

        image = np.concatenate((image, image_2), axis=1)

    cv2.imshow('Image', image)
    cv2.moveWindow('Image', 30, 0)
    key = cv2.waitKey(0)

    if key == 27:  # Check for ESC key press
        return -1
    else:
        return 0


def save_image(image, 
               image_2=None, 
               directory='./data', 
               file_name='default_name.jpg',
               bgr=True):
    
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_2 is not None:
        if bgr:
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
        image = np.concatenate((image, image_2), axis=1)
    
    file_name = os.path.join(directory, file_name)
    return cv2.imwrite(file_name, image)


def scan_horizon_files(directory: str):

    horizons = get_images_from_directory(directory)
    horizons = [get_file_id(f) for f in horizons]

    return horizons


def fill_horizon_line(image):

    copy_image = image.copy()
    copy_image = cv2.flip(image, 0)

    horizon_line = np.argmax(copy_image, axis=0)
    
    height = image.shape[0]
    width = image.shape[1]

    for col in range(width):
        corrected_horizon = height - horizon_line[col][0]
        image[corrected_horizon + 1:height, col] = (255, 255, 255)
        image[0:corrected_horizon, col] = (0, 0, 0)

    return image


def fill_horizon_line_top_down(image):

    horizon_line = np.argmax(image, axis=0)
    
    height = image.shape[0]
    width = image.shape[1]

    for col in range(width):
        if horizon_line[col][0] == 0:
            image[0:height, col] = (255, 255, 255)
        else:
            image[0:horizon_line[col][0], col] = (0, 0, 0)
            image[horizon_line[col][0] + 1:height, col] = (255, 255, 255)

    return image


def generate_depth_map(point_cloud: np.ndarray, 
                       normalize: bool = False):

    depth_estimate = np.zeros((point_cloud.shape[0], point_cloud.shape[1]))
    point_cloud = np.nan_to_num(point_cloud)

    for w in range(point_cloud.shape[0]):
        for h in range(point_cloud.shape[1]):
            value = point_cloud[w][h]

            if not np.isnan(value.any()):
                depth_estimate[w][h] = math.sqrt(value[0] * value[0] 
                                                 + value[1] * value[1]
                                                 + value[2] * value[2])
            else:
                depth_estimate[w][h] = 0

    if normalize:
        normalized_depth = (depth_estimate - np.min(depth_estimate)) / (np.max(depth_estimate) - np.min(depth_estimate)) * 255

        return normalized_depth.astype(np.uint8)

    else:
        return depth_estimate


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_images_from_directory(filepath: str, shuffle: bool = False) -> List[Path]:
    paths = []

    for entry in os.scandir(filepath):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".jpeg")
                or entry.path.endswith(".png")) and entry.is_file():
            paths.append(entry.path)

    if shuffle:
        random.shuffle(paths)
    else:
        paths.sort(key=natural_keys)

    return paths


def get_labels_from_directory(filepath: str, shuffle: bool = False) -> List[Path]:
    paths = []

    for entry in os.scandir(filepath):
        if entry.path.endswith(".xml") and entry.is_file():
            paths.append(entry.path)

    if shuffle:
        random.shuffle(paths)
    else:
        paths.sort(key=natural_keys)

    return paths


def get_csv_files_from_directory(filepath: str):

    paths = []

    for entry in os.scandir(filepath):
        if entry.path.endswith(".csv") and entry.is_file():
            paths.append(entry.path)

    paths.sort(key=natural_keys)
    return paths


def get_file_id(filepath: Path) -> str:
    if platform.system() == "Windows":
        file_id = ''.join(re.split(r'(\d+)', str(filepath).split('\\')[-1])[1:-1])
    else:
        file_id = ''.join(re.split(r'(\d+)', str(filepath).split('/')[-1])[1:-1])

    return file_id


def calculate_depth_error(gt_depth_map: np.ndarray, predicted_depth_map: np.ndarray) -> Tuple[np.ndarray, float]:

    error_map = np.subtract(predicted_depth_map, gt_depth_map).astype(np.float64)
    non_zero = np.count_nonzero(gt_depth_map)
    error = np.sum(error_map**2) / non_zero

    return error_map, error


def save_values(coordinates: list, 
                results_save_file: str):
    """
    Writes the training and testing accuracy for each trial into a csv file
    
    Parameters
    ----------
    row: list

    results_save_file: str
        
    """
    with open(results_save_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        for coord in coordinates:
            writer.writerow([coord[0], coord[1], coord[2]])


def read_positions_csv(filepath: str) -> List[Any]:
    ret = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ret.append(tuple([float(row[0]), float(row[2]), float(row[1])]))

    return np.array(ret)
