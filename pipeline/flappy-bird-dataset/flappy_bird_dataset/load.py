from typing import List, Tuple
import cv2
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

data_path = os.path.join(__file__, "data")
meta_file = os.path.join(__file__, "rewards.meta")

def _one_hot_y(y: np.ndarray) -> np.ndarray:
    one_hot = np.zeros(shape=[np.size(y), 3], dtype=np.float32)
    one_hot[np.arange(np.size(y)), y] = 1
    return one_hot

def load(resolution: Tuple, train_size: float = 0.8, test_size: float = 0.2) -> List:
    with open(meta_file, "r") as f_obj:
         data = json.load(f_obj)
    return train_test_split(X, y, train_size=train_size, test_size=test_size)

def load_flow(resolution: Tuple, train_size: float = 0.8, test_size: float = 0.2) -> List:
    with open(meta_file, "r") as f_obj:
         data = json.load(f_obj)
    return train_test_split(X, y, train_size=train_size, test_size=test_size)
