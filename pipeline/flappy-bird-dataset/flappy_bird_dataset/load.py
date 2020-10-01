"""
0 -> No reward
1 -> Reward
2 -> Hit
"""

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

def _map_rewards(reward) -> int:
    if reward == 0:
       return 0
    elif reward == 1:
       return 1
    else:
       return 2

def _get_y(data) -> np.ndarray:
    y_rewards = map(lambda x: x["reward"], data)
    return np.array(list(map(_map_rewards, y_rewards)))

def _get_X(data, resolution: Tuple, flow: bool = False) -> np.ndarray:
    if flow:
       X_flow_path = map(lambda x: x["flow"], data)
       X = np.zeros(shape=[0] + list(resolution) + [3], dtype=np.float32)
       for path in X_flow_path:
           img = cv2.imread(os.path.join(data_path, path))
           img_scaled = cv2.resize(img.astype(np.float32), resolution)
           X = np.concatenate([X, img_scaled])
    else:
       X_src_img_path = map(lambda x: x["src_image"], data)
       X_dest_img_path = map(lambda x: x["dest_image"], data)
       X = np.zeros(shape=[0, 2] + list(resolution) + [3], dtype=np.float32)
       for src_path, dest_path in zip(X_src_img_path, X_dest_img_path):
           src_img = cv2.imread(os.path.join(data_path, src_path))
           dest_img = cv2.imread(os.path.join(data_path, dest_path))
           src_img_scaled = cv2.resize(src_img.astype(np.float32), resolution)
           dest_img_scaled = cv2.resize(dest_img.astype(np.float32), resolution)
           img_scaled = np.stack([src_img_scaled, dest_img_scaled], axis=0)
           X = np.concatenate([X, img_scaled])
    return X

def load(resolution: Tuple, train_size: float = 0.8, test_size: float = 0.2) -> List:
    with open(meta_file, "r") as f_obj:
         data = json.load(f_obj)
    y = _get_y(data)
    X = _get_X(data, resolution)
    return train_test_split(X, y, train_size=train_size, test_size=test_size)

def load_flow(resolution: Tuple, train_size: float = 0.8, test_size: float = 0.2) -> List:
    with open(meta_file, "r") as f_obj:
         data = json.load(f_obj)
    y = _get_y(data)
    X = _get_X(data, resolution, flow=True)
    return train_test_split(X, y, train_size=train_size, test_size=test_size)
