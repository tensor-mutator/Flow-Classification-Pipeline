"""
0 -> No reward
1 -> Success
2 -> Hit
"""

from typing import List, Tuple
import cv2
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

path = os.path.split(__file__)[0]
data_path = os.path.join(path, "data")
meta_file = os.path.join(path, "rewards.meta")

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
    y_labels = np.array(list(map(_map_rewards, y_rewards)))
    return _one_hot_y(y_labels)

def _get_X(data, resolution: Tuple, flow: bool = False) -> np.ndarray:
    if flow:
       X_flow_path = map(lambda x: x["flow"], data)
       X = np.zeros(shape=[0] + list(resolution) + [3], dtype=np.float32)
       for path in X_flow_path:
           img = cv2.imread(os.path.join(data_path, path))
           img_scaled = cv2.resize(img.astype(np.float32), resolution)
           X = np.concatenate([X, np.expand_dims(img_scaled, axis=0)])
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
           X = np.concatenate([X, np.expand_dims(img_scaled, axis=0)])
    return X

def _filter_data(data, datapoints_per_class) -> List:
    no_reward_data = list(filter(lambda x: x["reward"]==0, data))
    success_data = list(filter(lambda x: x["reward"]==1, data))
    hit_data = list(filter(lambda x: x["reward"]==-5, data))
    n_no_reward = len(no_reward_data)
    n_success = len(success_data)
    n_hit = len(hit_data)
    n_data = min(n_no_reward, n_success, n_hit, datapoints_per_class)
    data = list()
    no_reward_indices = np.random.randint(n_no_reward, size=n_data)
    success_indices = np.random.randint(n_success, size=n_data)
    hit_indices = np.random.randint(n_hit, size=n_data)
    for no_reward_idx, success_idx, hit_idx in zip(no_reward_indices, success_indices, hit_indices):
        data.extend([no_reward_data[no_reward_idx], success_data[success_idx], hit_data[hit_idx]])
    return data

def load(resolution: Tuple, datapoints_per_class: int, train_size: float = 0.8, test_size: float = 0.2) -> List:
    with open(meta_file, "r") as f_obj:
         data = json.load(f_obj)
    data = _filter_data(data, datapoints_per_class)
    y = _get_y(data)
    X = _get_X(data, resolution)
    return train_test_split(X, y, train_size=train_size, test_size=test_size)

def load_flow(resolution: Tuple, datapoints_per_class: int, train_size: float = 0.8, test_size: float = 0.2) -> List:
    with open(meta_file, "r") as f_obj:
         data = json.load(f_obj)
    data = _filter_data(data, datapoints_per_class)
    y = _get_y(data)
    X = _get_X(data, resolution, flow=True)
    return train_test_split(X, y, train_size=train_size, test_size=test_size)
