import argparse
import cv2
import numpy as np
import os
import time
import torch
import pickle

from tqdm import tqdm

from models.dla import get_pose_net
from utils.transforms import decode_results

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--red_folders", required=True, type=str)
    parser.add_argument("--nored_folders", required=True, type=str)
    parser.add_argument("--input_size", type=int, default=32)

    args = parser.parse_args()
    return args

args = get_args()

red_folders = args.red_folders.split(',')
nored_folders = args.nored_folders.split(',')

class RedNoRedDataset():
    def __init__(self, red_forlder_paths, nored_folder_paths, size):
        self.size = size
        red_files = []
        nored_files = []

        for path in red_folders:
            red_files += [path+f for f in list(os.listdir(path)) if not f.startswith('.')]
        for path in nored_folder_paths:
            nored_files += [path+f for f in list(os.listdir(path)) if not f.startswith('.')]

        self.files = red_files + nored_files
        self.labels = np.array([1] * len(red_files) + [0] * len(nored_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        img = cv2.imread(self.files[index])
        img = cv2.resize(img,(self.size, self.size))
        label = self.labels[index]

        return img, label, file_path

dataset = RedNoRedDataset(red_folders, nored_folders, args.input_size)
