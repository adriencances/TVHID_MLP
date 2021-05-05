import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from dataset_aux import FrameProcessor


class TVHIDPairs(data.Dataset):
    def __init__(self, phase="train", seed=0):
        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.phase = phase
        self.features_dir = "/home/acances/Data/TVHID/features16"

        self.gather_video_ids()
        self.gather_positive_pairs()
        self.gather_negative_pairs()
        self.create_data()
    
    def gather_video_ids(self):
        self.video_ids = []
        video_ids_file = "/home/acances/Data/TVHID/split/{}.txt".format(self.phase)
        with open(video_ids_file, "r") as f:
            for line in f:
                video_id = line.strip()
                self.video_ids.append(video_id)

    def gather_positive_pairs(self):
        print("Gathering positive pairs")
        self.positive_pairs = []
        features_subdirs = glob.glob("{}/positive/*".format(self.features_dir))
        for subdir in tqdm.tqdm(features_subdirs):
            video_id = subdir.split("/")[-1]
            if video_id not in self.video_ids:
                continue
            self.positive_pairs += sorted(glob.glob("{}/*".format(subdir)))
        random.shuffle(self.positive_pairs)
    
    def gather_negative_pairs(self):
        print("Gathering negative pairs")
        self.negative_pairs = []
        features_subdirs = glob.glob("{}/negative/*".format(self.features_dir))
        for subdir in tqdm.tqdm(features_subdirs):
            video_id = subdir.split("/")[-1]
            if video_id not in self.video_ids:
                continue
            self.negative_pairs += sorted(glob.glob("{}/*".format(subdir)))
        random.shuffle(self.negative_pairs)
    
    def create_data(self):
        # Concatenate positive and negative pairs, and shuffle
        self.data = self.positive_pairs + self.negative_pairs
        random.shuffle(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        nb_pairs = len(self.data)
        assert index < nb_pairs
        features_file = self.data[index]
        
        with open(features_file, "rb") as f:
            tensor1, tensor2, label = pickle.load(f)

        return tensor1, tensor2, label

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)


if __name__ == "__main__":
    dataset = TVHIDPairs("train")


