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
        self.frames_dir = "/home/acances/Data/TVHID/keyframes"
        self.tracks_dir = "/home/acances/Data/TVHID/tracks"
        self.pairs_dir = "/home/acances/Data/TVHID/pairs16"

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.frames_dir, self.tracks_dir)

        self.class_indices = {
            "negative": 0,
            "handShake": 1,
            "highFive": 2,
            "hug": 3,
            "kiss": 4
        }

        random.seed(seed)
        self.gather_positive_pairs()
        self.gather_negative_pairs()
        self.create_data()

    def gather_positive_pairs(self):
        print("Gathering positive pairs")
        self.positive_pairs = []
        pairs_files = glob.glob("{}/positive/*".format(self.pairs_dir))
        for file in tqdm.tqdm(pairs_files):
            class_name = file.split("/")[-1].split("_")[1]
            class_index = self.class_indices[class_name]
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.positive_pairs.append(pair + [class_index])
        random.shuffle(self.positive_pairs)
    
    def gather_negative_pairs(self):
        print("Gathering negative pairs")
        self.negative_pairs = []
        pairs_files = glob.glob("{}/negative/*".format(self.pairs_dir))
        for file in tqdm.tqdm(pairs_files):
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.negative_pairs.append(pair + [0])
        random.shuffle(self.negative_pairs)
    
    def create_data(self):
        # Concatenate positive and negative pairs, and shuffle
        self.data = self.positive_pairs + self.negative_pairs
        random.shuffle(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        nb_pairs = len(self.data)
        assert index < nb_pairs
        pair = self.data[index]
        video_id1, track_id1, begin1, end1, video_id2, track_id2, begin2, end2, label = pair

        track_id1, begin1, end1 = list(map(int, [track_id1, begin1, end1]))
        track_id2, begin2, end2 = list(map(int, [track_id2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, track_id2, begin2, end2)
        
        return tensor1, tensor2, label

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)


if __name__ == "__main__":
    dataset = TVHID("train")
    # print(len(dataset))
    # tensor1, tensor2, label = dataset[0]
    # print(tensor1.shape)
    # print(tensor2.shape)


