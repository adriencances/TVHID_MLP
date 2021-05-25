import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from dataset_aux import FrameProcessor

sys.path.append("/home/acances/Code/human_interaction_SyncI3d")
from i3d import InceptionI3d as VICKY_InceptionI3d

sys.path.append("/home/acances/Code/gsig_I3d/PyVideoResearch")
from models.bases.aj_i3d import InceptionI3d as GSIG_InceptionI3d


MODELS = [(VICKY_InceptionI3d, "/home/acances/Code/human_interaction_SyncI3d/params/rgb_imagenet.pt", "baseline"),
          (GSIG_InceptionI3d, "/home/acances/Code/gsig_I3d/PyVideoResearch/downloaded_models/aj_rgb_imagenet.pth", "gsig-i3d")]


class TVHIDPairsHandler:
    def __init__(self, MODEL):
        self.model_class = MODEL[0]
        self.weights_file = MODEL[1]
        self.suffix = MODEL[2]

        self.w = 224
        self.h = 224
        self.alpha = 0.1

        self.frames_dir = "/home/acances/Data/TVHID/keyframes"
        self.tracks_dir = "/home/acances/Data/TVHID/tracks"
        self.pairs_dir = "/home/acances/Data/TVHID/pairs16"
        self.features_dir = "/home/acances/Data/TVHID/features16_{}".format(self.suffix)

        self.frame_processor = FrameProcessor(self.w, self.h, self.alpha, self.frames_dir, self.tracks_dir)

        self.prepare_i3d()

        self.class_indices = {
            "negative": 0,
            "handShake": 1,
            "highFive": 2,
            "hug": 3,
            "kiss": 4
        }
        self.class_names = {(self.class_indices[name], name) for name in self.class_indices}

        self.gather_positive_pairs()
        self.gather_negative_pairs()
    
    def prepare_i3d(self):
        self.model = self.model_class()
        weights = torch.load(self.weights_file)
        self.model.load_state_dict(weights)
        self.model.cuda()

    def gather_positive_pairs(self):
        self.positive_pairs_by_video = {}
        pairs_files = glob.glob("{}/positive/*".format(self.pairs_dir))
        for file in pairs_files:
            video_id = file.split("/")[-1].split(".")[0][6:]
            class_name = file.split("/")[-1].split("_")[1]
            class_index = self.class_indices[class_name]
            
            self.positive_pairs_by_video[video_id] = []
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.positive_pairs_by_video[video_id].append(pair + [class_index])
    
    def gather_negative_pairs(self):
        self.negative_pairs_by_video = {}
        pairs_files = glob.glob("{}/negative/*".format(self.pairs_dir))
        for file in pairs_files:
            video_id = file.split("/")[-1].split(".")[0][6:]

            self.negative_pairs_by_video[video_id] = []
            with open(file, "r") as f:
                for line in f:
                    pair = line.strip().split(",")
                    self.negative_pairs_by_video[video_id].append(pair + [0])
    
    def get_features(self, pair):
        video_id1, track_id1, begin1, end1, video_id2, track_id2, begin2, end2 = pair[:8]

        track_id1, begin1, end1 = list(map(int, [track_id1, begin1, end1]))
        track_id2, begin2, end2 = list(map(int, [track_id2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, track_id2, begin2, end2)

        # Add batch dimension and transfer to GPU
        tensor1 = tensor1.unsqueeze(0).cuda()
        tensor2 = tensor2.unsqueeze(0).cuda()

        self.model.eval()
        features1 = self.model.extract_features(tensor1)
        features2 = self.model.extract_features(tensor2)

        # Flatten output
        features1 = torch.flatten(features1, start_dim=1)
        features2 = torch.flatten(features2, start_dim=1)

        # Normalize each feature vector (separately)
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        # Remove batch dimension and transfer to CPU
        features1 = features1[0].cpu()
        features2 = features2[0].cpu()

        return features1, features2
    
    def compute_all_features(self):
        print("Computing features for positive pairs")
        for video_id in tqdm.tqdm(self.positive_pairs_by_video):
            class_name = video_id.split("_")[0]
            class_index = self.class_indices[class_name]

            pairs = self.positive_pairs_by_video[video_id]
            features_subdir = "{}/positive/{}".format(self.features_dir, video_id)
            Path(features_subdir).mkdir(parents=True, exist_ok=True)
            for i, pair in enumerate(pairs):
                features_1, features_2 = self.get_features(pair)
                output_file = "{}/features_pair{}.pkl".format(features_subdir, i)
                with open(output_file, "wb") as f:
                    pickle.dump((features_1, features_2, class_index), f)
        
        print("Computing features for negative pairs")
        for video_id in tqdm.tqdm(self.negative_pairs_by_video):
            class_name = "negative"
            class_index = self.class_indices[class_name]

            pairs = self.negative_pairs_by_video[video_id]
            features_subdir = "{}/negative/{}".format(self.features_dir, video_id)
            Path(features_subdir).mkdir(parents=True, exist_ok=True)
            for i, pair in enumerate(pairs):
                features_1, features_2 = self.get_features(pair)
                output_file = "{}/features_pair{}.pkl".format(features_subdir, i)
                with open(output_file, "wb") as f:
                    pickle.dump((features_1, features_2, class_index), f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide 'yes' as an argument to compute the features")
        sys.exit(1)
    if sys.argv[1] != "yes":
        print("Provide 'yes' as an argument to compute the features")
        sys.exit(1)
    
    handler = TVHIDPairsHandler(MODEL=MODELS[0])
    handler.compute_all_features()
