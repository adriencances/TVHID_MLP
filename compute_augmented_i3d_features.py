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
from torchvision import transforms

from torchvideotransforms import video_transforms, volume_transforms

from dataset_aux import FrameProcessor

sys.path.append("/home/acances/Code/human_interaction_SyncI3d")
from i3d import InceptionI3d as VICKY_InceptionI3d


MODEL = (VICKY_InceptionI3d, "/home/acances/Code/human_interaction_SyncI3d/params/rgb_imagenet.pt", "baseline")


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
        self.features_dir = "/home/acances/Data/TVHID/features16_aug_{}".format(self.suffix)

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
    
    def get_tensors(self, pair):
        video_id1, track_id1, begin1, end1, video_id2, track_id2, begin2, end2 = pair[:8]

        track_id1, begin1, end1 = list(map(int, [track_id1, begin1, end1]))
        track_id2, begin2, end2 = list(map(int, [track_id2, begin2, end2]))
        assert end1 - begin1 == end2 - begin2

        tensor1 = self.frame_processor.processed_frames(video_id1, track_id1, begin1, end1)
        tensor2 = self.frame_processor.processed_frames(video_id2, track_id2, begin2, end2)
        return tensor1, tensor2     # 3x16x224x244

    def get_i3d_features(self, tensor):
        # Add batch dimension and transfer to GPU
        tensor = tensor.unsqueeze(0).cuda()

        # Extract features
        self.model.eval()
        with torch.no_grad():
            features = self.model.extract_features(tensor)

        # Flatten output
        features = torch.flatten(features, start_dim=1)

        # Normalize feature vector
        features = F.normalize(features, p=2, dim=1)

        # Remove batch dimension and transfer to CPU
        features = features[0].cpu()

        return features

    def get_features(self, pair):
        tensor1, tensor2 = self.get_tensors(pair)

        features1 = self.get_i3d_features(tensor1)
        features2 = self.get_i3d_features(tensor2)

        return features1, features2

    def tensor_to_PIL(self, tensor):
        return [transforms.functional.to_pil_image(tensor[:,i]/255) \
            for i in range(tensor.shape[1])]
    
    def PIL_to_tensor(self, pil_list):
        return torch.stack([torch.tensor(np.array(e).transpose(2, 0, 1)) \
            for e in pil_list], dim=1).float()
    
    def apply_transformations(self, tensor, transformations):
        pil_list = self.tensor_to_PIL(tensor)
        for transf in transformations:
            pil_list = transf(pil_list)
        aug_tensor = self.PIL_to_tensor(pil_list)
        return aug_tensor
    
    def get_augmented_tensors(self, pair):
        tensor1, tensor2 = self.get_tensors(pair)

        # Transformation functions
        h_flip = video_transforms.RandomVerticalFlip(p=1)
        color_jitter = video_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        grayscale = video_transforms.RandomGrayscale(p=1)

        # Add untransformed tensors
        augmented_tensors = [(tensor1, tensor2)]

        # Gather the two tensors in a single tensor to apply same transformation
        tensor12 = torch.cat([tensor1, tensor2], dim=1)     # 3x32x224x244

        transformation_lists = [[grayscale], [h_flip, grayscale]] \
            + [[color_jitter] for i in range(4)] \
            + [[h_flip, color_jitter] for i in range(4)]

        for transformations in transformation_lists:
            aug_tensor12 = self.apply_transformations(tensor12, transformations)
            aug_tensor1 = aug_tensor12[:,:16]
            aug_tensor2 = aug_tensor12[:,16:]
            augmented_tensors.append((aug_tensor1, aug_tensor2))
        
        return augmented_tensors

    def print_augmented_tensors(self, pair):
        augmented_tensors = self.get_augmented_tensors(pair)
        for index, (tensor1, tensor2) in enumerate(augmented_tensors):
            self.print_tensors(tensor1, tensor2, "augmented_tensors/version_{}".format(index))
    
    def get_augmented_features(self, pair):
        augmented_tensors = self.get_augmented_tensors(pair)
        augmented_features = []

        for tensor1, tensor2 in augmented_tensors:
            features1 = self.get_i3d_features(tensor1)
            features2 = self.get_i3d_features(tensor2)
            augmented_features.append((features1, features2))

        return augmented_features

    def print_tensors(self, tensor1, tensor2, subdir):
        Path(subdir).mkdir(parents=True, exist_ok=True)

        for i in range(tensor1.shape[1]):
            filename1 = "{}/tensor1_frame_{}.jpg".format(subdir, i + 1)
            frame1 = tensor1[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename1, frame1)

            filename2 = "{}/tensor2_frame_{}.jpg".format(subdir, i + 1)
            frame2 = tensor2[:,i,:,:].numpy().transpose(2, 1, 0)
            cv2.imwrite(filename2, frame2)
    
    def compute_all_features(self):
        print("Computing features for positive pairs")
        for video_id in tqdm.tqdm(self.positive_pairs_by_video):
            class_name = video_id.split("_")[0]
            class_index = self.class_indices[class_name]

            pairs = self.positive_pairs_by_video[video_id]
            for i, pair in enumerate(pairs):
                features_subdir = "{}/positive/{}/pair_{}".format(self.features_dir, video_id, i)
                Path(features_subdir).mkdir(parents=True, exist_ok=True)
                augmented_features = self.get_augmented_features(pair)
                for index, (features_1, features_2) in enumerate(augmented_features):
                    output_file = "{}/features_v{}.pkl".format(features_subdir, index)
                    with open(output_file, "wb") as f:
                        pickle.dump((features_1, features_2, class_index), f)
        
        print("Computing features for negative pairs")
        for video_id in tqdm.tqdm(self.negative_pairs_by_video):
            class_name = "negative"
            class_index = self.class_indices[class_name]

            pairs = self.negative_pairs_by_video[video_id]
            for i, pair in enumerate(pairs):
                features_subdir = "{}/negative/{}/pair_{}".format(self.features_dir, video_id, i)
                Path(features_subdir).mkdir(parents=True, exist_ok=True)
                augmented_features = self.get_augmented_features(pair)
                for index, (features_1, features_2) in enumerate(augmented_features):
                    output_file = "{}/features_v{}.pkl".format(features_subdir, index)
                    with open(output_file, "wb") as f:
                        pickle.dump((features_1, features_2, class_index), f)


handler = TVHIDPairsHandler(MODEL=MODEL)
pair = ["handShake_0006",0,36,52,"handShake_0006",2,36,52]
t1, t2 = handler.get_tensors(pair)
p1 = handler.tensor_to_PIL(t1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide 'yes' as an argument to compute the features")
        sys.exit(1)
    if sys.argv[1] != "yes":
        print("Provide 'yes' as an argument to compute the features")
        sys.exit(1)
    
    handler = TVHIDPairsHandler(MODEL=MODEL)
    handler.compute_all_features()

