import cv2
import numpy as np
import pickle
import sys
import glob
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from MLP import ClassificationMLP
from compute_features import TVHIDPairsHandler


SEGMENT_LENGTH = 16

tracks_dir = "/home/acances/Data/TVHID/tracks"


def get_tracks(video_id):
    tracks_files = sorted(glob.glob("{}/{}/*".format(tracks_dir, video_id)))
    tracks = []
    for tracks_file in tracks_files:
        with open(tracks_file, "rb") as f:
            tracks += pickle.load(f)
    tracks = [e[0] for e in tracks]
    return tracks


def get_pairs_to_process(video_id):
    tracks = get_tracks(video_id)

    pairs = []
    for id1 in range(len(tracks)):
        track1 = tracks[id1]
        for id2 in range(id1 + 1, len(tracks)):
            track2 = tracks[id2]

            inter_b = int(max(track1[0, 0], track2[0, 0]))
            inter_e = int(min(track1[-1, 0], track2[-1, 0]))
            if inter_e - inter_b + 1 < SEGMENT_LENGTH:
                continue

            for begin_frame in range(inter_b, inter_e - SEGMENT_LENGTH + 2, SEGMENT_LENGTH // 2):
                pair = []
                pair += [video_id, id1, begin_frame, begin_frame + SEGMENT_LENGTH]
                pair += [video_id, id2, begin_frame, begin_frame + SEGMENT_LENGTH]
                pairs.append(pair)

    return pairs


def test_model(video_ids, checkpoint_file=None):
    model = ClassificationMLP()
    model.cuda()
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model"])

    # TVHID pairs handler using baseline I3D
    handler = TVHIDPairsHandler(checkpoint_file=None)

    for video_id in video_ids:
        pairs = get_pairs_to_process(video_id)

        for pair in pairs:
            features1, features2 = handler.get_features(pair)

            features1 = features1.unsqueeze(0).cuda()
            features2 = features2.unsqueeze(0).cuda()

            # Pass inputs to the model
            probs = model(features1, features2) # shape : bx5





