import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math as m
from torchvision import transforms, utils
import cv2
import sys
import pickle
import tqdm
import glob
import os


class FrameProcessor:
    def __init__(self, w, h, alpha, frames_dir, tracks_dir, normalized_boxes=False):
        # (w, h) : dimensions of processed frame
        # alpha : quantity by which the bounding box areas get enlarged
        self.w = w
        self.h = h
        self.alpha = alpha
        self.normalized_boxes = normalized_boxes

        self.frames_dir = frames_dir
        self.tracks_dir = tracks_dir


    def enlarged_box(self, box):
        # Enlarge the box area by 100*alpha percent while preserving
        # the center and the aspect ratio
        beta = 1 + self.alpha
        x1, y1, x2, y2 = box
        dx = x2 - x1
        dy = y2 - y1
        x1 -= (np.sqrt(beta) - 1)*dx/2
        x2 += (np.sqrt(beta) - 1)*dx/2
        y1 -= (np.sqrt(beta) - 1)*dy/2
        y2 += (np.sqrt(beta) - 1)*dy/2
        return x1, y1, x2, y2

    def preprocessed_frame(self, video_id, n):
        # n : frame index in the timestamp (frame indices start at 1)
        frame_file = "{}/{}/{:06d}.jpg".format(self.frames_dir, video_id, n)
        assert os.path.isfile(frame_file), frame_file
        # frame : H * W * 3
        frame = cv2.imread(frame_file)
        # frame : 3 * W * H
        frame = frame.transpose(2, 1, 0)
        frame = torch.from_numpy(frame)
        return frame

    def processed_frame(self, frame, box):
        # frame : 3 * W * H
        # (w, h) : dimensions of new frame

        C, W, H = frame.shape
        x1, y1, x2, y2 = box

        # If box is in normalized coords, i.e.
        # image top-left corner (0,0), bottom-right (1, 1),
        # then turn normalized coord into absolute coords
        if self.normalized_boxes:
            x1 = x1*W
            x2 = x2*W
            y1 = y1*H
            y2 = y2*H

        # Round coords to integers
        X1 = max(0, m.floor(x1))
        X2 = max(0, m.ceil(x2))
        Y1 = max(0, m.floor(y1))
        Y2 = max(0, m.ceil(y2))
        
        dX = X2 - X1
        dY = Y2 - Y1

        # Get the cropped bounding box
        boxed_frame = transforms.functional.crop(frame, X1, Y1, dX, dY)
        dX, dY = boxed_frame.shape[1:]

        # Compute size to resize the cropped bounding box to
        if dY/dX >= self.h/self.w:
            w_tild = m.floor(dX/dY*self.h)
            h_tild = self.h
        else:
            w_tild = self.w
            h_tild = m.floor(dY/dX*self.w)
        assert w_tild <= self.w
        assert h_tild <= self.h

        # Get the resized cropped bounding box
        resized_boxed_frame = transforms.functional.resize(boxed_frame, [w_tild, h_tild])

        # Put the resized cropped bounding box on a gray canvas
        new_frame = 127*torch.ones(C, self.w, self.h)
        i = m.floor((self.w - w_tild)/2)
        j = m.floor((self.h - h_tild)/2)
        new_frame[:, i:i+w_tild, j:j+h_tild] = resized_boxed_frame
        return new_frame

    def track(self, video_id, track_id):
        tracks_files = sorted(glob.glob("{}/{}/*".format(self.tracks_dir, video_id)))
        tracks = []
        for tracks_file in tracks_files:
            with open(tracks_file, "rb") as f:
                tracks += pickle.load(f)
        tracks = [e[0] for e in tracks]
        return tracks[track_id]

    def processed_frames(self, video_id, track_id, begin_frame, end_frame):
        # begin_frame, end_frame : 0-based indices

        track = self.track(video_id, track_id)
        b = int(track[0, 0])

        processed_frames = []
        for i in range(begin_frame, end_frame):
            frame = self.preprocessed_frame(video_id, i + 1)
            track_frame_index = i - b
            box = track[track_frame_index][1:5]
            box = self.enlarged_box(box)
            processed_frame = self.processed_frame(frame, box)
            processed_frames.append(processed_frame)
        processed_frames = torch.stack(processed_frames, dim=1)
        return processed_frames
