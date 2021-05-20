import os
import sys
import numpy as np
import cv2
import glob
import tqdm
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from train import train_model


nb_layers_VALS = [1, 2, 3, 4, 5]
optimizer_class_VALS = [optim.SGD, optim.Adam]
learning_rate_VALS = [10**i for i in [-1, -2, -3, -4, -5]]
batch_size_VALS = [8, 64]

all_VALS = [nb_layers_VALS, optimizer_class_VALS, learning_rate_VALS, batch_size_VALS]


results = {}
configurations = list(itertools.product(*all_VALS))
total = len(configurations)
epochs = 100
with tqdm.tqdm(total=total) as pbar:
    for cnt, config in enumerate(configurations):
        nb_layers, optimizer_class, learning_rate, batch_size = config
        train_losses, train_accs, train_mean_accs, all_train_accs_by_classes, val_losses, val_accs, val_mean_accs, all_val_accs_by_classes = \
        train_model(
            nb_layers=nb_layers,
            optimizer_class=optimizer_class,
            lr=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            record=False
        )
        results[config] = (train_losses[-1], train_accs[-1], train_mean_accs[-1], *all_train_accs_by_classes[-1], val_losses[-1], val_accs[-1], val_mean_accs[-1], *all_val_accs_by_classes[-1])
        pbar.update(1)


output_file = "/home/acances/Code/interaction_classification_MLP/MLP_results_by_configuration_{}epochs.csv".format(epochs)
with open(output_file, "w") as f:
    f.write(",".join(["nb_layers", "optimizer_class", "learning_rate", "batch_size", \
        "train_loss", "train_acc", "train_mean_acc", "train_negative_acc", "train_handShake_acc", "train_highFive_acc", "train_hug_acc", "train_kiss_acc", \
        "val_loss", "val_acc", "val_mean_acc", "val_negative_acc", "val_handShake_acc", "val_highFive_acc", "val_hug_acc", "val_kiss_acc",]) + "\n")
    for config in results:
        f.write(",".join(map(str, config + results[config])) + "\n")
