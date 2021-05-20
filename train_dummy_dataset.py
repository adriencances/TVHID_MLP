import os
import sys
import numpy as np
import cv2
import glob
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter

from dataset_aux import FrameProcessor
from dataset import TVHIDPairs
from MLP import ClassificationMLP, InternetMLP
from accuracy import multi_class_accuracy


class DummyDataset(data.Dataset):
    def __init__(self, size=100, nb_classes=2):
        self.size = size
        self.nb_classes = nb_classes
        self.create_data()

    def create_data(self):
        ref_vectors = [torch.zeros(2048) for k in range(self.nb_classes)]
        for k in range(self.nb_classes):
            ref_vectors[k][k] = 1
        
        self.data = []
        for i in range(self.size):
            k = i%self.nb_classes
            self.data.append((ref_vectors[k][:1024], ref_vectors[k][1024:], k))

    def __getitem__(self, index):
        assert index < self.size
        return self.data[index]
    
    def __len__(self):
        return self.size


def train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    train_loss = 0
    train_acc = 0

    model.train()
    for batch_id, (features1, features2, target) in enumerate(dataloader_train):
        # Pass inputs to GPU
        features1 = features1.cuda()
        features2 = features2.cuda()
        target = target.cuda()

        # Pass inputs to the model
        out = model(features1, features2) # shape : bx5

        # Compute loss
        loss = loss_fn(out, target)
        train_loss += loss.item()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute probabilities, accuracy and predictions
        probs = F.softmax(out, dim=1)
        value, preds = accuracy_fn(probs, target)
        train_acc += value

        nb_batches += 1

    train_loss /= nb_batches
    train_acc /= nb_batches

    return train_loss, train_acc


def test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        model.eval()
        for batch_id, (features1, features2, target) in enumerate(dataloader_val):
            # Pass inputs to GPU
            features1 = features1.cuda()
            features2 = features2.cuda()
            target = target.cuda()

            # Pass inputs to the model
            out = model(features1, features2) # shape : bx5

            # Compute loss
            loss = loss_fn(out, target)
            val_loss += loss.item()

            # Compute accuracy and predictions
            probs = F.softmax(out, dim=1)
            value, preds = accuracy_fn(probs, target)
            val_acc += value

            nb_batches += 1

        val_loss /= nb_batches
        val_acc /= nb_batches

    return val_loss, val_acc


def load_checkpoint_state(model, optimizer, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def train_model(epochs, batch_size=8, lr=0.001, record=True, chkpt_delay=10,
        nb_layers=3, optimizer_class=optim.Adam):
    if record:
        # Tensorboard writer
        writer = SummaryWriter("runs/run_dummy_lr{}".format(lr))

    # Model
    model = ClassificationMLP(nb_layers=nb_layers)
    # model = InternetMLP()
    model.cuda()

    # Loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizer_class(
        model.parameters(),
        lr=lr,
        # momentum=0.9,
        weight_decay=0.000001
    )

    # Accuracy function
    accuracy_fn = multi_class_accuracy

    # Datasets
    dataset_train = DummyDataset(size=1000)
    dataset_val = DummyDataset(size=50)

    # Dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        # Train epoch
        train_loss, train_acc = train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test epoch
        val_loss, val_acc = test_epoch(dataloader_val, model, epoch, loss_fn, optimizer, accuracy_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print("{}\t{}\t{}".format(epoch, train_loss, train_acc))

        if record:
            # Write losses and accuracies to Tensorboard
            writer.add_scalar("training_loss", train_loss, global_step=epoch)
            writer.add_scalar("training_accuracy", train_acc, global_step=epoch)
            writer.add_scalar("validation_loss", val_loss, global_step=epoch)
            writer.add_scalar("validation_accuracy", val_acc, global_step=epoch)
    
    for i in range(10):
        f1, f2, t = dataset_train[i]
        f1 = f1.unsqueeze(0).cuda()
        f2 = f2.unsqueeze(0).cuda()
        out = model(f1, f2)
        probs = F.softmax(out, dim=1)
        print(out[0].tolist())
        print([round(e, 3) for e in probs[0].tolist()], t)

    return train_losses, train_accs


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    lr = float(sys.argv[2])

    record = False
    chkpt_delay = 1000

    print("Nb epochs : {}".format(epochs))
    print("Learning rate : {}".format(lr))
    train_model(epochs=epochs, lr=lr, record=False, chkpt_delay=chkpt_delay)
