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
# from dataset import TVHIDPairs
from dataset_augmented import TVHIDPairs
from MLP import ClassificationMLP
from accuracy import multi_class_accuracy


class DummyLoss:
    def __init__(self, nb_classes=5):
        self.nb_classes = nb_classes
    def __call__(self, out, target):
        target = torch.clamp(target, max=self.nb_classes - 1)
        return F.cross_entropy(out, target)


def train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn):
    nb_batches = 0
    train_loss = 0
    train_acc = 0

    nb_true_predictions_by_class = [0 for i in range(5)]
    total_by_class = [0 for i in range(5)]

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

        # For accuracy by class
        for i in range(len(target)):
            k = target[i]
            total_by_class[k] += 1
            if preds[i] == target[i]:
                nb_true_predictions_by_class[k] += 1

        nb_batches += 1
    
    # Accuracy by class
    accuracies_by_class = [nb_true_predictions_by_class[k] / total_by_class[k] \
        for k in range(5)]
    train_mean_acc = sum(accuracies_by_class) / 5

    train_loss /= nb_batches
    train_acc /= nb_batches

    return train_loss, train_acc, train_mean_acc, accuracies_by_class


def test_epoch(dataloader_val, model, epoch, loss_fn, accuracy_fn):
    nb_batches = 0
    val_loss = 0
    val_acc = 0

    nb_true_predictions_by_class = [0 for i in range(5)]
    total_by_class = [0 for i in range(5)]

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

            # For accuracy by class
            for i in range(len(target)):
                k = target[i]
                total_by_class[k] += 1
                if preds[i] == target[i]:
                    nb_true_predictions_by_class[k] += 1

            nb_batches += 1
        
        # Accuracy by class
        accuracies_by_class = [nb_true_predictions_by_class[k] / total_by_class[k] \
            for k in range(5)]
        val_mean_acc = sum(accuracies_by_class) / 5

        val_loss /= nb_batches
        val_acc /= nb_batches

    return val_loss, val_acc, val_mean_acc, accuracies_by_class


def load_checkpoint_state(model, optimizer, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def train_model(epochs, batch_size=8, lr=0.001, record=True, do_chkpts=False, chkpt_delay=10, data_config="baseline", config_index=None,
        nb_layers=2, dropout_prob=0, optimizer_class=optim.Adam, weight_decay=0):
    if data_config not in ["baseline", "ii3d"]:
        print("Config argument must be 'baseline' or 'ii3d'")
        return

    if record:
        # Tensorboard writer
        writer = SummaryWriter("runs/run_{}_config{}".format(data_config, config_index))

    # Model
    model = ClassificationMLP(nb_layers=nb_layers, dropout_prob=dropout_prob)
    model.cuda()

    # Loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizer_class(
        model.parameters(),
        lr=lr,
        # momentum=0.9,
        weight_decay=weight_decay
    )

    # Accuracy function
    accuracy_fn = multi_class_accuracy

    # Datasets (AUGMENTATION ONLY FOR TRAINING SET)
    baseline = (data_config == "baseline")
    dataset_train = TVHIDPairs("train", baseline=baseline, augmented=True)
    dataset_val = TVHIDPairs("val", baseline=baseline, augmented=False)

    # Dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_losses = []
    train_accs = []
    train_mean_accs = []
    all_train_accs_by_classes = []

    val_losses = []
    val_accs = []
    val_mean_accs = []
    all_val_accs_by_classes = []

    for epoch in range(epochs):
        # Train epoch
        train_loss, train_acc, train_mean_acc, train_accs_by_classes = train_epoch(dataloader_train, model, epoch, loss_fn, optimizer, accuracy_fn)
        # print(train_loss, train_acc, train_mean_acc, train_accs_by_classes)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_mean_accs.append(train_mean_acc)
        all_train_accs_by_classes.append(train_accs_by_classes)

        # Test epoch
        val_loss, val_acc, val_mean_acc, val_accs_by_classes = test_epoch(dataloader_val, model, epoch, loss_fn, accuracy_fn)
        # print(val_loss, val_acc, val_mean_acc, val_accs_by_classes)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_mean_accs.append(val_mean_acc)
        all_val_accs_by_classes.append(val_accs_by_classes)

        # print("{}\t{}\t{}".format(epoch, train_loss, train_acc))

        if record:
            # Write losses and accuracies to Tensorboard
            writer.add_scalar("training_loss", train_loss, global_step=epoch)
            writer.add_scalar("training_accuracy", train_acc, global_step=epoch)
            writer.add_scalar("validation_loss", val_loss, global_step=epoch)
            writer.add_scalar("validation_accuracy", val_acc, global_step=epoch)

        # Save checkpoint
        if do_chkpts and epoch%chkpt_delay == chkpt_delay - 1:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            checkpoint_file = "checkpoints/checkpoint_{}_config{}_epoch{}.pt".format(
                data_config,
                config_index if config_index is not None else "",
                epoch)
            torch.save(state, checkpoint_file)
    
    # for i in range(10):
    #     f1, f2, t = dataset_train[i]
    #     f1 = f1.unsqueeze(0).cuda()
    #     f2 = f2.unsqueeze(0).cuda()
    #     out = model(f1, f2)
    #     probs = F.softmax(out, dim=1)
    #     print(out[0].tolist())
    #     print([round(e, 3) for e in probs[0].tolist()], t)
    
    # state = {
    #             'epoch': epoch,
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict()
    #         }
    # checkpoint_file = "checkpoints/checkpoint_{}layers_{}_lr{}_epoch{}.pt".format(nb_layers, data_config, lr, epoch)
    # torch.save(state, checkpoint_file)

    return train_losses, train_accs, train_mean_accs, all_train_accs_by_classes, val_losses, val_accs, val_mean_accs, all_val_accs_by_classes


def test_epoch_with_classes(dataloader_val, model, epoch, loss_fn, accuracy_fn, nb_classes=5):
    nb_batches = 0
    val_loss = 0
    val_acc = 0

    nb_true_predictions_by_class = [0 for i in range(nb_classes)]
    total_by_class = [0 for i in range(nb_classes)]

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

            # For accuracy by class
            for i in range(len(target)):
                k = min(target[i], nb_classes - 1)
                total_by_class[k] += 1
                if preds[i] == target[i]:
                    nb_true_predictions_by_class[k] += 1

            nb_batches += 1
        
        # Accuracy by class
        accuracies_by_class = [nb_true_predictions_by_class[k] / total_by_class[k] \
            for k in range(nb_classes)]
        val_mean_acc = sum(accuracies_by_class) / nb_classes

        val_loss /= nb_batches
        val_acc /= nb_batches

    return val_loss, val_acc, val_mean_acc, accuracies_by_class


def evaluate_untrained_model(batch_size=8, data_config="baseline", nb_layers=2, nb_classes=5):
    if data_config not in ["baseline", "ii3d"]:
        print("Config argument must be 'baseline' or 'ii3d'")
        return

    # Model
    model = ClassificationMLP(nb_layers=nb_layers)
    model.cuda()

    # Loss function
    loss_fn = DummyLoss(nb_classes=nb_classes)

    # Accuracy function
    accuracy_fn = multi_class_accuracy

    # Datasets (NO AUGMENTATION)
    baseline = (data_config == "baseline")
    dataset_train = TVHIDPairs("train", baseline=baseline, augmented=False)
    dataset_val = TVHIDPairs("val", baseline=baseline, augmented=False)

    # Dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    train_loss, train_acc, train_mean_acc, train_accs_by_classes = test_epoch_with_classes(
        dataloader_train, model, 0, loss_fn, accuracy_fn, nb_classes=nb_classes)
    val_loss, val_acc, val_mean_acc, val_accs_by_classes = test_epoch_with_classes(
        dataloader_val, model, 0, loss_fn, accuracy_fn, nb_classes=nb_classes)

    print("Training set")
    print("loss:\t {}".format(train_loss))
    print("accuracy:\t {}".format(train_acc))
    print("mean accuracy:\t {}".format(train_mean_acc))
    print("Validation set")
    print("loss:\t {}".format(val_loss))
    print("accuracy:\t {}".format(val_acc))
    print("mean accuracy:\t {}".format(val_mean_acc))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        for nb_classes in [2, 3, 4, 5]:
            print("NB_CLASSES={}".format(nb_classes))
            evaluate_untrained_model(nb_classes=nb_classes)
            print()
        sys.exit()

    epochs = int(sys.argv[1])
    lr = float(sys.argv[2])
    data_config = sys.argv[3]
    nb_layers = int(sys.argv[4])

    record = True
    chkpt_delay = 1000

    print("Nb epochs : {}".format(epochs))
    print("Learning rate : {}".format(lr))
    train_model(epochs=epochs, lr=lr, record=record, chkpt_delay=chkpt_delay, data_config=data_config, nb_layers=nb_layers)
