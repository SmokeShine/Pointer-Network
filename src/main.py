#!/usr/bin/env python

import pandas as pd
import configparser
import logging
import logging.config
import numpy as np
import pandas as pd
from plots import plot_loss_curve
from scipy.spatial.distance import pdist, squareform
from math import sin, cos, sqrt, atan2, radians
import warnings
import time, datetime
import argparse
import warnings
from pyCombinatorial.algorithm import bellman_held_karp_exact_algorithm
from utils import train, evaluate, save_checkpoint
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import train, evaluate, save_checkpoint
import random
import mymodels
import datetime
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Subset
from torchvision.utils import save_image
from tqdm import tqdm
import math

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

torch.manual_seed(42)

try:
    logging.config.fileConfig(
        "logging.ini",
        disable_existing_loggers=False,
        defaults={
            "logfilename": datetime.datetime.now().strftime(
                "../logs/PointerNetwork_%H_%M_%d_%m.log"
            )
        },
    )
except:
    pass


class SequenceWithLabelDataset(Dataset):
    def __init__(self, input_cities=[], ninstances=None):
        self.input_cities = input_cities
        self.ninstances = ninstances
        self.list_of_instances = self.CreateData()
        if len(self.list_of_instances) == 0:
            raise ValueError("No Data Generated")

    def CreateData(self):
        locations_, route_, distance_ = [], [], []
        for i in tqdm(range(self.ninstances)):
            for input_city in tqdm(self.input_cities):
                locations = self.CreateInstance(city=input_city)
                route, distance = self.HeldKarpHeuristics(locations)
                route_.append(
                    [x - 1 for x in route[:-1]]
                )  # start index at 0, remove last index for returning to start
                distance_.append(distance)
                locations_.append(locations)
        return (locations_, route_, distance_)

    # https://stackoverflow.com/questions/74453574/how-to-distance-between-all-points-in-list
    def pairdistance(self, locations):
        def distance(x, y):
            return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

        dist = []
        for i in locations:
            for j in locations:
                dist.append(distance(i, j))

        return np.array(dist).reshape(len(locations), len(locations))

    def HeldKarpHeuristics(self, locations):
        distanceMatrix = self.pairdistance(locations)
        route, distance = bellman_held_karp_exact_algorithm(distanceMatrix)
        return route, distance

    def CreateInstance(self, city):
        locations = [(random.random(), random.random()) for _ in range(city)]
        locations[0] = (0, 0)
        return locations

    def __len__(self):
        return len(self.list_of_instances[0])

    def __getitem__(self, index):
        location = self.list_of_instances[0][index]
        sequence = self.list_of_instances[1][index]
        return torch.FloatTensor(location), torch.LongTensor(sequence)


# How to use beam search for output?


def train_model(model_name="PointerNetwork"):
    logger.info("Generating DataLoader")
    logger.info("Training Model")
    model = mymodels.PointerNetwork()
    logger.info(model)
    save_file = "PointerNetwork.pth"
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
    if MODEL_PATH != None:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded Checkpoint from {MODEL_PATH}")
    logger.info(model)
    train_dataset = SequenceWithLabelDataset(
        # input_cities=[CUSTOMERCOUNT, CUSTOMERCOUNT + 1, CUSTOMERCOUNT + 2],
        input_cities=[CUSTOMERCOUNT],
        ninstances=100000,  # 1 million instances
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        # collate_fn=collate_fn,
    )
    valid_dataset = SequenceWithLabelDataset(
        # input_cities=[CUSTOMERCOUNT, CUSTOMERCOUNT + 1, CUSTOMERCOUNT + 2],
        input_cities=[CUSTOMERCOUNT],
        ninstances=5,  # variable length batches
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # collate_fn=collate_fn,
    )
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)
    train_loss_history = []
    valid_loss_history = []
    best_validation_loss = float("inf")
    early_stopping_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}")
        train_loss = train(model, DEVICE, train_loader, criterion, optimizer, epoch)
        logger.info(f"Average Loss for epoch {epoch} is {train_loss}")
        train_loss_history.append(train_loss)

        valid_loss = evaluate(model, DEVICE, valid_loader, criterion, optimizer)
        valid_loss_history.append(valid_loss)
        is_best = best_validation_loss > valid_loss
        if epoch % EPOCH_SAVE_CHECKPOINT == 0:
            logger.info(f"Saving Checkpoint for {model_name} at epoch {epoch}")
            save_checkpoint(model, optimizer, save_file + "_" + str(epoch) + ".tar")
        if is_best:
            early_stopping_counter = 0
            logger.info(
                f"New Best Identified: \t Old Loss: {best_validation_loss}  vs New loss:\t{valid_loss} "
            )
            best_validation_loss = valid_loss
            torch.save(model, "./best_model.pth", _use_new_zipfile_serialization=False)
        else:
            logger.info("Loss didnot improve")
            early_stopping_counter += 1
        if early_stopping_counter >= PATIENCE:
            break
    # final checkpoint saved
    save_checkpoint(model, optimizer, save_file + ".tar")
    # Loading Best Model
    best_model = torch.load("./checkpoint_model.pth")
    logger.info(f"Train Losses:{train_loss_history}")
    logger.info(f"Validation Losses:{valid_loss_history}")
    logger.info(f"Plotting Charts")

    plot_loss_curve(
        model_name,
        train_loss_history,
        valid_loss_history,
        "Loss Curve",
        f"{PLOT_OUTPUT_PATH}loss_curves.jpg",
    )
    logger.info(f"Train Losses:{train_loss_history}")
    logger.info(f"Training Finished for {model_name}")


def predict_model(best_model):
    pred_dataset = SequenceWithLabelDataset(input_cities=[CUSTOMERCOUNT], ninstances=1)
    pred_loader = DataLoader(
        dataset=pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    # No need to calculate grad as it is forward pass only
    best_model.eval()
    with torch.no_grad():
        for counter, (input, target) in tqdm(enumerate(pred_loader)):
            # Model is in GPU
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            output = best_model(input)
            logger.info(f"Input:{input}")
            _, pointer = torch.max(output, axis=1)
            logger.info(f"Prediction:{pointer}")
            logger.info(f"Target:{target}")
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pointer Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--customers",
        default=15,
        type=int,
        choices=range(1, 100),
        help="Number of customers",
    )
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train Model"
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=128,
        help="Batch size for training the model",
    )
    parser.add_argument(
        "--num_workers", nargs="?", type=int, default=0, help="Number of Available CPUs"
    )
    parser.add_argument(
        "--num_epochs",
        nargs="?",
        type=int,
        default=10,
        help="Number of Epochs for training the model",
    )
    parser.add_argument(
        "--learning_rate",
        nargs="?",
        type=float,
        default=1.0,
        help="Learning Rate for the optimizer",
    )
    parser.add_argument(
        "--sgd_momentum",
        nargs="?",
        type=float,
        default=0.0,
        help="Momentum for the SGD Optimizer",
    )
    parser.add_argument(
        "--plot_output_path", default="./Plots_", help="Output path for Plot"
    )
    parser.add_argument("--model_path", help="Model Path to resume training")
    parser.add_argument(
        "--epoch_save_checkpoint",
        nargs="?",
        type=int,
        default=5,
        help="Epochs after which to save model checkpoint",
    )
    parser.add_argument(
        "--pred_model",
        default="./checkpoint_model.pth",
        help="Model for prediction; Default is checkpoint_model.pth; \
                            change to ./best_model.pth for 1 sample best model",
    )
    parser.add_argument(
        "--patience", nargs="?", type=int, default=5, help="Early stopping epoch count"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    CUSTOMERCOUNT = args.customers

    global BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, LEARNING_RATE, SGD_MOMENTUM, PRED_MODEL, PATIENCE
    __train__ = args.train
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    SGD_MOMENTUM = args.sgd_momentum
    PLOT_OUTPUT_PATH = args.plot_output_path
    EPOCH_SAVE_CHECKPOINT = args.epoch_save_checkpoint
    MODEL_PATH = args.model_path
    PATIENCE = args.patience
    PRED_MODEL = args.pred_model
    DEVICE = torch.device("mps")

    logger.info(f"Problem Size:{CUSTOMERCOUNT}")

    if __train__:
        logger.info("Training")
        train_model()
    else:
        logger.info("Prediction")
        best_model = torch.load(PRED_MODEL)
        logger.info(f"Using {PRED_MODEL} for prediction")
        predict_model(best_model)
        logger.info("Prediction Step Complete")
