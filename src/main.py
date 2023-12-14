#!/usr/bin/env python

import pandas as pd
import configparser
import logging
import logging.config
import numpy as np
import pandas as pd
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


def HeldKarpAlgorithm(distanceMatrix=None):
    distanceMatrix = pd.read_csv(
        "https://github.com/Valdecy/Datasets/raw/master/Combinatorial/TSP-01-Distance%20Matrix.txt",
        sep="\t",
    )
    distanceMatrix = distanceMatrix.values
    parameters = {"verbose": True}
    route, distance = bellman_held_karp_exact_algorithm(distance_matrix, **parameters)
    import pdb

    pdb.set_trace()
    return route, distance


# def collate_fn(batch):
#     location, sequence = zip(*batch)
#     import pdb

#     pdb.set_trace()


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
                route_.append([x - 1 for x in route])  # start index at 0
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
        ninstances=BATCH_SIZE,  # variable length batches
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # collate_fn=collate_fn,
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)
    train_loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}")
        train_loss = train(model, DEVICE, train_loader, criterion, optimizer, epoch)
        logger.info(f"Average Loss for epoch {epoch} is {train_loss}")
        train_loss_history.append(train_loss)
        if epoch % EPOCH_SAVE_CHECKPOINT == 0:
            logger.info(f"Saving Checkpoint for {model_name} at epoch {epoch}")
            save_checkpoint(model, optimizer, save_file + "_" + str(epoch) + ".tar")
    # final checkpoint saved
    save_checkpoint(model, optimizer, save_file + ".tar")
    # Loading Best Model
    best_model = torch.load("./checkpoint_model.pth")

    logger.info(f"Plotting Charts")
    logger.info(f"Train Losses:{train_loss_history}")
    logger.info(f"Training Finished for {model_name}")


def predict_model(best_model, images_file, labels_file, pixel_classes):
    images_dataset = SequenceWithLabelDataset(
        images_file,
        labels_file,
        num_categories=len(pixel_classes),
        pixel_classes=pixel_classes,
    )
    pred_loader = DataLoader(
        dataset=images_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    # No need to calculate grad as it is forward pass only
    best_model.eval()
    with torch.no_grad():
        for counter, (input, target, img_name) in tqdm(enumerate(pred_loader)):
            # Model is in GPU
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            # which pixel belongs to which object, etc.
            # assign a class to each pixel of the image.
            output = best_model(input)
            # Output is 32 classes and we need to collapse back to 1
            # import pdb;pdb.set_trace()
            expected_width = output.shape[2]
            expected_height = output.shape[3]
            temp_image = torch.zeros((3, expected_width, expected_height))
            logging.info(f"Image is {img_name}")
            torch_pixel_classes = torch.from_numpy(pixel_classes)
            for i in range(expected_width):
                for j in range(expected_height):
                    temp_image[:, i, j] = torch_pixel_classes[
                        torch.argmax(output[0, :, i, j])
                    ]
            # import pdb;pdb.set_trace()
            # https://discuss.pytorch.org/t/convert-float-image-array-to-int-in-pil-via-image-fromarray/82167/4
            temp_image = temp_image.permute(1, 2, 0).numpy().astype(np.uint8)
            save_image(input, f"./predictions/actual_{counter}.png")
            transforms.ToPILImage()(temp_image).save(
                f"./predictions/pred_{counter}_{img_name}.png"
            )
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
        "--train", action="store_true", default=True, help="Train Model"
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
        default=0.01,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    CUSTOMERCOUNT = args.customers

    global BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, LEARNING_RATE, SGD_MOMENTUM, PRED_MODEL
    __train__ = args.train
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    SGD_MOMENTUM = args.sgd_momentum
    PLOT_OUTPUT_PATH = args.plot_output_path
    EPOCH_SAVE_CHECKPOINT = args.epoch_save_checkpoint
    MODEL_PATH = args.model_path

    PRED_MODEL = args.pred_model
    DEVICE = torch.device("mps")

    logger.info(f"Problem Size:{CUSTOMERCOUNT}")

    if __train__:
        logger.info("Training")
        train_model()
    else:
        logger.info("Training")
        predict_model(best_model)
        logger.info("Prediction Step Complete")
