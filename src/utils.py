import os
import time
import logging
import logging.config
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Boiler Plate Code From BD4H and DL Class for recording metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=50):
    model.train()
    losses = AverageMeter()
    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        assert not np.isnan(loss.item()), "Model diverged with loss = NaN"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        losses.update(loss.item(), target.size(0))
        if i % print_freq == 0:
            logger.info(
                f"Epoch: {epoch} \t iteration: {i} \t Training Loss Current:{losses.val:.4f} Average:({losses.avg:.4f})"
            )

    return losses.avg


def evaluate(model, device, data_loader, criterion, optimizer, print_freq=10):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), target.size(0))
            if i % print_freq == 0:
                logger.info(
                    f"Validation Loss Current:{losses.val:.4f} Average:({losses.avg:.4f})"
                )
    return losses.avg


def save_checkpoint(model, optimizer, path):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, path)
    torch.save(model, "./checkpoint_model.pth", _use_new_zipfile_serialization=False)
    logger.info(f"checkpoint saved at {path}")
