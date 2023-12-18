import os
import time
import logging
import logging.config
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_loss_curve(
    model_name, train_loss_history, valid_loss_history, title, output_path
):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    pd.Series(train_loss_history).plot(title="Loss", label="Train", ax=ax)
    pd.Series(valid_loss_history).plot(title="Loss", label="Validation", ax=ax)
    ax.title.set_text(model_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"loss curve saved at {output_path}.")
