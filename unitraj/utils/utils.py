import glob
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch


def find_latest_checkpoint(search_pattern):

    # List all files matching the pattern
    list_of_files = glob.glob(search_pattern, recursive=True)
    # Find the file with the latest modification time
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch Lightning based training.

    Args:
    seed_value (int): The seed value to be set for random number generators.
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed_value)

    # If using CUDA (PyTorch with GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    # Set the random seed for numpy (if using numpy in the project)
    np.random.seed(seed_value)

    # Set the random seed for Python's `random`
    random.seed(seed_value)

    # Set the seed for PyTorch Lightning's internal operations
    pl.seed_everything(seed_value, workers=True)
