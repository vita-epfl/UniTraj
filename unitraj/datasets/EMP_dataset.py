import numpy as np
from .base_dataset import BaseDataset
import torch


class EMPDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)

