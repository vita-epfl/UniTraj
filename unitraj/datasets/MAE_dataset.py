from .base_dataset import BaseDataset



class MAEDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)