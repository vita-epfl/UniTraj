from .base_dataset import BaseDataset


class AutoBotDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
