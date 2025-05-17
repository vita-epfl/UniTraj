from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .fmae_dataset import FMAEDataset
from .EMP_dataset import EMPDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
    'forecast': FMAEDataset,
    'MAE': FMAEDataset,
    'EMP': EMPDataset
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
