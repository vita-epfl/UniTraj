from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .fmae_dataset import FMAEDataset
from .EMP_dataset import EMPDataset
from .SMART_dataset import SMARTDataset
from .hpnet_dataset import HPNetDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
    'forecast': FMAEDataset,
    'MAE': FMAEDataset,
    'EMP': EMPDataset,
    'SMART': SMARTDataset,
    'hpnet': HPNetDataset,
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
