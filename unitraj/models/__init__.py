from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.forecast_mae.forecast import ForecastMAE
from unitraj.models.forecast_mae.MAE import MaskedAutoEncoder

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'forecast': ForecastMAE,
    'MAE': MaskedAutoEncoder,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
