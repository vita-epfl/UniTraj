from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.fmae.trainer_mae import TrainerMAE
from unitraj.models.fmae.trainer_forecast import TrainerForecast
from unitraj.models.emp.trainer_forecast import TrainerEMP

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'MAE': TrainerMAE,
    'forecast': TrainerForecast,
    'EMP': TrainerEMP,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
