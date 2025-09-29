import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

#from src.metrics import MR, minADE, minFDE
from .optim import WarmupCosLR
from utils.submission_av2 import SubmissionAv2

from .model_forecast import ModelForecast
from unitraj.models.base_model.base_model import BaseModel


class TrainerForecast(BaseModel):
    def __init__(
        self,
        config = None,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(TrainerForecast, self).__init__(config=config)

        ## for UniTraj integration
        if config is not None:
            dim = config["dim"]
            historical_steps = config["past_len"]
            future_steps = config["future_len"]
            encoder_depth = config["num_encoder_layers"]
            num_heads = config["tx_num_heads"]
            mlp_ratio = config["mlp_ratio"]
            qkv_bias = config["qkv_bias"]
            drop_path = config["drop_path"]
            epochs = config["max_epochs"]
            warmup_epochs = config["warmup_epochs"]
            lr = config["learning_rate"]
            weight_decay = config["weight_decay"]
            pretrained_weights = config["pretrained_weights"]

        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.net = ModelForecast(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
            historical_steps=historical_steps,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print("Loading pretrained weights from: ", pretrained_weights)

        #metrics = MetricCollection(
        #    {
        #        "minADE1": minADE(k=1),
        #        "minADE6": minADE(k=6),
        #        "minFDE1": minFDE(k=1),
        #        "minFDE6": minFDE(k=6),
        #        "MR": MR(),
        #    }
        #)
        #self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, data):
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy())
        return self.net(converted_data)

    def predict(self, data):
        with torch.no_grad():
            converted_data = self.convert_to_fmae_format(data['input_dict'].copy())
            out = self.net(converted_data)
        #predictions, prob = self.submission_handler.format_data(
        #    data, out["y_hat"], out["pi"], inference=True
        #)
        prediction = {}
        prediction['predicted_trajectory'] = out["y_hat"][..., :2]
        prediction['predicted_probability'] = torch.softmax(out["pi"].double(), dim=-1)
        return prediction

    def cal_loss(self, out, data):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.hparams.historical_steps:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = agent_reg_loss + agent_cls_loss + others_reg_loss

        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy()) ##TODO: maybe only do it once, not here and in forward?
        #fmae
        losses = self.cal_loss(out, converted_data)

        #unitraj
        prediction = {}
        prediction['predicted_trajectory'] = out["y_hat"][..., :2]
        prediction['predicted_probability'] = torch.softmax(out["pi"].double(), dim=-1)
        self.compute_official_evaluation(data, prediction)
        self.log_info(data, batch_idx,  prediction, status='train')
        # for k, v in losses.items():
        #     self.log(
        #         f"train/{k}",
        #         v,
        #         on_step=True,
        #         on_epoch=True,
        #         prog_bar=False,
        #         sync_dist=True,
        #     )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy()) 
        losses = self.cal_loss(out, converted_data)

        #unitraj
        prediction = {}
        prediction['predicted_trajectory'] = out["y_hat"][..., :2]
        prediction['predicted_probability'] = torch.softmax(out["pi"].double(), dim=-1)
        self.compute_official_evaluation(data, prediction)
        self.log_info(data, batch_idx,  prediction, status='val')

        #metrics = self.val_metrics(out, data["y"][:, 0])

        # self.log(
        #     "val/reg_loss",
        #     losses["reg_loss"],
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     sync_dist=True,
        # )
        # self.log_dict(
        #     metrics,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=1,
        #     sync_dist=True,
        # )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir#), filename=f"forecast_mae_{timestamp}"
        )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
