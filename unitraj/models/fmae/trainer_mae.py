import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .optim import WarmupCosLR

from .model_mae import ModelMAE
from unitraj.models.base_model.base_model import BaseModel

class TrainerMAE(BaseModel):
    def __init__(
        self,
        config = None, #new for UniTraj
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        actor_mask_ratio: float = 0.5,
        lane_mask_ratio: float = 0.5,
        epochs: int = 60,
        warmup_epochs: int = 10,
        lr: float = 1e-3,
        loss_weight=[1.0, 1.0, 0.35],
        weight_decay: float = 1e-4,
    ) -> None:
        super(TrainerMAE, self).__init__(config=config)

        ## for UniTraj integration
        if config is not None:
            dim = config["dim"]
            historical_steps = config["past_len"]
            future_steps = config["future_len"]
            encoder_depth = config["num_encoder_layers"]
            decoder_depth = config["num_decoder_layers"]
            num_heads = config["tx_num_heads"]
            mlp_ratio = config["mlp_ratio"]
            qkv_bias = config["qkv_bias"]
            drop_path = config["drop_path"]
            actor_mask_ratio = config["actor_mask_ratio"]
            lane_mask_ratio = config["lane_mask_ratio"]
            epochs = config["max_epochs"]
            warmup_epochs = config["warmup_epochs"]
            lr = config["learning_rate"]
            weight_decay = config["weight_decay"]
            loss_weight = config["loss_weight"]

        self.config = config
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

            

        self.net = ModelMAE(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            actor_mask_ratio=actor_mask_ratio,
            lane_mask_ratio=lane_mask_ratio,
            history_steps=historical_steps,
            future_steps=future_steps,
            loss_weight=loss_weight,
        )

    def forward(self, data):
        converted_data = self.convert_to_fmae_format(data['input_dict'].copy())
        return self.net(converted_data)

    def training_step(self, data, batch_idx):
        out = self(data)

        #unitraj
        prediction = {}
        prediction['predicted_trajectory'] = out["y_hat"][0, :, :1, :, :2]
        prediction['predicted_probability'] = out["pi"]
        self.log_info(data, batch_idx,  prediction, status='train')
        self.compute_official_evaluation(data, prediction)

        return out["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)

        #unitraj
        prediction = {}
        prediction['predicted_trajectory'] = out["y_hat"][0, :, :1, :, :2]
        prediction['predicted_probability'] = out["pi"]
        self.compute_official_evaluation(data, prediction)
        self.log_info(data, batch_idx,  prediction, status='val')

        #self.log("val_loss", out["loss"], on_epoch=True)
        return out["loss"]

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


    