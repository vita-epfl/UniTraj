import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metrics import AvgMinADE, AvgMinFDE, ActorMR
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2_multiagent import SubmissionAv2MultiAgent

from .model_forecast_ma import ModelForecastMultiAgent


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.2,
        num_modes=6,
        use_cls_token=False,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        submission_handler=SubmissionAv2MultiAgent(),
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["submission_handler"])
        self.num_modes = num_modes

        self.net = ModelForecastMultiAgent(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
            num_modes=num_modes,
            use_cls_token=use_cls_token,
        )

        metrics = MetricCollection([AvgMinADE(), AvgMinFDE(), ActorMR()])
        self.val_metrics = metrics.clone(prefix="val_")
        self.submission_handler = submission_handler

    def forward(self, data):
        return self.net(data)

    def cal_loss(self, outputs, data):
        y_hat, pi = outputs["y_hat"], outputs["pi"]

        x_scored, y, y_padding_mask = (
            data["x_scored"],
            data["y"],
            data["x_padding_mask"][..., 50:],
        )

        # only consider scored agents
        valid_mask = ~y_padding_mask
        valid_mask[~x_scored] = False  # [b,n,t]
        valid_mask = valid_mask.unsqueeze(1).float()  # [b,1,n,t]

        scene_avg_ade = (
            torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1) * valid_mask
        ).sum(dim=(-1, -2)) / valid_mask.sum(dim=(-1, -2))
        best_mode = torch.argmin(scene_avg_ade, dim=-1)

        reg_mask = ~y_padding_mask  # [b,n,t]
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        reg_loss = F.smooth_l1_loss(y_hat_best[reg_mask], y[reg_mask])
        cls_loss = F.cross_entropy(pi, best_mode.detach())

        loss = reg_loss + cls_loss
        out = {
            "loss": loss,
            "reg_loss": reg_loss.item(),
            "cls_loss": cls_loss.item(),
        }

        return out

    def training_step(self, data, batch_idx):
        outputs = self(data)
        res = self.cal_loss(outputs, data)

        for k, v in res.items():
            if k.endswith("loss"):
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=outputs["y_hat"].shape[0],
                )

        return res["loss"]

    def validation_step(self, data, batch_idx):
        outputs = self(data)
        res = self.cal_loss(outputs, data)
        metrics = self.val_metrics(outputs, data["y"], data["x_scored"])

        for k, v in res.items():
            if k.endswith("loss"):
                self.log(
                    f"val/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=outputs["y_hat"].shape[0],
                )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, data, batch_idx) -> None:
        outputs = self(data)
        y_hat, pi = outputs["y_hat"], outputs["pi"]

        bs, k, n, t, _ = y_hat.shape
        centers = data["x_centers"].view(bs, 1, n, 1, 2)
        y_hat += centers

        self.submission_handler.format_data(data, y_hat, pi)

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
