from pathlib import Path

import lightning
import lightning.pytorch
import lightning.pytorch.callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from convnext import ConvNeXtBlock, RMSNorm2d
from dataset import ChessDatasetClient


class AlphaZeroChessNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 119,
        residual_blocks: int = 30,
        channels: int = 128,
    ):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            RMSNorm2d(channels, eps=1e-6),
        )

        self.residual_layers = nn.Sequential(
            *[ConvNeXtBlock(channels) for _ in range(residual_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 73, kernel_size=1),
            nn.Flatten(),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.input_conv(x)
        x = self.residual_layers(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class AlphaZeroChessLitModule(lightning.LightningModule):
    def __init__(self, model: AlphaZeroChessNetwork):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        # x: (N, 119, 8, 8), float32
        # policy_y_mask: (N, 4672), bool (True = illegal / masked, False = legal / not masked)
        # policy_y: (N,), long (0-4671)
        # value_y: (N,), long (0, 1, 2)
        x, policy_y_mask, policy_y, value_y = batch

        # policy: (N, 4672), float32 (raw network output)
        # value: (N,), float32 (raw network output)
        policy, value = self.model(x)

        policy = policy.masked_fill(policy_y_mask, float("-inf"))
        policy_loss = F.cross_entropy(policy, policy_y, reduction="none")
        with torch.no_grad():
            win_mask = (value_y != 2).float()
        policy_loss = policy_loss * win_mask
        policy_loss = policy_loss.sum() / win_mask.sum()

        value_loss = F.cross_entropy(value, value_y)

        self.log(
            "policy_loss",
            policy_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "value_loss",
            value_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return policy_loss + value_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.001)


def main() -> None:
    torch.set_float32_matmul_precision("medium")

    dataset = ChessDatasetClient(
        server_binary=Path("/home/tanmay/chester/target/release/slchess"),
        data_folder=Path("/home/tanmay/chess-ntp/pgns"),
        threads=12,
        queue_size=4096,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2048,
        num_workers=1,
        # shuffle=True,
    )

    model = AlphaZeroChessNetwork()
    lit_module = AlphaZeroChessLitModule(model)

    trainer = lightning.Trainer(
        max_steps=15000,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision="bf16-mixed",
    )
    trainer.fit(lit_module, dataloader)


if __name__ == "__main__":
    main()
