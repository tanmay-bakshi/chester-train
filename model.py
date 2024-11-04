from pathlib import Path

import lightning
import lightning.pytorch
import lightning.pytorch.callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader

from attention import Attention, SinCosPosEmbed
from convnext import ConvNeXtBlock, RMSNorm2d
from dataset import ChessDatasetClient


class AlphaZeroChessNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 119,
        channels: int = 128,
        conv_blocks: int = 7,
        attn_blocks: int = 10,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            RMSNorm2d(channels, eps=1e-6),
            *[ConvNeXtBlock(channels) for _ in range(conv_blocks)]
        )

        self.pos_embed = nn.Parameter(SinCosPosEmbed(dim=channels)(8, 8)[None], requires_grad=False)
        self.pos_embed_proj = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(self.pos_embed_proj.weight)
        nn.init.constant_(self.pos_embed_proj.bias, 0)

        self.value_token = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.value_token)

        self.attn_blocks = nn.Sequential(
            *[
                Attention(
                    dim=channels,
                    num_query_heads=32,
                    num_key_value_heads=8,
                    mlp_ratio=4,
                )
                for _ in range(attn_blocks)
            ]
        )

        self.policy_head = nn.Linear(channels, 73)
        self.value_head = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_blocks(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        pos_embed = self.pos_embed_proj(self.pos_embed)
        x = x + pos_embed
        x = torch.cat((self.value_token.expand(x.size(0), -1, -1), x), dim=1)

        x = self.attn_blocks(x)
        x_value = x[:, 0]
        x_policy = x[:, 1:]

        policy = self.policy_head(x_policy)
        policy = rearrange(policy, "b (h w) c -> b (c h w)", h=8, w=8)

        value = self.value_head(x_value)

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
