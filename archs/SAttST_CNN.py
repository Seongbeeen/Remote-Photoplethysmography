"""
This module contains the used architectures for pulse signal and pulse rate extracton
"""
import torch
import torch.nn as nn

# -------------------------------------------------------------------------------------------------------------------
# PhyNet network
# -------------------------------------------------------------------------------------------------------------------


class SAttST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(
                1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(
                1, 1, 1), stride=1, padding=(0, 0, 0))
        )

        self.encoder_attn = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(
                3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(
                1, 1, 1), stride=1, padding=(0, 0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.start(x)
        attn = self.loop1(attn)
        attn = self.encoder_attn(attn)
        x = self.start(x)
        x = self.loop1(x)
        x = self.encoder1(x)
        x = torch.mul(x, self.attn)
        x = self.encoder2(x)
        x = self.loop4(x)
        x = self.decoder(x)
        x = self.end(x)

        return x
