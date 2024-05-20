# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入数据X:(2*D,N,M)，其中N表示期限，M表示Monte Carlo模拟次数，D为资产数
class OptionTransformer(nn.Module):
    def __init__(
        self, d_conv, d_model, n_layers, n_head, n_assets, n_mcmc, dropout
    ) -> None:
        super().__init__()
        self.d_conv = d_conv  # 即是n
        self.d_model = d_model  # 即是m
        self.n_head = n_head  # 即是h
        self.n_assets = n_assets  # 即是D
        self.n_mcmc = n_mcmc  # 即是M

        # 先经过一个1*1 Conv变为(n*D,N,M)
        # 然后经过一个池化层变为(n*D,N,1)，压为(n*D,N)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * n_assets, out_channels=d_conv * n_assets, kernel_size=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=(1, n_mcmc), padding=1, dilation=1),
        )

        # 然后Transformer: Decoder only，输出(N,h*m)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            activation=F.gelu,  # TODO可能ReLU会更好？不知道
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer=decoder_layers, num_layers=n_layers
        )

        # 最后swish+sigmoid/softplus
        self.price = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

        self.delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, mask=None):
        # 先卷积+池化
        X_conv = self.conv(X)  # (n*D,N,1)
        X_input = torch.transpose(X_conv.squeeze(2), 0, 1)  # (N,n*D)

        # 再transformer
        # TODO不知道是否需要mask
        # 暂时先弄一个mask吧
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(self.d_model)
        X_hidden = self.transformer(X_input, mask)  # (N,m)

        # 然后分别通过线性层
        # TODO与某个线性衰减的成分线性组合
        price_hat = self.price(X_hidden)  # (N)
        delta_hat = self.delta(X_hidden)  # (N)

        # TODO输出可能还得想着整理一下
        return price_hat, delta_hat
