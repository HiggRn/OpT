# Zhang Yuxiang, 2024/5/26
# 尝试利用百分位数这样的统计信息对M进行降维


# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入数据X:(D,N,M)，其中N表示期限，M表示Monte Carlo模拟次数，D为资产数（包含期权的模拟收益）
class OptionTransformer(nn.Module):
    def __init__(
        self, d_model, n_layers, n_head, n_assets, n_mcmc, dropout, n_features=100
    ) -> None:
        super().__init__()
        self.n_features = n_features  # 即是n，表示我们想从M次模拟中提取出多少特征（本模型使用百分位数）
        self.d_model = d_model  # 即是m
        self.n_head = n_head  # 即是h
        self.n_assets = n_assets  # 即是D
        self.n_mcmc = n_mcmc  # 即是M

        # 先归一化
        self.norm = nn.BatchNorm1d(n_features)

        # 然后Transformer: Encoder only，输出(N,h*n*D)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=n_features * (n_assets + 1),
            nhead=n_head,
            dropout=dropout,
            activation=F.gelu,  # TODO可能ReLU会更好？不知道
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=n_layers
        )

        # 最后swish+sigmoid/softplus
        self.price = nn.Sequential(
            nn.Linear(n_features * (n_assets + 1), d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

        self.delta = nn.Sequential(
            nn.Linear(n_features * (n_assets + 1), d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_assets),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, X, mask=None):
        # Zhang Yuxiang, 2024/5/26
        # 先对M维度上的数据排序，然后取百分位数
        # Zijie Gu, 2024/5/26
        # Modified using torch.quantile()
        q = torch.linspace(0, self.n_mcmc - 1, steps=self.n_features)
        q /= self.n_mcmc  # calculate quantiles
        X_quantiles = torch.quantile(X, q, dim=2, keepdim=False)  # (n,N,D)
        X_quantiles = torch.transpose(X_quantiles, 0, 1)  # (N,n,D)
        X_norm = self.norm(X_quantiles)
        X_input = torch.flatten(X_norm, 1, 2)  # (N,n*D)

        # 再transformer
        # TODO不知道是否需要mask
        # 暂时先弄一个mask吧
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(X_input.size()[0])
        X_hidden = self.transformer(X_input, mask)  # (N,m)

        # 然后分别通过线性层
        # TODO与某个线性衰减的成分线性组合
        price_hidden = self.price(X_hidden)  # (N)
        delta_hidden = self.delta(X_hidden)  # (N)

        price_hat = price_hidden[-1, :].squeeze()
        delta_hat = delta_hidden[-1, :].squeeze()
        return price_hat, delta_hat
