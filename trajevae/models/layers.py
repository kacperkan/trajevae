import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import trajevae.utils.general as general_utils


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DCTLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_components"]
    in_features: int
    out_features: int
    num_components: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_components: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_components = num_components

        self.weight = nn.Parameter(
            torch.empty(num_components, in_features, out_features),
            requires_grad=True,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty((num_components, 1, out_features))
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.baddbmm(self.bias, x, self.weight)
        return torch.bmm(x, self.weight)


class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.layers(x) + self.skip(x)))


class TrajectoryVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        dct_components: int,
        deterministic: bool = False,
        dropout: float = 0.0,
        max_len: int = 512,
        transformer_layers: int = 2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.max_len = max_len
        self.transformer_layers = transformer_layers
        self.input_dim = input_dim
        self.deterministic = deterministic
        self.dct_components = dct_components

        self.initial = nn.Linear(input_dim, output_dim)
        self.pose_encoder = PositionalEncoding(
            output_dim, self.dropout, self.max_len
        )

        # pre dct
        self.pre_dct_encoder = create_transformer_encoder_layer(
            output_dim, self.num_heads, transformer_layers, self.dropout
        )

        # post dct
        self.post_dct_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.mu = nn.Linear(output_dim, output_dim)
        self.logvar = (
            nn.Linear(output_dim, output_dim)
            if not deterministic
            else nn.Identity()
        )

    def reparametrize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        e = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return e * std + mu

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        out = {"mu": mu, "z": z, "logvar": logvar}
        return out

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        latent = self.pre_dct_encoder(self.pose_encoder(self.initial(x)))
        latent = general_utils.dct(latent[: self.dct_components], dim=0)
        latent = self.post_dct_encoder(latent)

        mu = self.mu(latent)
        if not self.deterministic or self.training:
            logvar = self.logvar(latent)
            out_latent = self.reparametrize(mu, logvar)
            return out_latent, mu, logvar
        return mu, mu, torch.zeros_like(mu)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        return next(self.parameters()).dtype

    def sample(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            (batch_size, self.latent_dim), dtype=self.dtype, device=self.device
        )

    def sample_trajectory(self, batch_size: int) -> torch.Tensor:
        latent = self.sample(batch_size)
        return self.deocder(latent)


def generate_square_subsequent_mask(
    sz: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    mask = (
        torch.triu(torch.ones(sz, sz, device=device, dtype=dtype)) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_transformer_encoder_layer(
    features: int,
    num_heads: int,
    num_transformer_layers: int,
    dropout: float = 0.0,
) -> nn.TransformerEncoder:
    single_layer = nn.TransformerEncoderLayer(
        features,
        num_heads,
        dim_feedforward=features,
        dropout=dropout,
    )
    model = nn.TransformerEncoder(
        single_layer,
        num_transformer_layers,
        norm=nn.LayerNorm(features),
    )
    return model


def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation="tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


class RNN(nn.Module):
    def __init__(self, input_dim, out_dim, cell_type="lstm", bi_dir=False):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = "batch"
        rnn_cls = nn.LSTMCell if cell_type == "lstm" else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == "step":
            self.hx = (
                torch.zeros((batch_size, self.rnn_f.hidden_size))
                if hx is None
                else hx
            )
            if self.cell_type == "lstm":
                self.cx = (
                    torch.zeros((batch_size, self.rnn_f.hidden_size))
                    if cx is None
                    else cx
                )

    def forward(self, x):
        if self.mode == "step":
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == "lstm":
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = torch.zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == "lstm":
            cx = torch.zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == "lstm":
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out
