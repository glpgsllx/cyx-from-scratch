import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_heads: int = 32 # heads for Queries
    n_kv_heads: Optional[int] = None # heads for Keys and Values'
    vocab_size: int = -1 # defined later
    multiple_of: int = 256
    fnn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str=None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # 旋转位置编码
    assert head_dim % 2 == 0, "dimension must be even"
    # theta 参数列表
    # len: (head_dim / 2), i \in [0, head_dim / 2 - 1]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,...,dim/2]
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)  # [theta_1, ....theta_{dim/2}]
    # Construct the positions (the m parameter)
    m = torch.arange(seq_len, device=device) # shape: [seq_len]
    # 每一个位置 m 和 theta_i 的组合
    # shape: (seq_len, dim/2)
    freqs = torch.outer(m, theta).float()
    # 转化为复数形式计算 c = R * exp(i * m * theta)， where R = 1
    # (seq_len, dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    ## as_complex：把最后一维看成复数形式，2变1
    ## as_real: 把最后一维看成实数，1变2
    # first step: (B, seq_len, H, dim) -> (B, seq_len, H, dim / 2, 2) -> (B, seq_len, H, dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, dim / 2) -> (1, seq_len, 1, dim / 2)
    freqs_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, dim / 2) * (1, seq_len, 1, dim/2) -> (B, seq_len, H, dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, dim / 2) -> (B, seq_len, H, dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, dim/2, 2) -> (B, seq_len, H, dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # 因为有KV-Cache，只需要给最新的token
        # （B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1

        # (B, seq_len) -> (B, seq_len, Dim)
        h = self.tok_embeddings(tokens) # 向量化

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output





