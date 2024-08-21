import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # heads for Queries
    n_kv_heads: Optional[int] = None  # heads for Keys and Values'
    vocab_size: int = -1  # defined later
    multiple_of: int = 256
    fnn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # 旋转位置编码
    assert head_dim % 2 == 0, "dimension must be even"
    # theta 参数列表
    # len: (head_dim / 2), i \in [0, head_dim / 2 - 1]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,...,dim/2]
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)  # [theta_1, ....theta_{dim/2}]
    # Construct the positions (the m parameter)
    m = torch.arange(seq_len, device=device)  # shape: [seq_len]
    # 每一个位置 m 和 theta_i 的组合
    # shape: (seq_len, dim/2)
    freqs = torch.outer(m, theta).float()
    # 转化为复数形式计算 c = R * exp(i * m * theta)， where R = 1
    # (seq_len, dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    # as_complex：把最后一维看成复数形式，2变1
    # as_real: 把最后一维看成实数，1变2
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


def repeat_kv(x: torch.Tensor, n_reps: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_reps == 1:
        return x
    else:
        return (
            # (B, seq_len, H_kv_heads, 1, Head_dim) -> (B, seq_len, H_kv_heads * n_reps, Head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_reps, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_reps, head_dim)
        )


class RMSNorm(nn.Module):
    # RMSNorm implementation for RMSNorm.
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # gamma
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) * (B, seq_len, 1) = (B, seq_len, dim)
        # rsqrt = 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # multiply gamma.
        # (Dim) * (B, seq_len, dim) = (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before self-attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # 因为每次只传入最新一个的，所以要start_pos来表达这一个所处的位置
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class SelfAttention(nn.Module):
    # KV-Cache + MultiQueryAttention.
    def __init__(self, args: ModelArgs):
        super().__init__()

        # kv 的头数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # q 的头数
        self.n_heads_q = args.n_heads
        # GMQA 中每组的 q 的头数
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # 每个 q 头的维度
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, 1, dim)

        # 输入转化成 k, q, v
        # (B,1,dim) -> (B,1,H_q * Head_dim)
        xq = self.wq(x)
        # (B,1,dim) -> (B,1,H_kv * Head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B,1,H_q * Head_dim) -> (B,1,H_q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B,1,H_kv * Head_dim) -> (B,1,H_kv, Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Q, K 加 RPE
        xq = apply_rotary_embeddings(xq, freqs_complex, device=xq.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=xk.device)

        # Update KV-Cache
        self.cache_k[:batch_size, start_pos: start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos+seq_len] = xv

        # Use KV-Cache
        # K, V: (B, Seq_len_now, H_kv, Head_dim)
        keys = self.cache_k[:batch_size, 0: start_pos+seq_len]
        values = self.cache_v[:batch_size, 0: start_pos+seq_len]

        # Repeat the heads of k and v to reach the number of the heads of q
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # calculate attention
        # 每个头都要看整个序列，所以先换位置
        # (B, 1, H_q, Head_dim) -> (B, H_q, 1, Head_dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_len_now, H_q, Head_dim) -> (B, H_q, Seq_len_now, Head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_q, 1, Head_dim) @ (B, H_q, Head_dim, Seq_len_now) --> (B, H_q, 1, Seq_len_now)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_q, 1, Seq_len_now) @ (B, H_q, Seq_len_now, Head_dim) -> (B, H_q, 1, Head_dim)
        out = torch.matmul(scores, values)
        # (B, H_q, 1, Head_dim) -> (B, 1, H_q, Head_dim) -> (B, 1, dim)
        output = (out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output)  # (B, 1, dim) -> (B, 1, dim)


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

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # 因为有KV-Cache，只需要给最新的token
        # （B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1

        # (B, seq_len) -> (B, seq_len, Dim)
        h = self.tok_embeddings(tokens)  # 向量化

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
