"""
2D-RoPE-STR model and positional-encoding variants.

A single `STRModel` implements the full CNN -> Transformer-encoder ->
autoregressive-decoder pipeline. The encoder's positional encoding is fully
pluggable via `pe_type`, so the ablation study compares strictly identical
architectures that differ only in how position is injected:

    none | 1d_sin | 1d_learn | 2d_sin | 2d_learn | 1drope | 2drope

RoPE variants ('1drope', '2drope') apply rotation *inside* the attention
(to queries and keys), which is the mathematically correct formulation.
Additive variants add the encoding to the tokens before the encoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# CNN backbone with configurable output stride
# ============================================================
def _disable_stride(layer):
    """Set the spatial stride of a ResNet stage's first block to 1."""
    block = layer[0]
    block.conv2.stride = (1, 1)            # 3x3 conv carries the stride (ResNet v1.5)
    if block.downsample is not None:
        block.downsample[0].stride = (1, 1)


def build_backbone(feature_stride: int = 8, pretrained: bool = True):
    """ResNet-50 truncated before avgpool/fc, with reduced total stride so the
    feature map keeps genuine 2D spatial extent (default /8: 32x128 -> 4x16)."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    net = resnet50(weights=weights)
    if feature_stride <= 16:
        _disable_stride(net.layer4)
    if feature_stride <= 8:
        _disable_stride(net.layer3)
    if feature_stride <= 4:
        _disable_stride(net.layer2)
    backbone = nn.Sequential(*list(net.children())[:-2])  # -> (B, 2048, H', W')
    return backbone, 2048


# ============================================================
# Rotary position embeddings
# ============================================================
class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device)).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # (N, dim/2) complex


class RotaryEmbedding2D(nn.Module):
    """Half of the head dimension is rotated by the row index, half by the
    column index. Optional learnable per-axis frequency scales (beta) and
    configurable per-axis frequency bases (theta). Defaults (base 10000 on both
    axes) reproduce the original buffer layout exactly, so older checkpoints
    that store only `inv_freq`/`beta_*` still load."""

    def __init__(self, head_dim, learnable_scale=False,
                 base_h=10000.0, base_w=10000.0, dim_split=0.5):
        super().__init__()
        self.base_h = float(base_h)
        self.base_w = float(base_w)
        # Per-axis real-dim counts, matching apply_2d_rope's split EXACTLY so the
        # frequency table length equals the number of complex pairs on each axis.
        d_h = int(head_dim * dim_split)
        if d_h % 2:
            d_h += 1
        d_w = head_dim - d_h

        def _inv(base, d):
            return 1.0 / (base ** (torch.arange(0, d, 2).float() / d))

        # Non-persistent: follow .to(device) but stay out of the state_dict, so
        # checkpoints remain compatible across any split/base choice.
        self.register_buffer("inv_freq_h", _inv(base_h, d_h), persistent=False)
        self.register_buffer("inv_freq_w", _inv(base_w, d_w), persistent=False)
        # Legacy buffer kept only so older checkpoints that stored `inv_freq`
        # (equal split, base 10000) still load strictly.
        self.register_buffer("inv_freq", _inv(10000.0, head_dim // 2))
        if learnable_scale:
            self.beta_h = nn.Parameter(torch.ones(1))
            self.beta_w = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("beta_h", torch.ones(1))
            self.register_buffer("beta_w", torch.ones(1))

    def forward(self, h_pos, w_pos):
        h_freqs = torch.einsum("i,j->ij", h_pos.float() * self.beta_h.float(), self.inv_freq_h).float()
        w_freqs = torch.einsum("i,j->ij", w_pos.float() * self.beta_w.float(), self.inv_freq_w).float()
        h_freqs = torch.polar(torch.ones_like(h_freqs), h_freqs)  # (H, d_h/2)
        w_freqs = torch.polar(torch.ones_like(w_freqs), w_freqs)  # (W, d_w/2)
        return h_freqs, w_freqs


def apply_1d_rope(x, freqs):
    """x: (B, heads, N, head_dim); freqs: (N, head_dim/2) complex."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs.unsqueeze(0).unsqueeze(0)).flatten(-2)
    return out.type_as(x)


def apply_2d_rope(x, h_freqs, w_freqs, H, W, dim_split=0.5):
    """x: (B, heads, N, head_dim) with N = H*W (row-major)."""
    B, nh, N, D = x.shape
    x = x.view(B, nh, H, W, D)
    d_h = int(D * dim_split)
    if d_h % 2:  # complex pairing needs an even split
        d_h += 1
    x_h, x_w = x[..., :d_h], x[..., d_h:]
    x_h = torch.view_as_complex(x_h.float().reshape(*x_h.shape[:-1], -1, 2))
    x_w = torch.view_as_complex(x_w.float().reshape(*x_w.shape[:-1], -1, 2))
    # freqs: (H, d_h/2) and (W, d_w/2) -> broadcast over (B, heads, H, W)
    hf = h_freqs[:, : x_h.shape[-1]].unsqueeze(0).unsqueeze(0).unsqueeze(3)  # (1,1,H,1,d_h/2)
    wf = w_freqs[:, : x_w.shape[-1]].unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1,1,1,W,d_w/2)
    x_h = torch.view_as_real(x_h * hf).flatten(-2)
    x_w = torch.view_as_real(x_w * wf).flatten(-2)
    out = torch.cat([x_h, x_w], dim=-1).view(B, nh, N, D)
    return out.type_as(x)


# ============================================================
# Additive (non-rotary) positional encodings
# ============================================================
class Additive1DSinPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, H, W):
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class Additive2DSinPE(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=128):
        super().__init__()
        d_half = d_model // 2

        def sinusoid(length, dim):
            pe = torch.zeros(length, dim)
            pos = torch.arange(0, length).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        self.register_buffer("pe_h", sinusoid(max_h, d_half))
        self.register_buffer("pe_w", sinusoid(max_w, d_half))

    def forward(self, x, H, W):
        B, N, D = x.shape
        d_half = D // 2
        x = x.view(B, H, W, D).clone()
        x[..., :d_half] = x[..., :d_half] + self.pe_h[:H].to(x.dtype).unsqueeze(1)
        x[..., d_half:] = x[..., d_half:] + self.pe_w[:W].to(x.dtype).unsqueeze(0)
        return x.view(B, N, D)


class Learnable1DPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x, H, W):
        return x + self.pe[:, : x.size(1)]


class Learnable2DPE(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=128):
        super().__init__()
        self.pe_h = nn.Parameter(torch.randn(1, max_h, d_model // 2) * 0.02)
        self.pe_w = nn.Parameter(torch.randn(1, max_w, d_model // 2) * 0.02)

    def forward(self, x, H, W):
        B, N, D = x.shape
        d_half = D // 2
        x = x.view(B, H, W, D).clone()
        x[..., :d_half] = x[..., :d_half] + self.pe_h[:, :H].unsqueeze(2)
        x[..., d_half:] = x[..., d_half:] + self.pe_w[:, :W].unsqueeze(1)
        return x.view(B, N, D)


ROPE_TYPES = {"1drope", "2drope"}
ADDITIVE_TYPES = {"1d_sin", "1d_learn", "2d_sin", "2d_learn"}


# ============================================================
# Encoder with RoPE-aware attention
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, rope=None):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        if rope is not None:
            q, k = rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope=None):
        x = x + self.dropout(self.attn(self.norm1(x), rope))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Decoder layer with optional RoPE in cross-attention keys
# ============================================================
class CrossRoPEDecoderLayer(nn.Module):
    """Pre-norm decoder layer: masked self-attention, then cross-attention to the
    encoder memory, then FFN. The cross-attention can rotate the memory *keys*
    by their position (1D flattened index or 2D row/col grid); queries (text
    tokens) are left unrotated, matching the paper's "RoPE in cross-attention
    keys" formulation. `cross_pe="none"` is the no-PE control."""

    def __init__(self, d_model, n_heads, d_ff, dropout, cross_pe, dim_split):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cross_pe = cross_pe
        self.dim_split = dim_split

        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if cross_pe == "2drope":
            self.rope = RotaryEmbedding2D(self.head_dim, dim_split=dim_split)
        elif cross_pe == "1drope":
            self.rope = RotaryEmbedding1D(self.head_dim)
        else:
            self.rope = None

    def _self_attn(self, x, tgt_mask):
        B, T, D = x.shape
        qkv = self.self_qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=tgt_mask)
        return self.self_out(out.transpose(1, 2).contiguous().view(B, T, D))

    def _cross_attn(self, x, memory, H, W):
        B, T, D = x.shape
        S = memory.size(1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, S, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)  # each (B, heads, S, head_dim)
        if self.cross_pe == "2drope":
            h_freqs, w_freqs = self.rope(torch.arange(H, device=x.device),
                                         torch.arange(W, device=x.device))
            k = apply_2d_rope(k, h_freqs, w_freqs, H, W, self.dim_split)
        elif self.cross_pe == "1drope":
            k = apply_1d_rope(k, self.rope(S, x.device))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.cross_out(out.transpose(1, 2).contiguous().view(B, T, D))

    def forward(self, x, memory, tgt_mask, H, W):
        x = x + self.dropout(self._self_attn(self.norm1(x), tgt_mask))
        x = x + self.dropout(self._cross_attn(self.norm2(x), memory, H, W))
        x = x + self.ffn(self.norm3(x))
        return x


# ============================================================
# Full STR model
# ============================================================
class STRModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 dropout=0.1, pe_type="2drope", learnable_freq=False,
                 dim_split=0.5, feature_stride=8, pretrained_backbone=True,
                 max_len=27, rope_base_h=10000.0, rope_base_w=10000.0,
                 cross_pe=None):
        super().__init__()
        self.pe_type = pe_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dim_split = dim_split
        self.cross_pe = cross_pe

        self.cnn, cnn_channels = build_backbone(feature_stride, pretrained_backbone)
        self.cnn_proj = nn.Linear(cnn_channels, d_model)

        # Positional encoding module(s)
        self.add_pe = None
        self.rope = None
        if pe_type in ADDITIVE_TYPES:
            self.add_pe = {
                "1d_sin": Additive1DSinPE, "1d_learn": Learnable1DPE,
                "2d_sin": Additive2DSinPE, "2d_learn": Learnable2DPE,
            }[pe_type](d_model)
        elif pe_type == "1drope":
            self.rope = RotaryEmbedding1D(self.head_dim)
        elif pe_type == "2drope":
            self.rope = RotaryEmbedding2D(self.head_dim, learnable_scale=learnable_freq,
                                          base_h=rope_base_h, base_w=rope_base_w,
                                          dim_split=dim_split)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder with sinusoidal PE on the target sequence.
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_pe = Additive1DSinPE(d_model, max_len=max_len + 4)
        if cross_pe is None:
            # Legacy path: standard Transformer decoder (no PE on cross-attn).
            dec_layer = nn.TransformerDecoderLayer(
                d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
            self.cross_decoder = None
        else:
            # Ablation path: custom layers that can rotate cross-attention keys.
            self.decoder = None
            self.cross_decoder = nn.ModuleList([
                CrossRoPEDecoderLayer(d_model, n_heads, d_ff, dropout, cross_pe, dim_split)
                for _ in range(num_decoder_layers)
            ])
        self.generator = nn.Linear(d_model, vocab_size)

    def _build_rope_fn(self, H, W, device):
        if self.pe_type == "2drope":
            h_pos = torch.arange(H, device=device)
            w_pos = torch.arange(W, device=device)
            h_freqs, w_freqs = self.rope(h_pos, w_pos)

            def fn(q, k):
                q = apply_2d_rope(q, h_freqs, w_freqs, H, W, self.dim_split)
                k = apply_2d_rope(k, h_freqs, w_freqs, H, W, self.dim_split)
                return q, k
            return fn
        if self.pe_type == "1drope":
            freqs = self.rope(H * W, device)

            def fn(q, k):
                return apply_1d_rope(q, freqs), apply_1d_rope(k, freqs)
            return fn
        return None

    def encode(self, images):
        feats = self.cnn(images)                      # (B, C, H, W)
        B, C, H, W = feats.shape
        self._mem_hw = (H, W)                          # needed by cross-attn RoPE
        feats = feats.flatten(2).transpose(1, 2)      # (B, H*W, C), row-major
        feats = self.cnn_proj(feats)
        if self.add_pe is not None:
            feats = self.add_pe(feats, H, W)
        rope_fn = self._build_rope_fn(H, W, images.device)
        x = feats
        for layer in self.encoder_layers:
            x = layer(x, rope_fn)
        return self.encoder_norm(x)

    def decode(self, tgt_input, memory):
        tgt = self.tgt_pe(self.tgt_embed(tgt_input), 0, 0)
        T = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_input.device)
        if self.cross_decoder is None:
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        else:
            H, W = self._mem_hw
            out = tgt
            for layer in self.cross_decoder:
                out = layer(out, memory, tgt_mask, H, W)
        return self.generator(out)

    def forward(self, images, tgt_input):
        memory = self.encode(images)
        return self.decode(tgt_input, memory)

    @torch.no_grad()
    def greedy_decode(self, images, sos_idx, eos_idx, max_len=27):
        memory = self.encode(images)
        B = images.size(0)
        seq = torch.full((B, 1), sos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(max_len):
            logits = self.decode(seq, memory)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(1) == eos_idx
            if finished.all():
                break
        return seq[:, 1:]  # drop the SOS token
