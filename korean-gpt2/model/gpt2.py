"""
GPT-2 Model Implementation from Scratch in PyTorch.

This module implements the GPT-2 architecture following the original paper:
"Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

Architecture follows GPT-2 Small by default:
  - 12 layers, 12 heads, 768 embedding dim
  - Context window of 1024 tokens
  - GELU activation, pre-LayerNorm style
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model."""

    vocab_size: int = 50257  # Will be overridden by Korean tokenizer
    max_seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    bias: bool = True  # True for GPT-2 faithfulness, False for modern practice


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Implements scaled dot-product attention with a causal mask so that
    each token can only attend to preceding tokens (and itself).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix
        # Registered as buffer so it moves to GPU with the model but isn't a parameter
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.size()

        # Compute Q, K, V in one shot, then split
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Optimized scaled dot-product attention using PyTorch 2.0+ kernels
        # This automatically uses FlashAttention or optimized MPS/CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None, # Not needed if is_causal=True
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=True
        )

        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Two linear transformations with a GELU activation in between,
    following the GPT-2 convention.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")  # GPT-2 uses tanh-approx GELU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single GPT-2 Transformer block.

    Uses pre-LayerNorm (LN before attention and FFN), which is the
    GPT-2 convention and improves training stability.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    GPT-2 Language Model.

    Composed of:
      - Token embedding + positional embedding
      - N Transformer blocks
      - Final LayerNorm
      - Linear head (tied to token embeddings)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings
                wte=nn.Embedding(config.vocab_size, config.d_model),
                # Positional embeddings (learned, not sinusoidal)
                wpe=nn.Embedding(config.max_seq_len, config.d_model),
                # Dropout after embeddings
                drop=nn.Dropout(config.dropout),
                # Transformer blocks
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layers)]
                ),
                # Final layer norm
                ln_f=nn.LayerNorm(config.d_model, bias=config.bias),
            )
        )

        # Language model head — weight-tied with token embeddings
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resizes input token embeddings and the output head.
        Used when adding special tokens to the tokenizer.
        """
        old_embeddings = self.transformer.wte
        new_embeddings = nn.Embedding(new_num_tokens, self.config.d_model)
        
        # Initialize new embeddings
        nn.init.normal_(new_embeddings.weight, mean=0.0, std=0.02)
        
        # Copy old weights
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        # Update model
        self.transformer.wte = new_embeddings
        self.lm_head = nn.Linear(self.config.d_model, new_num_tokens, bias=False)
        self.lm_head.weight = self.transformer.wte.weight # Re-tie weights
        self.config.vocab_size = new_num_tokens
        
        print(f"🔄 Resized token embeddings to {new_num_tokens}")

        print(f"🔄 Resized token embeddings to {new_num_tokens}")

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        # Subtract weight-tied params (counted once in wte, once in lm_head)
        # We don't subtract here to see the total including tied head
        print(f"GPT-2 model resized: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            idx: Token indices, shape (batch, seq_len)
            targets: Optional target indices for computing loss, shape (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)

        # Embed tokens and positions
        tok_emb = self.transformer.wte(idx)  # (B, T, d_model)
        pos_emb = self.transformer.wpe(pos)  # (T, d_model)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Compute logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are given
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx: Conditioning sequence of token indices, shape (batch, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, < 1.0 = sharper)
            top_k: If set, only sample from the top k most likely tokens

        Returns:
            Extended sequence of token indices, shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len :]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for the last position and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running GPT-2 sanity check...")

    config = GPT2Config(vocab_size=32000, n_layers=4, n_heads=4, d_model=256, d_ff=1024)
    model = GPT2(config)

    # Dummy input: batch of 2, sequence length 64
    dummy_ids = torch.randint(0, config.vocab_size, (2, 64))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 64))

    logits, loss = model(dummy_ids, targets=dummy_targets)
    print(f"  Input shape:  {dummy_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss:         {loss.item():.4f}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.8, top_k=50)
    print(f"  Generated shape: {generated.shape}")

    print("✅ GPT-2 sanity check passed!")
