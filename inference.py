#!/usr/bin/env python
"""
Inference script for modded-nanogpt on MPS/CPU.
Replaces Flash Attention 3 with PyTorch's scaled_dot_product_attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import tiktoken


# Model configuration (matches train_gpt.py)
@dataclass
class ModelConfig:
    vocab_size: int = 50304  # padded to multiple of 128
    num_layers: int = 11
    num_heads: int = 6
    head_dim: int = 128
    model_dim: int = 768
    block_size: int = 128


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMS normalization."""
    return F.rms_norm(x, (x.size(-1),))


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class Rotary(nn.Module):
    """Simplified rotary embeddings for inference."""
    def __init__(self, head_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Half-truncated RoPE
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4, dtype=torch.float32)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim // 2)])

        t = torch.arange(2 * max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)

        self.register_buffer('cos', theta.cos())
        self.register_buffer('sin', theta.sin())
        self.sin[..., 1::2] *= -1
        self.attn_scale = 0.1  # from Yarn class

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embeddings. x: (B, T, H, D)"""
        T = x.size(-3)
        cos = self.cos[offset:offset + T].unsqueeze(0).unsqueeze(-2)
        sin = self.sin[offset:offset + T].unsqueeze(0).unsqueeze(-2)

        x_flip = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
        return cos * x + sin * x_flip


class CausalSelfAttention(nn.Module):
    """Self-attention using PyTorch SDPA instead of Flash Attention."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dim = config.model_dim

    def forward(
        self,
        x: torch.Tensor,
        qkvo_w: torch.Tensor,
        rotary: Rotary,
        sa_lambdas: torch.Tensor,
        attn_gate_w: Optional[torch.Tensor] = None,
        ve: Optional[torch.Tensor] = None,
        ve_gate_w: Optional[torch.Tensor] = None,
        key_offset: bool = False,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        B, T, _ = x.shape

        # Project to Q, K, V
        qkv = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x))
        qkv = qkv.view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)

        # QK normalization
        q, k = norm(q), norm(k)

        # Rotary embeddings
        offset = kv_cache[0].size(1) if kv_cache is not None else 0
        q = rotary(q, offset)
        k = rotary(k, offset)

        # Key offset (shift keys forward for stationary head dims)
        if key_offset and T > 1:
            k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:].clone()

        # Value embeddings
        if ve is not None and ve_gate_w is not None:
            ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w.type_as(x)))
            ve_gate_out = ve_gate_out.view(B, T, self.num_heads, 1)
            v = v + ve_gate_out * ve.view(B, T, self.num_heads, self.head_dim).type_as(v)

        # KV cache for generation
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
        new_kv_cache = (k, v)

        # Reshape for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(kv_cache is None),  # only causal when no cache
            scale=rotary.attn_scale
        )

        # Reshape back: (B, T, H, D)
        y = y.transpose(1, 2)

        # Attention gating
        if attn_gate_w is not None:
            attn_gate = torch.sigmoid(F.linear(x[..., :12], attn_gate_w.type_as(x)))
            y = y * attn_gate.view(B, T, self.num_heads, 1)

        # Output projection
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))

        return y, new_kv_cache


class MLP(nn.Module):
    """MLP with ReLU^2 activation."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, c_fc: torch.Tensor, c_proj: torch.Tensor) -> torch.Tensor:
        # ReLU^2 activation
        h = F.relu(F.linear(x, c_fc.type_as(x)))
        h = h * h  # square
        return F.linear(h, c_proj.T.type_as(x))


class Block(nn.Module):
    """Transformer block."""
    def __init__(self, config: ModelConfig, has_attn: bool, has_mlp: bool):
        super().__init__()
        self.attn = CausalSelfAttention(config) if has_attn else None
        self.mlp = MLP() if has_mlp else None

    def forward(
        self,
        x: torch.Tensor,
        qkvo_w: Optional[torch.Tensor],
        c_fc: Optional[torch.Tensor],
        c_proj: Optional[torch.Tensor],
        rotary: Rotary,
        sa_lambdas: torch.Tensor,
        attn_gate_w: Optional[torch.Tensor] = None,
        ve: Optional[torch.Tensor] = None,
        ve_gate_w: Optional[torch.Tensor] = None,
        key_offset: bool = False,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        new_kv_cache = None

        if self.attn is not None:
            attn_out, new_kv_cache = self.attn(
                norm(x), qkvo_w, rotary, sa_lambdas,
                attn_gate_w, ve, ve_gate_w, key_offset, kv_cache
            )
            x = x + attn_out

        if self.mlp is not None:
            x = x + self.mlp(norm(x), c_fc, c_proj)

        return x, new_kv_cache


class GPTInference(nn.Module):
    """GPT model for inference."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.bigram_embed = nn.Embedding(config.vocab_size * 5, config.model_dim)
        self.value_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.model_dim) for _ in range(3)
        ])

        # Gates
        self.smear_gate = nn.Linear(12, 1, bias=False)
        self.skip_gate = nn.Linear(12, 1, bias=False)

        # Parameter banks
        self.attn_bank = nn.Parameter(torch.empty(10, 4 * config.model_dim, config.num_heads * config.head_dim))
        self.mlp_bank = nn.Parameter(torch.empty(12, 2, 4 * config.model_dim, config.model_dim))
        self.attn_gate_bank = nn.Parameter(torch.empty(10, config.num_heads, 12))
        self.ve_gate_bank = nn.Parameter(torch.empty(5, config.num_heads, 12))

        # Scalars
        self.scalars = nn.Parameter(torch.empty(51))
        self.x0_lambdas = nn.Parameter(torch.empty(config.num_layers))

        # Output
        self.lm_head_weight = nn.Parameter(torch.empty(config.model_dim, config.vocab_size))

        # Layer configuration
        self.attn_layer_indices = [i for i in range(config.num_layers) if i != 6]
        self.mlp_layer_indices = list(range(config.num_layers))
        self.layer_to_attn_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.attn_layer_indices)}
        self.layer_to_mlp_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.mlp_layer_indices)}
        self.paired_head_layers = [0, 2, 5, 9]

        # Create blocks
        self.blocks = nn.ModuleList([
            Block(config, has_attn=(i != 6), has_mlp=True)
            for i in range(config.num_layers)
        ])

        # Rotary embeddings
        self.rotary = Rotary(config.head_dim)

    def load_from_checkpoint(self, state_dict: dict):
        """Load weights from extracted checkpoint."""
        # Remove _orig_mod. prefix if present
        clean_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace('_orig_mod.', '')
            clean_state[clean_k] = v

        self.embed.weight.data = clean_state['embed.weight']
        self.bigram_embed.weight.data = clean_state['bigram_embed.weight']
        for i in range(3):
            self.value_embeds[i].weight.data = clean_state[f'value_embeds.{i}.weight']

        self.smear_gate.weight.data = clean_state['smear_gate.weight']
        self.skip_gate.weight.data = clean_state['skip_gate.weight']

        self.attn_bank.data = clean_state['attn_bank']
        self.mlp_bank.data = clean_state['mlp_bank']
        self.attn_gate_bank.data = clean_state['attn_gate_bank']
        self.ve_gate_bank.data = clean_state['ve_gate_bank']

        self.scalars.data = clean_state['scalars']
        self.x0_lambdas.data = clean_state['x0_lambdas']

        self.lm_head_weight.data = clean_state['lm_head.weight']

    def forward(
        self,
        input_ids: torch.Tensor,
        bigram_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> tuple[torch.Tensor, list]:
        """Forward pass for inference."""
        B, T = input_ids.shape
        device = input_ids.device

        # Compute bigram IDs if not provided
        if bigram_ids is None:
            # Simple bigram: current_token * vocab_size + prev_token
            if T > 1:
                prev_tokens = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
            else:
                prev_tokens = torch.zeros_like(input_ids)
            bigram_ids = input_ids + prev_tokens * self.config.vocab_size
            bigram_ids = bigram_ids.clamp(0, self.config.vocab_size * 5 - 1)

        # Extract lambdas from scalars
        resid_lambdas = self.scalars[:self.num_layers]
        sa_lambdas = self.scalars[self.num_layers:3 * self.num_layers].view(-1, 2)
        bigram_lambdas = self.scalars[3 * self.num_layers:4 * self.num_layers]
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers + 1]
        skip_lambda = self.scalars[4 * self.num_layers + 2]

        # Key offset pattern
        key_offset = [False, False, False, True, False, False, False, False, False, False, True]

        # Embeddings
        x = self.embed(input_ids)
        x0_bigram = self.bigram_embed(bigram_ids)

        # Value embeddings
        ve_list = [ve(input_ids) for ve in self.value_embeds]
        ve = [ve_list[1], ve_list[2]] + [None] * (self.num_layers - 5) + [ve_list[0], ve_list[1], ve_list[2]]

        # Smear gate (skip for single token generation)
        if T > 1:
            smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :12]))
            x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)

        x = x0 = norm(x)

        # Unbind gate banks
        ag = list(self.attn_gate_bank.unbind(0))
        veg = list(self.ve_gate_bank.unbind(0))
        attn_gates = ag[:6] + [None] + ag[6:]
        ve_gates = [veg[0], veg[1]] + [None] * (self.num_layers - 5) + [veg[2], veg[3], veg[4]]

        # Unbind weight banks
        attn_weights = self.attn_bank.unbind(0)
        mlp_fcs = self.mlp_bank[:, 0, :, :].unbind(0)
        mlp_projs = self.mlp_bank[:, 1, :, :].unbind(0)

        # Initialize KV caches if needed
        if kv_caches is None:
            kv_caches = [None] * self.num_layers
        new_kv_caches = []

        # Skip connections
        skip_connections = []
        skip_in = [3]
        skip_out = [6]
        x_backout = None
        backout_layer = 7

        for i in range(self.num_layers):
            # Skip connection out
            if i in skip_out:
                skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :12]))
                if skip_connections:
                    x = x + skip_gate_out * skip_connections.pop()

            # Residual with lambdas
            if i == 0:
                x = (resid_lambdas[0] + self.x0_lambdas[0]) * x + bigram_lambdas[0] * x0_bigram
            else:
                x = resid_lambdas[i] * x + self.x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

            # Get weights for this layer
            qkvo_w = attn_weights[self.layer_to_attn_idx[i]] if i in self.layer_to_attn_idx else None
            c_fc = mlp_fcs[self.layer_to_mlp_idx[i]] if i in self.layer_to_mlp_idx else None
            c_proj = mlp_projs[self.layer_to_mlp_idx[i]] if i in self.layer_to_mlp_idx else None

            x, new_kv = self.blocks[i](
                x, qkvo_w, c_fc, c_proj,
                self.rotary, sa_lambdas[i],
                attn_gates[i], ve[i], ve_gates[i],
                key_offset[i], kv_caches[i]
            )
            new_kv_caches.append(new_kv)

            # Skip connection in
            if i in skip_in:
                skip_connections.append(x.clone())

            # Backout
            if i == backout_layer:
                x_backout = x.clone()

        # Back out early layer contributions
        if x_backout is not None:
            x = x - backout_lambda * x_backout

        x = norm(x)

        # Output projection (transposed weight)
        logits = x @ self.lm_head_weight.type_as(x)

        # Note: Original uses softcapping: 23 * torch.sigmoid((logits + 5) / 7.5)
        # But this squashes logits to [0, 23] which hurts generation quality
        # For generation, we use raw logits instead

        return logits, new_kv_caches


def generate(
    model: GPTInference,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "mps",
) -> str:
    """Generate text from a prompt with repetition penalty."""
    model.eval()
    model.to(device)

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    # Generate
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Always run full sequence (no KV cache for now)
            logits, _ = model(generated_ids, kv_caches=None)
            logits = logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated_ids[0].tolist():
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop on EOS
            if next_token.item() == enc.eot_token:
                break

    # Decode
    output_ids = generated_ids[0].tolist()
    return enc.decode(output_ids)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on modded-nanogpt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/state_step001600_model_only.pt",
                        help="Path to model weights")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu, cuda)")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    config = ModelConfig()
    model = GPTInference(config)

    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_from_checkpoint(state_dict)

    # Convert to float32 for MPS compatibility (bfloat16 has issues on MPS)
    model = model.float()

    print(f"Generating with prompt: '{args.prompt}'")
    output = generate(
        model, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
    )
    print("\n" + "=" * 50)
    print(output)
    print("=" * 50)


if __name__ == "__main__":
    main()
