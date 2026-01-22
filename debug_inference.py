#!/usr/bin/env python
"""Debug the inference model."""

import torch
import tiktoken
from inference import GPTInference, ModelConfig

def debug_model():
    print("Loading model...")
    config = ModelConfig()
    model = GPTInference(config)

    state_dict = torch.load("checkpoints/state_step001600_model_only.pt", map_location='cpu', weights_only=True)
    model.load_from_checkpoint(state_dict)
    model = model.float()
    model.eval()

    # Check weights
    print("\n=== Weight Statistics ===")
    print(f"embed.weight: mean={model.embed.weight.mean():.4f}, std={model.embed.weight.std():.4f}")
    print(f"lm_head_weight: mean={model.lm_head_weight.mean():.4f}, std={model.lm_head_weight.std():.4f}")
    print(f"attn_bank: mean={model.attn_bank.mean():.4f}, std={model.attn_bank.std():.4f}")
    print(f"mlp_bank: mean={model.mlp_bank.mean():.4f}, std={model.mlp_bank.std():.4f}")

    # Test a simple forward pass
    enc = tiktoken.get_encoding("gpt2")
    prompt = "Hello world"
    input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)

    print(f"\n=== Forward Pass ===")
    print(f"Input: '{prompt}'")
    print(f"Input IDs: {input_ids[0].tolist()}")

    with torch.no_grad():
        logits, _ = model(input_ids, kv_caches=None)

    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean():.4f}")
    print(f"Logits std: {logits.std():.4f}")
    print(f"Logits min: {logits.min():.4f}")
    print(f"Logits max: {logits.max():.4f}")

    # Check top predictions for last position
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\n=== Top {top_k} predictions for next token ===")
    for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
        token = enc.decode([idx])
        print(f"  {i+1}. '{token}' (id={idx}): {prob*100:.2f}%")

    # Check if there's a strong bias
    print(f"\n=== Token distribution analysis ===")
    print(f"Top 1 prob: {top_probs[0]*100:.2f}%")
    print(f"Top 10 prob sum: {top_probs.sum()*100:.2f}%")
    print(f"Entropy: {-(probs * torch.log(probs + 1e-10)).sum():.4f}")

    # Try different prompts
    prompts = [
        "The",
        "Once upon a time",
        "In the year 2025",
    ]

    print("\n=== Testing different prompts ===")
    for prompt in prompts:
        input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(input_ids, kv_caches=None)

        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)

        next_tokens = [enc.decode([idx]) for idx in top_indices.tolist()]
        print(f"'{prompt}' -> {next_tokens[:3]}")


if __name__ == "__main__":
    debug_model()
