#!/usr/bin/env python
"""Extract model weights from checkpoint, handling custom classes."""

import torch
import sys


# Define stub classes that match what's in the checkpoint
class ParamConfig:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Hyperparameters:
    pass


class ForwardScheduleConfig:
    pass


class AttnArgs:
    pass


# Inject these into __main__ so torch.load can find them
import __main__
__main__.ParamConfig = ParamConfig
__main__.Hyperparameters = Hyperparameters
__main__.ForwardScheduleConfig = ForwardScheduleConfig
__main__.AttnArgs = AttnArgs


def load_checkpoint(path):
    """Load checkpoint with stub classes."""
    return torch.load(path, map_location='cpu', weights_only=False)


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/state_step001600.pt"
    print(f"Loading checkpoint from {ckpt_path}...")

    try:
        ckpt = load_checkpoint(ckpt_path)
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        print(f"Step: {ckpt.get('step')}")

        if 'model' in ckpt:
            model_state = ckpt['model']
            print(f"\nModel state dict keys ({len(model_state)}):")
            for k in sorted(model_state.keys()):
                v = model_state[k]
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape} ({v.dtype})")
                else:
                    print(f"  {k}: {type(v)}")

            # Save just the model weights
            output_path = ckpt_path.replace('.pt', '_model_only.pt')
            torch.save(model_state, output_path)
            print(f"\nSaved model weights to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
