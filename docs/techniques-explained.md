# NanoGPT Speedrun Techniques Explained

A comprehensive breakdown of the techniques used in the modded-nanogpt repository for achieving fast GPT-2 training.

## 1. Modernized Architecture: Rotary Embeddings (RoPE), QK-Norm, ReLU²

- **RoPE** replaces learned absolute position embeddings with a rotation applied to **Q/K** so attention becomes position-aware via relative phase. This tends to extrapolate better to longer contexts and is now a standard "modern GPT" positional scheme. [arXiv](https://arxiv.org/pdf/2309.00071)
- **QK-Norm** normalizes queries/keys (often L2/RMS) before the dot-product so attention logits don't explode or saturate softmax; it typically improves stability and permits more aggressive learning rates. [arXiv](https://arxiv.org/abs/2010.04245)
- **ReLU²** (squared ReLU) is a cheap nonlinearity that often trains stably in small/fast regimes and is easy to fuse efficiently (helping both quality-per-step and throughput).

## 2. The Muon Optimizer

Muon is an optimizer aimed primarily at **2D weight matrices** (linear layers). It combines a momentum-like update with an **orthogonalization / polar-factor** post-processing step so the update behaves like a "spectral" step rather than coordinate-wise adaptive scaling (as in Adam). In this repo it's used largely because it tends to be **more sample-efficient than AdamW** at this scale while staying fast enough to matter. [kellerjordan.github.io](https://kellerjordan.github.io/posts/muon/)

## 3. FP8 Matmul for Head, Asymmetric Rescale and Softcap Logits

- **FP8 lm_head matmul** accelerates the expensive vocabulary projection (hidden → vocab logits) by using lower precision on Hopper-class GPUs.
- Lower precision can distort logit magnitudes; so the code applies **rescaling** (including asymmetric scaling) plus **logit softcapping**—a saturating transform that prevents extreme logits that destabilize softmax/cross-entropy, similar in spirit to what's reported for Gemma 2. [Hugging Face](https://huggingface.co/blog/gemma2)

## 4. Initialization of Projections to Zero (muP-like)

Setting some residual-branch projections to **start at (near) zero** makes early training behave closer to an identity mapping (reducing optimization chaos at very high learning rates). The "muP-like" reference is to the broader family of scale-aware parameterizations/initializations that aim to keep updates and activations well-scaled across depth/width. [OpenReview](https://openreview.net/pdf?id=AFxEdJwQcp)

## 5. Skip Connections from Embedding to Every Block and from Block 3 to 6

These are extra "highway" paths (a U-Net-ish idea) so later blocks can directly access earlier representations (including near-raw token embeddings). This improves gradient flow and can raise sample efficiency in short training runs where you cannot rely on slow emergence of deep features. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 6. Extra Embeddings Mixed into Values in Attention Layers

(Inspired by Zhou et al. 2024)

This adds a learned signal into the **V (value)** pathway (or equivalently adds a residual/value "anchor" that attention can transport). Conceptually: attention normally routes content via V; adding "extra value embeddings" gives the model an additional, cheap content channel that can alleviate attention pathologies and improve early learning. This is related to "value residual / shared value" style ideas explored in the literature. [OpenReview](https://openreview.net/forum?id=kn3GT7LbxT)

## 7. FlashAttention 3 with Long-Short Sliding Window Attention; Window Warmup with YaRN

- **FlashAttention-3** is a highly optimized attention kernel for Hopper GPUs that improves utilization (and can support low precision). [arXiv](https://arxiv.org/abs/2407.08608)
- **Long–short sliding windows** means some layers attend locally (cheap) while some attend with a larger window (more global signal). This mirrors the "hybrid local/global" pattern popularized in models like Gemma 2. [Hugging Face](https://huggingface.co/blog/gemma2)
- **Warmup with YaRN**: as the window/context is increased over training, YaRN adjusts RoPE scaling so the model tolerates longer context lengths without falling apart. [arXiv](https://arxiv.org/pdf/2309.00071)

## 8. Align Training Batch Starts with EoS and Set a Max Document Length

This is "document alignment": start sequences at document boundaries (EoS/BOS) and cap per-document length. It reduces cross-document leakage and gradient correlation (one long doc dominating a step), and keeps the BOS token in-window. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 9. Accumulate Gradients for 2 Steps for Embedding and lm_head Before Updating

Embeddings and lm_head are large and communication-heavy in distributed training. Updating them every other step (gradient accumulation for those params) creates **heterogeneous effective batch sizes** inside the model and reduces sync/step overhead for those parameter groups, often improving time-to-target. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 10. Enable Model to Back Out Contributions from First 2/3 Layers Before Prediction

The "backout" trick subtracts (a learned fraction of) an earlier residual-stream snapshot right before the final norm/head. Intuition: early layers may add features that help downstream computation but harm the final linear readout; backout lets the model use those features internally while partially removing them for prediction. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 11. Polar Express Implementation in Muon

Muon's orthogonalization step approximates a **polar decomposition / matrix sign** operation. "Polar Express" is a newer GPU-friendly iterative method that can converge faster/better than Newton–Schulz in the precision regime used here, improving Muon's effectiveness without adding much overhead. [arXiv](https://arxiv.org/abs/2505.16932)

## 12. Smear Module to Enable 1-Token Look Back

Transformers frequently learn "previous-token heads" that mostly attend to position *t−1*—but attention is an expensive way to do that. Smear implements an explicit, gated shift-and-mix: add a learned fraction of the previous token's representation to the current token, giving the benefit of a prev-token head at far lower cost. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 13. Sparse Attention Gate

Attention lacks a true "no-op" by default. A per-head sigmoid gate (driven by a small slice of the residual stream) can scale attention output toward zero when it's not useful, reducing wasted computation/instability (e.g., BOS attention sink behavior) while keeping compatibility with fast attention kernels. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 14. NorMuon

NorMuon is a Muon variant designed to be more stable/scalable by maintaining more uniform neuron/row norms during training (the paper frames it as improving efficiency and scalability of Muon's orthogonalized updates). In practice, it's "Muon, but better behaved" in the regimes this benchmark explores. [arXiv](https://arxiv.org/html/2510.05491v1)

## 15. Cautious Weight Decay with Schedule Tied to LR

Cautious Weight Decay (CWD) applies decay only on coordinates where the decay direction doesn't fight the optimizer's update (formally: sign alignment with the update). Tying its strength to the LR schedule matches regularization pressure to learning dynamics during fast ramps/cooldowns. [arXiv](https://arxiv.org/abs/2510.12402)

## 16. Exponential Decay of Residual Stream

This typically means multiplying the residual stream (or specific residual contributions) by a depth- or time-dependent factor < 1, so earlier contributions fade in a controlled way instead of accumulating unchecked. The speedrun motivation is stability at aggressive hyperparameters; residual norms are known to drift systematically with depth, and explicit decay is one way to control that. [LessWrong](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward)

## 17. Batch Size Schedule

Rather than a fixed batch size, you change it over training (often increasing it) to balance gradient noise vs throughput as optimization progresses. There's supporting work showing batch schedules can be guided by gradient-noise-scale estimates and reduce time-to-quality. [OpenReview](https://openreview.net/forum?id=S7THlpvH8i)

## 18. Partial Key Offset

In the speedrun community description, this is a tiny, zero-parameter attention tweak that *mixes/offsets key information* so each query effectively retrieves a slightly richer signal while remaining causal (the public one-liner description is "tie each key projection to both its own value and the subsequent value while maintaining causality"). Treat this as an empirically discovered micro-architecture change rather than a standardized technique. [X](https://x.com/kellerjordan0/highlights)

## 19. Multi-token Prediction

Instead of predicting only token *t+1* at each position, add auxiliary heads to predict multiple future tokens (e.g., *t+1…t+n*) from the same trunk representation. This can improve sample-efficiency (fewer steps/tokens to a given loss) at the cost of some extra head computation. [arXiv](https://arxiv.org/abs/2404.19737)

## 20. Untie Embed and lm_head at 2/3 of Training

"Tying" shares parameters between input embeddings and the output softmax matrix; it can stabilize and regularize early training. "Untying" later increases effective capacity and reduces gradient interference between "encode tokens" and "decode logits," which can matter when you're chasing the last bits of validation loss quickly.

## 21. Additional Gating on Value Embeddings and Skip Connection

Where earlier versions used fixed-strength extra connections (value-embeds, skip paths), gating makes their contribution learnable and often initializes them small. This prevents the model from overusing shortcuts early while still allowing it to exploit them later if beneficial. [LessWrong](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)

## 22. Paired Head Attention

This is described publicly as letting a query attend not only to its own head's keys but also a neighboring head's keys—so each query can retrieve "two values per position instead of one" without adding parameters. It's a compute/expressivity trade that can be helpful in small models where head specialization is tight. [TwStalker](https://w.twstalker.com/kellerjordan0)

## 23. Bigram Hash Embedding

Add an extra embedding derived from **hashed bigram features** (token pairs) and inject it into the residual stream (e.g., before each layer). The point is to give the model a cheap, explicit n-gram signal and a kind of "lookup memory" without maintaining a full bigram table. This is inspired by classic hash embeddings and modern "conditional memory / engram"-style ideas. [AI News](https://news.smol.ai/issues/26-01-20-not-much/)

---

## References

- [GitHub - modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [LessWrong - How the NanoGPT Speedrun WR dropped by 20% in 3 months](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)
- [Muon optimizer blog post](https://kellerjordan.github.io/posts/muon/)
