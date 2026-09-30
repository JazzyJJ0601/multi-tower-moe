# Multi-Tower Mixture of Experts (MoE) Design

## Overview
This project implements a Mixture of Experts (MoE) model from scratch using pure Python and NumPy. The goal is to demonstrate efficient sparse computation for sequence modeling tasks.

## Architecture

```
[Input Tokens] --> [Embedding Layer] --> [Multi-Tower MoE Block] --> [Output Head]
                             |
                             v
                     [Gating Network (Softmax)]
                             |
                     +-------+-------+
                     |       |       |
                  [Expert 1] [Expert 2] ... [Expert N]
                     |       |       |
                     +-------+-------+
                             |
                     [Weighted Sum of Outputs]
```

## Expert Architecture
Each expert is a tiny transformer block:
- **Layers:** 2
- **Attention Heads:** 4
- **Model Dimension (d_model):** 64
- **Feed-Forward:** 2x expansion
- **Normalization:** LayerNorm pre-attention

## Routing Mechanism
### Top-K Gating
The gating network computes scores for each expert. We select the top-k experts (k=2) for each token based on these scores.
- Input: Token embedding (d_model)
- Output: Expert logits (num_experts)
- Selection: Argmax top-k indices

### Load-Balancing Loss
To prevent expert collapse (where one expert dominates), we add a auxiliary loss term. The load-balancing loss encourages uniform usage of experts across a batch.
- Formula: `L_lb = alpha * sum(mean(gating_scores_per_expert))`
- Regularization: Penalizes high variance in expert selection frequency

### Entropy-Based Load Balancing
Additionally, we maximize the entropy of the routing distribution to ensure diverse expert utilization. This complements the frequency-based load balancing loss.

## Training
### Dataset
- **Source:** Shakespeare char-level text corpus
- **Preprocessing:** Character tokenization, fixed sequence length (128 tokens)
- **Batch Size:** 32

### Optimization
- **Optimizer:** Adam (lr=1e-3)
- **Loss:** Cross-Entropy (Language Modeling) + Load-Balancing Loss
- **Steps:** 5000 iterations

## Evaluation Plan
1. **Baseline:** Train a dense transformer (same parameter count as single expert)
2. **Metric:** Perplexity on held-out validation set
3. **Comparison:** Compare MoE perplexity vs. dense baseline perplexity
4. **Analysis:** Measure expert utilization statistics (how often each expert is chosen)

## Implementation Details
- Pure NumPy for matrix operations
- No external deep learning frameworks
- Custom autograd for backpropagation (or manual gradient derivation)
- Floating point: float32

## Next Steps
1. Implement core MoE layer
2. Integrate routing logic
3. Train on Shakespeare dataset
4. Run evaluation benchmarks
