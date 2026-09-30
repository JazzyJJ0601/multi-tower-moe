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

### Expert Capacity Constraints
Each expert has a fixed capacity (tokens it can process per batch). Tokens exceeding capacity are routed to a secondary expert or dropped:
- Capacity Factor: 1.25 (allows 25% over-provisioning)
- Overflow tokens: routed to least-loaded expert with available capacity

## Training
### Dataset
- **Source:** Shakespeare char-level text corpus
- **Preprocessing:** Character tokenization, fixed sequence length (128 tokens)
- **Batch Size:** 32

### Training Loop Design
1. **Forward Pass:**
   - Compute gating scores for all tokens
   - Route top-k experts per token (k=2)
   - Compute expert outputs in parallel
   - Aggregate outputs with gating weights
   - Pass through output head
2. **Loss Computation:**
   - Cross-entropy loss for language modeling
   - Add auxiliary load-balancing loss (alpha=0.01)
3. **Backward Pass:**
   - Compute gradients for task loss
   - Compute gradients for auxiliary loss
   - Backpropagate through gating network and selected experts
4. **Optimization:**
   - Adam optimizer with lr=1e-3
   - Gradient clipping (max_norm=1.0)
   - 5000 iterations with checkpointing every 500 steps

### Optimization
- **Optimizer:** Adam (lr=1e-3)
- **Loss:** Cross-Entropy (Language Modeling) + Load-Balancing Loss
- **Steps:** 5000 iterations

## Evaluation Plan
1. **Baseline:** Train a dense transformer (same parameter count as MoE model)
2. **Small Real Model:**
   - Use 12-layer TinyLlama (1.1B) or smaller GPT-2 (117M) as reference
   - Fine-tune on same Shakespeare dataset for comparison
3. **Metric:** Perplexity on held-out validation set (lower is better)
4. **Efficiency Metrics:**
   - Tokens processed per second
   - GPU memory usage
   - FLOPs per token
5. **Analysis:** Measure expert utilization statistics (how often each expert is chosen)
6. **Convergence Analysis:** Plot perplexity vs. training steps for MoE vs. dense

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
