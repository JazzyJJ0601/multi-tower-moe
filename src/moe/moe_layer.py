import numpy as np


class TopKGating:
    """Top-k softmax gating for Mixture-of-Experts."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # Gating weights: (input_dim, num_experts)
        self.W = np.random.randn(input_dim, num_experts) * 0.02

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gating logits, select top-k, return weights and expert indices.
        x: (batch, seq, input_dim) or (batch * seq, input_dim)
        """
        orig_shape = x.shape
        if x.ndim == 3:
            B, T, D = x.shape
            x_flat = x.reshape(-1, D)
        else:
            x_flat = x

        # logits: (N, num_experts)
        logits = x_flat @ self.W

        # Top-k indices and values
        # kth must be < num_experts for argpartition
        kth = min(self.top_k, self.num_experts - 1) if self.num_experts > 1 else 0
        indices = np.argpartition(-logits, kth, axis=-1)[:, :self.top_k]
        top_vals = np.take_along_axis(logits, indices, axis=-1)

        # Softmax over top-k only
        exp_vals = np.exp(top_vals - top_vals.max(axis=-1, keepdims=True))
        weights = exp_vals / exp_vals.sum(axis=-1, keepdims=True)

        if x.ndim == 3:
            weights = weights.reshape(B, T, self.top_k)
            indices = indices.reshape(B, T, self.top_k)

        return weights, indices

    def params(self):
        return [self.W]


class FeedForward:
    """Simple 2-layer MLP (the "expert tower")."""

    def __init__(self, d_model: int, d_ff: int):
        self.w1 = np.random.randn(d_model, d_ff) * 0.02
        self.w2 = np.random.randn(d_ff, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.w1)  # ReLU
        return h @ self.w2

    def params(self):
        return [self.w1, self.w2]


class MoELayer:
    """Sparse Mixture-of-Experts layer with top-k routing."""

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        self.gate = TopKGating(d_model, num_experts, top_k)
        self.top_k = top_k
        self.experts = [FeedForward(d_model, d_ff) for _ in range(num_experts)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)"""
        B, T, D = x.shape
        weights, indices = self.gate.forward(x)  # (B,T,k), (B,T,k)

        out = np.zeros_like(x)

        # For each expert, gather tokens that routed to it
        for e_idx, expert in enumerate(self.experts):
            mask = (indices == e_idx).any(axis=-1)  # (B,T)
            if not mask.any():
                continue
            rows, cols = np.where(mask)
            token_inputs = x[rows, cols]  # (N, D)
            expert_out = expert.forward(token_inputs)  # (N, D)

            # Weighted sum: each token's weight for this expert
            w = np.zeros(len(rows))
            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                for k in range(self.top_k):
                    if indices[r, c, k] == e_idx:
                        w[i] = weights[r, c, k]
                        break

            out[rows, cols] += expert_out * w[:, np.newaxis]

        return out

    def params(self):
        p = [self.gate.params()]
        for e in self.experts:
            p.extend(e.params())
        return p
