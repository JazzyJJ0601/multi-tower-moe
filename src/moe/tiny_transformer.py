import numpy as np
from .moe_layer import MoELayer


class TinyTransformer:
    """A minimal decoder-only transformer with one MoE feedforward layer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_ff: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        max_seq_len: int = 64,
    ):
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02

        # Single self-attention head
        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02

        # MoE feedforward
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)

        # Layer norm and output projection
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def layer_norm(self, x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + 1e-5) + b

    def attention(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Causal mask
        scores = Q @ K.swapaxes(-1, -2) / np.sqrt(D)
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores = scores + mask

        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)

        out = attn @ V
        return out @ self.Wo

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """tokens: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)"""
        B, T = tokens.shape

        # Embed
        x = self.token_embed[tokens] + self.pos_embed[:T][np.newaxis, :, :]

        # Attention + residual + layernorm
        x = x + self.attention(x)
        x = self.layer_norm(x, self.ln1_g, self.ln1_b)

        # MoE + residual + layernorm
        x = x + self.moe.forward(x)
        x = self.layer_norm(x, self.ln2_g, self.ln2_b)

        # LM head
        logits = x @ self.lm_head
        return logits

    def generate(self, tokens: np.ndarray, max_new: int = 20) -> np.ndarray:
        """Autoregressive generation."""
        for _ in range(max_new):
            if tokens.shape[1] > self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len:]
            logits = self.forward(tokens)
            next_logits = logits[:, -1, :]
            probs = np.exp(next_logits - next_logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            next_token = np.random.choice(probs.shape[-1], p=probs[0])
            tokens = np.concatenate([tokens, np.array([[next_token]])], axis=1)
        return tokens

    def params(self):
        return [
            self.token_embed,
            self.pos_embed,
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            self.lm_head,
            *self.moe.params(),
        ]
