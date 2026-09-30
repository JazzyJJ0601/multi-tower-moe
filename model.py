"""
Multi-Tower Mixture of Experts (MoE) Model - Pure PyTorch Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden)


class GatingNetwork(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor, top_k: int = 2):
        router_logits = self.router(x)
        topk_logits, topk_indices = torch.topk(router_logits, top_k, dim=-1)
        routed_weights = F.softmax(topk_logits, dim=-1)
        return routed_weights, topk_indices, router_logits


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2, alpha: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = GatingNetwork(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        routed_weights, routed_indices, router_logits = self.gate(x, self.top_k)
        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            mask = (routed_indices == expert_idx).any(dim=-1)
            if mask.any():
                expert_weights = routed_weights * (routed_indices == expert_idx).to(x.dtype)
                expert_weights_sum = expert_weights.sum(dim=-1, keepdim=True)
                expert_output = self.experts[expert_idx](x)
                output = output + expert_output * expert_weights_sum

        aux_loss = self._load_balancing_loss(router_logits)
        return output, aux_loss

    def _load_balancing_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(router_logits, dim=-1)
        expert_usage = probs.mean(dim=(0, 1))
        uniform_dist = torch.ones_like(expert_usage) / self.num_experts
        return self.alpha * F.mse_loss(expert_usage, uniform_dist)


class MoEModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, d_ff: int = 256,
                 num_experts: int = 8, top_k: int = 2, num_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MoELayer(d_model, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=x.dtype)
        for layer in self.layers:
            x, layer_aux_loss = layer(x)
            aux_loss = aux_loss + layer_aux_loss
        x = self.output_norm(x)
        logits = self.head(x)
        return logits, aux_loss
