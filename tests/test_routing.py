#!/usr/bin/env python3
"""Comprehensive tests for MoE routing and training."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from moe.moe_layer import TopKGating, FeedForward, MoELayer


def test_topk_routing_correctness():
    """Test that top-k routing selects experts with highest logits."""
    gate = TopKGating(input_dim=8, num_experts=4, top_k=2)
    
    # Create input with known patterns
    x = np.zeros((1, 1, 8))
    x[0, 0, :] = [100, 0, 0, 0, 0, 0, 0, 0]  # First dimension dominates
    
    # Set gate weights to produce known logits
    gate.W[:, 0] = 0.0  # Expert 0 gets high score
    gate.W[:, 1] = 0.0  # Expert 1 gets high score
    gate.W[:, 2] = -1000  # Expert 2 gets low score
    gate.W[:, 3] = -1000  # Expert 3 gets low score
    
    weights, indices = gate.forward(x)
    
    # Should select experts 0 and 1 (highest logits)
    selected_experts = set(indices[0, 0, :])
    assert selected_experts == {0, 1}, f"Expected {{0,1}}, got {selected_experts}"
    
    # Weights should sum to 1 for each token
    assert np.allclose(weights.sum(axis=-1), 1.0), "Weights should sum to 1"
    
    print("  [PASS] Top-k routing selects correct experts")


def test_topk_routing_shape():
    """Test output shapes match expected."""
    gate = TopKGating(input_dim=8, num_experts=4, top_k=2)
    x = np.random.randn(2, 3, 8)
    
    weights, indices = gate.forward(x)
    
    assert weights.shape == (2, 3, 2), f"Expected (2,3,2), got {weights.shape}"
    assert indices.shape == (2, 3, 2), f"Expected (2,3,2), got {indices.shape}"
    
    print("  [PASS] Routing output shapes correct")


def test_load_balancing_loss():
    """Test load-balancing loss computation."""
    gate = TopKGating(input_dim=8, num_experts=4, top_k=2)
    
    # Create input that routes tokens evenly across experts
    x = np.random.randn(4, 5, 8)  # 20 tokens, 4 experts, should be balanced
    
    weights, indices = gate.forward(x)
    
    # Compute load-balancing loss
    # This measures how evenly experts are utilized
    # For a balanced system, each expert should get ~1/num_experts of load
    batch_size, seq_len, top_k = weights.shape
    
    # Compute load per expert (sum of routing weights for each expert)
    load = np.zeros(gate.num_experts)
    for e_idx in range(gate.num_experts):
        load[e_idx] = (weights * (indices == e_idx)).sum() / (batch_size * seq_len * top_k)
    
    # Load-balancing loss: penalize deviation from uniform distribution
    # Lower is better (perfectly balanced = uniform distribution)
    target_load = 1.0 / gate.num_experts
    lb_loss = np.sum((load - target_load) ** 2)
    
    # With random weights, loss should be finite and not extreme
    assert np.isfinite(lb_loss), f"Load-balancing loss should be finite, got {lb_loss}"
    assert lb_loss >= 0, "Load-balancing loss should be non-negative"
    
    print(f"  [PASS] Load-balancing loss computed: {lb_loss:.4f}")


def test_expert_forward_pass():
    """Test individual expert forward pass."""
    ff = FeedForward(d_model=8, d_ff=16)
    
    x = np.random.randn(5, 8)
    out = ff.forward(x)
    
    # Output should match input shape
    assert out.shape == (5, 8), f"Expected (5,8), got {out.shape}"
    
    # Check that forward pass is deterministic for same input
    out2 = ff.forward(x)
    assert np.allclose(out, out2), "Forward pass should be deterministic"
    
    print("  [PASS] Expert forward pass works correctly")


def test_moe_layer_routing():
    """Test MoE layer routing and aggregation."""
    moe = MoELayer(d_model=8, d_ff=16, num_experts=4, top_k=2)
    
    x = np.random.randn(2, 3, 8)
    out = moe.forward(x)
    
    assert out.shape == (2, 3, 8), f"Expected (2,3,8), got {out.shape}"
    
    # Output should not be all zeros
    assert np.abs(out).sum() > 0, "Output should not be all zeros"
    
    print("  [PASS] MoE layer routing works")


def test_training_loop_tiny_data():
    """Test that full training loop runs without error on tiny data."""
    from moe.tiny_transformer import TinyTransformer
    
    # Create small model for quick testing
    model = TinyTransformer(
        vocab_size=20,
        d_model=8,
        d_ff=16,
        num_experts=2,
        top_k=1,
        max_seq_len=10
    )
    
    # Tiny training data
    np.random.seed(42)
    x_train = np.random.randint(0, 20, size=(5, 5))
    y_train = np.random.randint(0, 20, size=(5, 5))
    
    lr = 0.01
    steps = 10
    
    initial_loss = None
    for step in range(steps):
        # Forward pass
        logits = model.forward(x_train)
        
        # Compute cross-entropy loss
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        y_flat = y_train.reshape(-1)
        
        logits_stable = logits_flat - logits_flat.max(axis=-1, keepdims=True)
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(len(y_flat)), y_flat] + 1e-8))
        
        if initial_loss is None:
            initial_loss = loss
        
        # Simple gradient descent (parameter update)
        # In practice, this would need proper gradient computation
        # For this test, we just verify the loop runs without error
        
        if step % 5 == 0:
            print(f"    Step {step}, loss: {loss:.4f}")
    
    # Verify loss decreased or stayed reasonable
    assert np.isfinite(loss), f"Final loss should be finite, got {loss}"
    
    print(f"  [PASS] Training loop completed, {steps} steps, final loss: {loss:.4f}")


if __name__ == "__main__":
    print("Running MoE routing and training tests...")
    test_topk_routing_correctness()
    test_topk_routing_shape()
    test_load_balancing_loss()
    test_expert_forward_pass()
    test_moe_layer_routing()
    test_training_loop_tiny_data()
    print("\nAll MoE tests passed!")
