#!/usr/bin/env python3
"""Test harness for multi-tower-moe."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from moe.moe_layer import TopKGating, FeedForward, MoELayer
from moe.tiny_transformer import TinyTransformer


def test_topk_gating():
    gate = TopKGating(input_dim=8, num_experts=4, top_k=2)
    x = np.random.randn(2, 3, 8)
    weights, indices = gate.forward(x)
    assert weights.shape == (2, 3, 2), f"Expected (2,3,2), got {weights.shape}"
    assert indices.shape == (2, 3, 2), f"Expected (2,3,2), got {indices.shape}"
    assert np.allclose(weights.sum(axis=-1), 1.0)
    print("  [PASS] TopKGating shape and sum")


def test_feedforward():
    ff = FeedForward(8, 16)
    x = np.random.randn(5, 8)
    out = ff.forward(x)
    assert out.shape == (5, 8), f"Expected (5,8), got {out.shape}"
    print("  [PASS] FeedForward shape")


def test_moe_layer():
    moe = MoELayer(d_model=8, d_ff=16, num_experts=4, top_k=2)
    x = np.random.randn(2, 3, 8)
    out = moe.forward(x)
    assert out.shape == (2, 3, 8), f"Expected (2,3,8), got {out.shape}"
    print("  [PASS] MoELayer shape")


def test_tiny_transformer_forward():
    model = TinyTransformer(vocab_size=20, d_model=8, d_ff=16, num_experts=2)
    tokens = np.random.randint(0, 20, size=(1, 10))
    logits = model.forward(tokens)
    assert logits.shape == (1, 10, 20), f"Expected (1,10,20), got {logits.shape}"
    print("  [PASS] TinyTransformer forward shape")


def test_tiny_transformer_generate():
    model = TinyTransformer(vocab_size=20, d_model=8, d_ff=16, num_experts=2)
    tokens = np.random.randint(0, 20, size=(1, 5))
    result = model.generate(tokens, max_new=5)
    assert result.shape == (1, 10), f"Expected (1,10), got {result.shape}"
    print("  [PASS] TinyTransformer generation shape")


if __name__ == "__main__":
    print("Running multi-tower-moe tests...")
    test_topk_gating()
    test_feedforward()
    test_moe_layer()
    test_tiny_transformer_forward()
    test_tiny_transformer_generate()
    print("\nAll tests passed!")
