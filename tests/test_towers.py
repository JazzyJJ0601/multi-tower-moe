import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
import run_real  # noqa: E402
from run_real import TowerFFN, balanced_kmeans, calibrate  # noqa: E402

MODES = ["static", "random-lowrank", "cluster-centroid", "cluster-lowrank", "neuron-lowrank",
         "random-oracle", "cluster-oracle"]


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = Qwen3ForCausalLM(cfg).float().eval()
    calib = [torch.randint(0, 256, (48,)) for _ in range(4)]
    run_real.E = 8
    stats = calibrate(model, calib, rank=64, n_towers=8)     # rank = hidden size: the router is exact
    ids = torch.randint(0, 256, (1, 40))
    with torch.no_grad():
        dense = model(ids).logits
    for l, st in zip(model.model.layers, stats):
        l.mlp = TowerFFN(l.mlp, st)
    return model, stats, ids, dense


def set_mode(model, mode):
    for l in model.model.layers:
        l.mlp.mode = mode


def test_balanced_kmeans_gives_equal_towers():
    feat = torch.nn.functional.normalize(torch.randn(96, 10), dim=1)
    a = balanced_kmeans(feat, 8)
    assert torch.bincount(a, minlength=8).tolist() == [12] * 8


@pytest.mark.parametrize("mode", MODES)
def test_keeping_everything_is_dense(tiny, mode):
    model, _, ids, dense = tiny
    set_mode(model, (mode, 1.0))
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, dense, atol=1e-4)


def test_full_rank_router_matches_oracle(tiny):
    model, _, ids, _ = tiny
    out = {}
    for mode in ["cluster-lowrank", "cluster-oracle"]:
        set_mode(model, (mode, 0.25))
        with torch.no_grad():
            out[mode] = model(ids).logits
    assert torch.allclose(out["cluster-lowrank"], out["cluster-oracle"], atol=1e-3)


def test_sparse_modes_change_the_output(tiny):
    model, _, ids, dense = tiny
    set_mode(model, ("cluster-lowrank", 0.25))
    with torch.no_grad():
        assert not torch.allclose(model(ids).logits, dense, atol=1e-3)


def test_static_keeps_exact_fraction(tiny):
    _, stats, _, _ = tiny
    assert int((stats[0]["static_rank"] < 32).sum()) == 32
