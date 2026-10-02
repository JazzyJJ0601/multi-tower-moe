"""Turn Qwen3-8B's dense FFNs into mixtures of expert "towers" without training, and measure the cost.

Each layer's 12,288 FFN neurons are split into E equal towers. Per token, a router picks the top-k towers;
only those neurons' gate/up/down rows are used. Everything is calibrated on WikiText-2 train and scored
on WikiText-2 test, at equal fraction of FFN neurons computed per token.

Partitions:  cluster (balanced k-means on neuron co-activation, ours) | random.
Routers:     lowrank (ours: the layer's own gate/up projections restricted to the top-r principal
             directions of its input, so the router predicts each neuron's output without training)
             | centroid (mean gate row of the tower, the usual training-free router)
             | oracle (the exact tower contributions; needs the full FFN, an upper bound only).
Baselines:   static pruning (same neurons for every token, ranked on calibration data) and
             per-neuron selection with the same low-rank predictor (finer-grained, not tower-shaped).

Usage: python results/run_real.py   -> results/real.json
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qcommon import load_model, perplexity, save, windows  # noqa: E402

E = 64          # towers per layer (192 neurons each in Qwen3-8B)
RANK = 256      # low-rank router: principal directions of the FFN input
N_CALIB = 16    # 16 x 512 train tokens
N_EVAL = 40     # 40 x 512 test tokens (same eval set as the other repos)
KEEP = [0.5, 0.25]
OUT = Path(__file__).resolve().parent / "real.json"


def balanced_kmeans(feat, k, iters=15, seed=0):
    """Cluster rows of feat (N x d, L2-normalised) into k clusters of exactly N/k rows (cosine)."""
    n = feat.shape[0]
    cap = n // k
    g = torch.Generator(device="cpu").manual_seed(seed)
    cent = feat[torch.randperm(n, generator=g)[:k].to(feat.device)]
    assign = None
    for _ in range(iters):
        sim = (feat @ cent.T).cpu()                      # N x k
        best, _ = sim.max(1)
        prefs = sim.argsort(1, descending=True).tolist()
        load = [0] * k
        new = [0] * n
        for i in best.argsort(descending=True).tolist():  # most confident neurons choose first
            for c in prefs[i]:
                if load[c] < cap:
                    new[i] = c
                    load[c] += 1
                    break
        new = torch.tensor(new)
        if assign is not None and torch.equal(new, assign):
            break
        assign = new
        a = assign.to(feat.device)
        cent = F.normalize(torch.zeros_like(cent).index_add_(0, a, feat), dim=1)
    return assign


class TowerFFN(torch.nn.Module):
    """Drop-in replacement for a Qwen3 MLP that keeps only the routed neurons per token."""

    def __init__(self, mlp, stats):
        super().__init__()
        self.mlp, self.s = mlp, stats
        self.mode = "dense"

    def forward(self, x):
        m, s = self.mlp, self.s
        a = m.act_fn(m.gate_proj(x)) * m.up_proj(x)                       # ... x N
        if self.mode == "dense":
            return m.down_proj(a)
        kind, keep = self.mode
        n = a.shape[-1]
        if kind == "static":
            mask = s["static_rank"] < round(keep * n)                     # N bools, same for all tokens
            return m.down_proj(a * mask.to(a.dtype))
        if kind == "neuron-lowrank":
            score = self.approx(x).abs() * s["dnorm"]
            idx = score.topk(round(keep * n), -1).indices
            mask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)
            return m.down_proj(a * mask.to(a.dtype))
        part, router = kind.split("-")
        tower = s["tower_" + part]                                        # N -> tower id
        if router == "oracle":
            contrib = a.float().abs() * s["dnorm"]
        elif router == "lowrank":
            contrib = self.approx(x).abs() * s["dnorm"]
        if router == "centroid":
            score = x.float() @ s["centroid_" + part].T                   # ... x E
        else:
            score = torch.zeros(*contrib.shape[:-1], E, device=x.device).index_add_(-1, tower, contrib)
        top = score.topk(round(keep * E), -1).indices
        tmask = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, top, True)
        mask = tmask[..., tower]
        return m.down_proj(a * mask.to(a.dtype))

    def approx(self, x):
        z = x.float() @ self.s["P"]                                       # ... x r
        return F.silu(z @ self.s["Gp"].T) * (z @ self.s["Up"].T)


@torch.no_grad()
def calibrate(model, calib, rank=RANK, n_towers=E):
    """Per-layer statistics: neuron importance, towers (clustered and random), routers."""
    layers = model.model.layers
    xs = [[] for _ in layers]
    hooks = [l.mlp.register_forward_pre_hook(lambda m, inp, i=i: xs[i].append(inp[0][0].cpu()))
             for i, l in enumerate(layers)]
    for w in calib:
        model(w.unsqueeze(0).to(model.device))
    for h in hooks:
        h.remove()
    stats = []
    for i, l in enumerate(layers):
        mlp = l.mlp
        dev = mlp.gate_proj.weight.device
        X = torch.cat(xs[i]).to(dev).float()
        xs[i] = None
        Wg, Wu = mlp.gate_proj.weight.float(), mlp.up_proj.weight.float()
        dnorm = mlp.down_proj.weight.float().norm(dim=0)
        A = F.silu(X @ Wg.T) * (X @ Wu.T)                                  # T x N
        C = A.abs() * dnorm
        n = C.shape[1]
        static_rank = torch.empty(n, dtype=torch.long, device=dev)
        static_rank[C.mean(0).argsort(descending=True)] = torch.arange(n, device=dev)
        feat = F.normalize((C / C.mean(0).clamp_min(1e-8)).T.contiguous(), dim=1)   # co-activation profile
        tower_cluster = balanced_kmeans(feat, n_towers, seed=i).to(dev)
        g = torch.Generator(device="cpu").manual_seed(1000 + i)
        tower_random = (torch.randperm(n, generator=g) % n_towers).to(dev)
        evals, evecs = torch.linalg.eigh(X.T @ X)
        P = evecs[:, -rank:].contiguous()                                  # top principal directions
        st = dict(dnorm=dnorm, static_rank=static_rank, tower_cluster=tower_cluster,
                  tower_random=tower_random, P=P, Gp=Wg @ P, Up=Wu @ P,
                  var_kept=float(evals[-rank:].sum() / evals.sum()))
        for part in ("cluster", "random"):
            t = st["tower_" + part]
            cent = torch.zeros(n_towers, Wg.shape[1], device=dev).index_add_(0, t, F.normalize(Wg, dim=1))
            st["centroid_" + part] = cent / (n // n_towers)
        stats.append(st)
        del X, A, C, feat
        torch.cuda.empty_cache()
    return stats


def main():
    t0 = time.time()
    model, tok = load_model()
    calib, test = windows(tok, "train", N_CALIB), windows(tok, "test", N_EVAL)
    stats = calibrate(model, calib)
    t_cal = time.time() - t0
    ffns = []
    for l, st in zip(model.model.layers, stats):
        l.mlp = TowerFFN(l.mlp, st)
        ffns.append(l.mlp)

    def run(mode):
        for f in ffns:
            f.mode = mode
        return round(perplexity(model, test), 4)

    hid, n = model.config.hidden_size, model.config.intermediate_size
    ffn_macs = 3 * hid * n
    router_macs = {"lowrank": RANK * hid + 2 * n * RANK, "centroid": E * hid, "oracle": None, "neuron": RANK * hid + 2 * n * RANK}
    res = dict(model="Qwen3-8B", eval=f"wikitext-2 test {N_EVAL}x512", calib=f"wikitext-2 train {N_CALIB}x512",
               towers=E, neurons_per_tower=n // E, rank=RANK,
               router_overhead_pct={k: (round(100 * v / ffn_macs, 2) if v else None) for k, v in router_macs.items()},
               var_kept_mean=round(sum(s["var_kept"] for s in stats) / len(stats), 4),
               calib_seconds=round(t_cal), rows=[])
    res["rows"].append(dict(method="dense", keep=1.0, ppl=run("dense")))
    print(res["rows"][-1], flush=True)
    for keep in KEEP:
        for m in ["static", "random-lowrank", "cluster-centroid", "cluster-lowrank", "neuron-lowrank",
                  "random-oracle", "cluster-oracle"]:
            res["rows"].append(dict(method=m, keep=keep, ppl=run((m, keep))))
            print(res["rows"][-1], flush=True)
            save(OUT, res)
    res["seconds"] = round(time.time() - t0)
    save(OUT, res)


if __name__ == "__main__":
    main()
