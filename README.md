# Multi-tower MoE from a dense model, without training

A dense FFN computes all of its neurons for every token. This repo turns each of Qwen3-8B's 36 FFNs into a
mixture of 64 expert "towers" (192 neurons each) **without any training**, so a token only needs the towers a
router picks. Two parts are new here:

- **Clustered towers.** Neurons are grouped by balanced k-means on how they co-activate on calibration text, so
  neurons that fire together land in the same tower.
- **Low-rank router.** The router is the layer's own gate/up projections restricted to the top 256 principal
  directions of the FFN input. It predicts every neuron's output cheaply, with no learned weights, and a tower's
  score is the sum of its neurons' predicted contributions.

Calibration takes 49 seconds on 16 × 512 WikiText-2 train tokens.

## Result (Qwen3-8B, real model)

Perplexity on WikiText-2 test (40 windows of 512 tokens), lower is better. Every method computes the same share
of FFN neurons per token. Dense Qwen3-8B scores **12.03**.

| Method | 50% of neurons | 25% of neurons |
|---|---:|---:|
| static pruning (same neurons for every token) | 33.04 | 353.0 |
| random towers + low-rank router | 51.01 | 2,098 |
| clustered towers + centroid router (usual training-free router) | 170.2 | 39,098 |
| **clustered towers + low-rank router (this repo)** | **23.42** | **125.5** |
| per-neuron selection, low-rank predictor (not tower-shaped) | 18.93 | 29.62 |
| oracle router, random towers (upper bound) | 25.94 | 916.4 |
| oracle router, clustered towers (upper bound) | 17.37 | 59.67 |

At half the FFN neurons the tower MoE scores 23.42, against 33.04 for static pruning and 51.01 for random towers
with the same router. At a quarter it scores 125.5 against 353.0 for static pruning.

**What the numbers say, plainly:**

- **The clustering matters most.** With a perfect router, clustered towers reach 17.37 against 25.94 for random
  towers at 50%, and 59.67 against 916.4 at 25%.
- **The low-rank router beats the usual one by far.** The centroid router scores 170.2 on the same towers.
- **The router leaves quality on the table.** It reaches 23.42, against 17.37 for a perfect router on the same
  towers. Its 256 directions keep 72% of the input variance.
- **Picking single neurons is better than picking towers.** Per-neuron selection with the same predictor scores
  18.93 and 29.62. Towers only pay off if contiguous 192-neuron blocks make the sparse FFN faster than scattered
  neurons, and this repo doesn't measure speed.
- **25% is too aggressive.** Every training-free method falls apart there; 125.5 is far from usable.

## Limits

- This measures **quality, not speed**. Unchosen neurons are masked out of a full dense FFN; there is no sparse
  kernel here, so no latency or throughput numbers are claimed.
- The low-rank router costs about 4.9% of the FFN's multiply-adds; the centroid router about 0.2%.
- One model (Qwen3-8B), one dataset (WikiText-2), 20,480 evaluation tokens. No fine-tuning after conversion.

## Run it

```bash
pip install -r requirements.txt
python results/run_real.py      # writes results/real.json (~3 min on an RTX 3090 Ti)
python -m pytest -q tests       # clustering, routing and mask tests on a tiny Qwen3
```

The model path is set in `results/qcommon.py`. Full numbers: [RESULTS.md](RESULTS.md).
