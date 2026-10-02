# Results

All numbers come from `results/run_real.py`, stored in `results/real.json`.

## Setup

- **Model:** Qwen3-8B (bf16), 36 layers, 12,288 FFN neurons per layer, on one RTX 3090 Ti.
- **Calibration:** WikiText-2 train, 16 windows × 512 tokens (49 s).
- **Evaluation:** WikiText-2 test, 40 windows × 512 tokens. This is the same evaluation set as the other repos
  in this portfolio.
- **Towers:** 64 per layer, 192 neurons each.
- **Low-rank router:** rank 256, which keeps 72.2% of the FFN input variance on average.
- **Router cost:** low-rank 4.86% of the FFN's multiply-adds; centroid 0.17%.
- **Selection:** each token keeps the top 50% or 25% of towers (or neurons) by score, in every FFN.

## Perplexity

| Method | keep | ppl |
|---|---:|---:|
| dense | 100% | 12.0346 |
| static | 50% | 33.0416 |
| random-lowrank | 50% | 51.0126 |
| cluster-centroid | 50% | 170.2268 |
| **cluster-lowrank** | 50% | **23.4189** |
| neuron-lowrank | 50% | 18.9267 |
| random-oracle | 50% | 25.9403 |
| cluster-oracle | 50% | 17.3696 |
| static | 25% | 352.9988 |
| random-lowrank | 25% | 2098.2351 |
| cluster-centroid | 25% | 39098.1562 |
| **cluster-lowrank** | 25% | **125.5188** |
| neuron-lowrank | 25% | 29.6243 |
| random-oracle | 25% | 916.4255 |
| cluster-oracle | 25% | 59.6713 |

## Method names

- **static:** the same neurons for every token, ranked by mean |activation| × ‖down column‖ on calibration data.
- **random-\* / cluster-\*:** towers made by a random split, or by balanced k-means on co-activation (ours).
- **\*-lowrank:** router scores from the layer's own gate/up weights in the top-256 principal input subspace
  (ours).
- **\*-centroid:** a tower's score is the input dotted with the mean of its normalised gate rows.
- **\*-oracle:** exact tower contributions from the full FFN. This is an upper bound, not a usable method.
- **neuron-lowrank:** the same low-rank predictor choosing single neurons instead of towers.

## Reading

- **Against static pruning,** the tower MoE wins at both budgets: 23.42 vs 33.04 at 50%, and 125.5 vs 353.0 at
  25%.
- **Against random towers with the same router,** it wins: 23.42 vs 51.01.
- **It loses to per-neuron selection** (18.93), and its router sits 6.05 ppl above the oracle on the same towers.
- **Next steps:** a higher-rank or per-layer-rank router to close the oracle gap, and a block-sparse FFN kernel
  to test whether towers are faster than scattered neurons in practice.
