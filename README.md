# Multi-Tower Mixture of Experts (MoE)

A flexible implementation of the Multi-Tower MoE architecture for scalable language models. This project features multiple tower configurations, dynamic routing, and efficient parameter utilization.

## Architecture

The MoE layer implements a sparse mixture of experts where input tokens are routed to the most relevant expert network. Key features:

- Multi-tower configuration: Support for heterogeneous expert blocks
- Soft/Hard routing selection
- Token-level load balancing
- Dropout and regularization strategies

## How to Run

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Run training:
   ```bash
   python src/moe/train.py
   ```

## Project Structure

- `src/moe/` Core MoE layer and training logic
- `tests/` Unit tests for routing and experts
