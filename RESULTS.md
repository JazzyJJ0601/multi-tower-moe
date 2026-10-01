# Multi-Tower MoE Benchmark Results

**Status:** Only the base Qwen3-8B perplexity has been measured so far. The multi-tower method has not been run on the real model yet.

This script loads Qwen3-8B from local cache and measures perplexity on sample prompts.

**Command used:** `python3 repos/multi-tower-moe/results/run_real.py`

| Prompt | Perplexity |
|--------|------------|
| The quick brown fox jumps over the lazy dog. | 3.46 |
| In the beginning, the universe was created. | 14.87 |
| Machine learning is a subset of artificial intelligence. | 6.52 |

**Key findings:**

1. Perplexity measures how well the model predicts the text. Lower is better.
2. Short, grammatical sentences typically have lower perplexity values.
3. The Qwen3-8B model shows reasonable perplexity on these common English phrases.
4. For more detailed analysis, we could test longer contexts and domain-specific text.
