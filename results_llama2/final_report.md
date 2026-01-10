# Final Report: Llama-2-7b-hf

## Perplexity Results (WikiText-2)

| Method | PPL | Δ from Dense |
|--------|-----|--------------|
| Dense (Baseline) | 5.1140 | - |
| Magnitude 30% | 5.7350 | +12.14% |
| Wanda 30% | 5.5244 | +8.02% |
| Magnitude 50% | 11.8324 | +131.37% |
| Wanda 50% | 6.4154 | +25.45% |
| Magnitude 70% | nan | +nan% |
| Wanda 70% | 68.8045 | +1245.40% |

## Zero-Shot Accuracy

*Average across 5 tasks: piqa, hellaswag, arc_easy, boolq, rte*

| Method | Accuracy | Retention |
|--------|----------|-----------|
| Dense (Baseline) | 70.78% | 100.00% |
| Magnitude 30% | 69.09% | 97.61% |
| Wanda 30% | 69.08% | 97.60% |
| Magnitude 50% | 62.26% | 87.96% |
| Wanda 50% | 66.21% | 93.54% |
| Magnitude 70% | 38.82% | 54.85% |
| Wanda 70% | 42.28% | 59.74% |

## Key Findings


### 30% Sparsity

- **Perplexity Improvement:** Wanda improves by 3.67% over Magnitude
- **Accuracy Improvement:** Wanda improves by -0.01 percentage points

### 50% Sparsity

- **Perplexity Improvement:** Wanda improves by 45.78% over Magnitude
- **Accuracy Improvement:** Wanda improves by 3.95 percentage points

### 70% Sparsity

- **Perplexity Improvement:** Wanda improves by nan% over Magnitude
- **Accuracy Improvement:** Wanda improves by 3.46 percentage points

## Per-Task Zero-Shot Breakdown


### 30% Sparsity

| Task | Dense | Magnitude | Wanda |
|------|-------|-----------|-------|
| piqa | 78.13% | 78.40% | 78.02% |
| hellaswag | 57.10% | 58.13% | 56.56% |
| arc_easy | 75.46% | 75.97% | 75.59% |
| boolq | 79.33% | 75.20% | 78.56% |
| rte | 63.90% | 57.76% | 56.68% |

### 50% Sparsity

| Task | Dense | Magnitude | Wanda |
|------|-------|-----------|-------|
| piqa | 78.13% | 74.59% | 76.06% |
| hellaswag | 57.10% | 53.09% | 51.82% |
| arc_easy | 75.46% | 67.85% | 72.56% |
| boolq | 79.33% | 63.43% | 76.09% |
| rte | 63.90% | 52.35% | 54.51% |

### 70% Sparsity

| Task | Dense | Magnitude | Wanda |
|------|-------|-----------|-------|
| piqa | 78.13% | 52.77% | 54.24% |
| hellaswag | 57.10% | 25.87% | 27.89% |
| arc_easy | 75.46% | 25.63% | 29.55% |
| boolq | 79.33% | 37.86% | 47.03% |
| rte | 63.90% | 51.99% | 52.71% |
