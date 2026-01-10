# Final Report: Llama-3.1-8B

## Perplexity Results (WikiText-2)

| Method | PPL | Δ from Dense |
|--------|-----|--------------|
| Dense (Baseline) | 5.8464 | - |
| Magnitude 30% | 8.5034 | +45.45% |
| Wanda 30% | 6.3185 | +8.07% |
| Magnitude 50% | 37.8210 | +546.91% |
| Wanda 50% | 8.9966 | +53.88% |
| Magnitude 70% | 263197.6250 | +4501756.16% |
| Wanda 70% | 96.7458 | +1554.79% |

## Zero-Shot Accuracy

*Average across 5 tasks: piqa, hellaswag, arc_easy, boolq, rte*

| Method | Accuracy | Retention |
|--------|----------|-----------|
| Dense (Baseline) | 75.27% | 100.00% |
| Magnitude 30% | 71.74% | 95.31% |
| Wanda 30% | 74.49% | 98.97% |
| Magnitude 50% | 56.43% | 74.97% |
| Wanda 50% | 66.05% | 87.76% |
| Magnitude 70% | 40.28% | 53.52% |
| Wanda 70% | 42.42% | 56.36% |

## Key Findings


### 30% Sparsity

- **Perplexity Improvement:** Wanda improves by 25.69% over Magnitude
- **Accuracy Improvement:** Wanda improves by 2.76 percentage points

### 50% Sparsity

- **Perplexity Improvement:** Wanda improves by 76.21% over Magnitude
- **Accuracy Improvement:** Wanda improves by 9.63 percentage points

### 70% Sparsity

- **Perplexity Improvement:** Wanda improves by 99.96% over Magnitude
- **Accuracy Improvement:** Wanda improves by 2.14 percentage points

## Per-Task Zero-Shot Breakdown


### 30% Sparsity

| Task | Dense | Magnitude | Wanda |
|------|-------|-----------|-------|
| piqa | 79.27% | 76.61% | 78.35% |
| hellaswag | 60.67% | 56.37% | 59.55% |
| arc_easy | 82.20% | 78.37% | 80.51% |
| boolq | 83.09% | 80.18% | 82.94% |
| rte | 71.12% | 67.15% | 71.12% |

### 50% Sparsity

| Task | Dense | Magnitude | Wanda |
|------|-------|-----------|-------|
| piqa | 79.27% | 70.46% | 73.88% |
| hellaswag | 60.67% | 43.27% | 50.53% |
| arc_easy | 82.20% | 62.21% | 71.84% |
| boolq | 83.09% | 52.42% | 79.14% |
| rte | 71.12% | 53.79% | 54.87% |

### 70% Sparsity

| Task | Dense | Magnitude | Wanda |
|------|-------|-----------|-------|
| piqa | 79.27% | 54.73% | 55.17% |
| hellaswag | 60.67% | 26.44% | 27.48% |
| arc_easy | 82.20% | 29.71% | 32.32% |
| boolq | 83.09% | 37.83% | 44.43% |
| rte | 71.12% | 52.71% | 52.71% |
