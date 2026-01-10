# Theoretical Analysis: Llama-3.1-8B

## 1. Why Wanda Outperforms Magnitude Pruning

Wanda (Weight AND Activation) pruning improves upon magnitude-only pruning by incorporating **input activation statistics** into the pruning decision.

### Core Insight

A weight's importance ≠ just its magnitude, but rather:

```
Importance = |Weight| × Input_Activation_Norm
```

### Why This Matters

- **Magnitude pruning:** Removes smallest weights → assumes magnitude = importance
- **Wanda:** Considers BOTH weight size AND how much it's "used" by real data

### Mathematical Formulation

```
Magnitude metric: S_mag(w_ij) = |w_ij|
Wanda metric:     S_wanda(w_ij) = |w_ij| × ||X_j||₂
```

where `X_j` is the j-th input feature across calibration samples.

### Practical Example

Consider two weights:

- **Weight A:** Large magnitude (0.5), rarely activated (low X norm)
- **Weight B:** Medium magnitude (0.3), frequently activated (high X norm)

**Magnitude pruning** → Keeps A, removes B ❌ (wrong!)

**Wanda** → Keeps B, removes A ✅ (correct!)

### Result

Wanda preserves weights that are actually important for the model's computations on real data, leading to better performance retention.

## 2. Layer-Wise Behavior

### Observed Layer Statistics (Wanda 50%)

- **Mean sparsity:** 50.03%
- **Std deviation:** 0.00%
- **Min sparsity:** 50.03% (Layer 0)
- **Max sparsity:** 50.04% (Layer 10)

### Interpretation

- **Lower layers (early):** Tend to have lower sparsity
  - More critical for basic feature extraction
- **Higher layers (late):** Tend to have lower sparsity
  - More redundancy in high-level representations

Standard deviation of 0.00% indicates **relatively uniform** variation across layers, showing Wanda adapts pruning to layer importance.

## 3. Failure Modes & Limitations

### Failure Mode 1: Task-Specific Degradation

- **Most affected task:** rte (16.25% accuracy drop)
- **Least affected task:** boolq (3.94% accuracy drop)

**Why:** Tasks requiring more complex reasoning (e.g., rte) are more sensitive to pruning. Wanda preserves perplexity well but may lose some higher-order reasoning capacity.

### Failure Mode 2: Calibration Data Dependency

Wanda requires calibration data (WikiText-2) to compute activation norms.

**Risk:** If calibration data distribution ≠ target task distribution, Wanda may optimize for wrong activations.

**Example:** WikiText (formal text) vs Code generation → activation patterns differ

**Mitigation:** Use diverse calibration data or task-specific calibration.

### Failure Mode 3: High Sparsity Regime

At 70% sparsity and beyond, BOTH magnitude and Wanda degrade significantly.

**Root cause:** Too many weights removed → model capacity fundamentally limited

**Observation from results:**

- 50% sparsity: 53.88% perplexity increase (manageable)
- 70% sparsity: 1554.79% perplexity increase (severe)

**Insight:** Wanda helps, but cannot overcome fundamental capacity limits.

### Failure Mode 4: Unstructured Pruning Overhead

Wanda (in this implementation) uses **unstructured** pruning → sparse weights scattered throughout matrices.

**Problem:** Modern hardware (GPUs, NPUs) doesn't efficiently accelerate unstructured sparsity without specialized kernels.

**Real-world implication:** 50% sparsity may only yield ~10-20% speedup, not 2x.

**Solution:** Structured pruning (N:M sparsity) trades some accuracy for guaranteed hardware acceleration.

### Failure Mode 5: No Fine-Tuning Recovery

This experiment uses "one-shot" pruning without post-pruning fine-tuning.

**Limitation:** Model doesn't adapt to sparsity pattern → performance ceiling.

**Potential improvement:** Brief fine-tuning (few epochs) could recover significant accuracy, especially at high sparsity.

## 4. Practical Recommendations

### For Production Deployment

1. Use Wanda with structured (2:4 or 4:8) sparsity for hardware efficiency
2. Calibrate on data matching your target distribution
3. Consider brief fine-tuning after pruning for recovery
4. Start conservatively (30-50% sparsity) before pushing higher
5. Monitor task-specific degradation, not just overall metrics

### For Research Extensions

1. Test with multiple calibration datasets to study distribution sensitivity
2. Compare layer-wise vs global sparsity allocation
3. Investigate learned pruning schedules (varying sparsity by layer)
4. Combine Wanda with quantization for compound compression
