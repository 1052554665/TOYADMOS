The original training epoch was set to 200 in `Config.py`, but it has been reduced to 20 for the current configuration.

# The ROC Curve

## ROC Curve Explained

### The Core Idea

An anomaly detection system outputs an **anomaly score** for each sound clip. You need to pick a **threshold** $\tau$:

- If score $> \tau$ → flag as **anomaly**
- If score $\leq \tau$ → flag as **normal**

The ROC curve answers: **"For every possible threshold, how well does the system separate normal from anomalous?"**

### The Two Axes

| Metric | Meaning in this code | Formula |
|--------|----------------------|---------|
| **FPR** (x-axis) | Fraction of **normal** sounds wrongly flagged as anomalous | $\frac{\text{# normal with score} > \tau}{\text{# total normal}}$ |
| **TPR** (y-axis) | Fraction of **anomalous** sounds correctly flagged | $\frac{\text{# anomalous with score} > \tau}{\text{# total anomalous}}$ |

### What Your Code Does Step by Step

```python
Thres = np.sort(MN)[::-1]    # ① Sort normal scores: highest → lowest
```

Each normal sample's score becomes a candidate threshold. Since `MN` has $N_{\text{normal}}$ entries, you get $N_{\text{normal}}$ thresholds. At the $j$-th threshold (the $j$-th largest normal score), exactly $j$ normals exceed it → $\text{FPR} = j / N_{\text{normal}}$.

```python
for jj in range( len(A_set) ):
    TPR[jj] = np.sum( MA > Thres[jj] ) / len(A_set)   # ② TPR at each threshold
```

For each threshold, count how many anomalous scores exceed it → that's the TPR.

### 🔴 Bug in the x-axis

```python
plt.plot( np.linspace(0, 1, len(TPR)), TPR )  # ← WRONG
```

`np.linspace(0, 1, len(TPR))` assumes FPR goes **uniformly** from 0 to 1. But FPR depends on how many normals exceed each threshold — it's **not** guaranteed to be uniform. The correct x-axis should be:

```python
plt.plot( np.arange(len(TPR)) / len(Thres), TPR )  # ← CORRECT: FPR = j / N_normal
```

### How to Interpret the Curve

```
TPR ↑
1.0 │        ●━━━━━━━━━  ← ideal: hugs top-left
    │      ●
    │    ●
    │  ●
    │ ●
    │●
0.0 └──────────────────→ FPR
    0.0              1.0
```

- **Perfect classifier**: curve goes straight up to (0,1), then flat right → AUC = 1.0
- **Random guessing**: diagonal line from (0,0) to (1,1) → AUC = 0.5
- **Better than random**: curve bows toward top-left → AUC > 0.5
- The red vertical line in your code marks $\text{FPR} = 10\%$ — you read the TPR at that operating point

### AUC (Area Under Curve)

$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$

Interpretation: **probability that a random anomalous sample scores higher than a random normal sample**. AUC = 0.5 is random; AUC = 1.0 is perfect separation.

---

## Summary of Issues in This Code Block

| # | Severity | Line | Issue |
|---|----------|------|-------|
| 1 | 🔴 | 166 | x-axis `np.linspace(...)` should be `np.arange(len(TPR)) / len(Thres)` |
| 2 | 🟡 | 162 | Loop bound `len(A_set)` vs `len(Thres)` — if anomalies > normals, IndexError; if fewer, unused thresholds |
| 3 | 🟡 | 172 | `np.mean(TPR)` is a very rough AUC approximation (Riemann sum with uneven x-spacing) — `sklearn.metrics.auc` would be more accurate |

# F-measure


## F-measure Explained (Lines 177–184)

### What It Computes

This is the **F1-score at a fixed operating point**: FPR = 10% ($\rho = 0.1$). It answers: *"If we tolerate 10% of normal sounds being falsely flagged, how good is the detector?"*

---

### Step-by-Step Breakdown

```
                    Thres (sorted high→low)
                    ┌─────────────────────┐
                    │ ████████████████████ │  N_N normal samples
                    │ ████████████████████ │  → N_N thresholds
                    └──┬──────────────────┘
                       │
              rho_index = N_N × 0.1
                       │
              ┌────────▼──────────────────┐
              │  10% FPR region  │  90%   │
              │  (ρ·N_N normals  │        │
              │   exceed this τ) │        │
              └───────────────────────────┘
```

| Variable | Formula | Meaning |
|----------|---------|---------|
| `rho_index` | $\lfloor N_{\text{normal}} \times 0.1 \rfloor$ | Index into sorted thresholds giving ~10% FPR |
| `TP` | $\text{TPR}[\text{rho\_index}] \times N_{\text{anomaly}}$ | **True Positives**: # anomalous correctly detected |
| `FP` | $\lfloor 0.1 \times N_{\text{normal}} \rfloor$ | **False Positives**: # normal wrongly flagged (= rho_index) |
| `FN` | $N_{\text{anomaly}} - \text{TP}$ | **False Negatives**: # anomalous missed |

Then:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \qquad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \text{TPR}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### Intuition

| Metric | Question Answered |
|--------|-------------------|
| **Precision** | *"Of all alarms raised, how many are real?"* |
| **Recall** (= TPR) | *"Of all real anomalies, how many did we catch?"* |
| **F1-score** | Harmonic mean — balances both; high only when **both** are high |

**Why harmonic mean?** It penalizes extreme imbalance. If Precision=1.0 but Recall=0.1, F1 ≈ 0.18 — exposing a system that's too conservative.

---

### Your Result

```
F-measure under FPR = 10.0% condition: 0.8927
```

At the operating point where 10% of normal sounds are falsely flagged, the detector achieves ~89% F1 — strong performance balancing both missed anomalies and false alarms.

---

### 🔍 Subtle Note on the Code

The code assumes $\text{FP} = \rho \cdot N_{\text{normal}}$ (an *exact* integer). This is **approximately correct** since the thresholds come from sorting normal scores — the $\rho$-th percentile threshold by construction has $\approx \rho \cdot N$ normals above it. The approximation is:

$$\text{FP} = \texttt{int}(0.1 \times N_{\text{normal}}) \quad \text{vs. actual} \quad \text{FP} = \texttt{rho\_index}$$

These are identical when `int(rho * N_N) == int(rho * N_N)` so it's consistent — not a bug, just worth noting.## F-measure Explained (Lines 177–184)
