# Gaussian Process Regression & Classification
### Implementation of Rasmussen & Williams Algorithms 2.1, 3.1, and 3.2

A from-scratch implementation of Gaussian Process (GP) models for regression and binary classification, following the algorithms laid out in *Gaussian Processes for Machine Learning* by Rasmussen & Williams (2006).

---

## Overview

This project implements three core GP algorithms:

- **Algorithm 2.1** — GP Regression with predictive mean and variance computation
- **Algorithm 3.1** — Laplace Approximation for GP Classification (Newton's method to find the posterior mode)
- **Algorithm 3.2** — Averaged Predictive Class Probability for binary GP Classification

All algorithms are implemented using NumPy/SciPy — no GP libraries (e.g., GPyTorch or GPflow) are used.

---

## Project Structure

```
.
├── AML_algo2_1_gp_regression.ipynb     # Algorithm 2.1: GP Regression
├── AML_Algo_3_1_and_3_2.ipynb          # Algorithms 3.1 & 3.2: GP Classification
├── AML_algo2_1_regression_dataset.csv  # 1D regression dataset (sin + noise)
├── AML_algo2_1_toy2D_dataset.csv       # Toy 2D training dataset
└── README.md
```

---

## Algorithm 2.1 — GP Regression

### What it does

Given training data `(X_train, y_train)` and a test input `X_test`, Algorithm 2.1 computes:

- **Predictive mean** `f̄(x*)` — the GP's best estimate at a test point
- **Predictive variance** `𝕍[f(x*)]` — quantifying uncertainty around that estimate

### Key Steps

1. Compute the training covariance matrix `K(X, X)` using the RBF (squared exponential) kernel
2. Add noise variance to the diagonal: `K_y = K + σ_n² I`
3. Cholesky decompose `K_y = L Lᵀ`
4. Solve for `α = L⁻ᵀ L⁻¹ y`
5. Compute cross-covariance `K(X, x*)` between training and test points
6. Predictive mean: `f̄* = K(X, x*)ᵀ α`
7. Solve `v = L⁻¹ K(X, x*)` and compute variance: `𝕍[f*] = k(x*, x*) − vᵀv`

### RBF Kernel

The squared exponential (RBF) kernel is used throughout:

```
k(x, x') = σ_f² · exp(−|x − x'|² / 2ℓ²)
```

where `ℓ` is the length-scale and `σ_f` is the signal variance.

### Datasets

**1D Regression Dataset** (`AML_algo2_1_regression_dataset.csv`):  
20 training points sampled from `y = sin(x) + ε`, where `ε ~ N(0, 0.3²)`, over `x ∈ [0, 10]`.

**Toy 2D Dataset** (`AML_algo2_1_toy2D_dataset.csv`):  
5 hand-crafted training points in 2D space `(x₁, x₂)` used to visualize the covariance structure for a 2D test input.

### Outputs

- GP regression curve with 95% confidence interval (mean ± 2σ)
- Covariance matrix heatmap for the 2D toy dataset, showing how correlated training points are with each other and with the test input

---

## Algorithms 3.1 & 3.2 — Binary GP Classification

### Dataset

The **Breast Cancer Wisconsin dataset** (from `sklearn.datasets`) is used — a standard binary classification benchmark with 569 samples and 30 features. Labels are:
- `1` → Malignant
- `0` → Benign

150 training samples and 114 test samples are used (after a stratified 80/20 split). All features are standardized with `StandardScaler`.

### Algorithm 3.1 — Laplace Approximation (Newton's Method)

GP classification requires approximating a non-Gaussian posterior. Algorithm 3.1 finds the **mode of the posterior** (the MAP estimate of the latent function `f`) using Newton's method with the Laplace approximation.

**Steps:**

1. Initialize latent function `f = 0`
2. Iteratively update using Newton steps:
   - Compute likelihood derivatives: `π = σ(f)`, `W = diag(π(1−π))`, `∇ = y − π`
   - Form `B = I + W^{1/2} K W^{1/2}`, Cholesky decompose `B = LLᵀ`
   - Update: `a = b − W^{1/2} L⁻ᵀ L⁻¹ (W^{1/2} K b)`, where `b = Wf + ∇`
   - Set `f_new = Ka`
3. Repeat until `‖f_new − f‖ < tol` (convergence)

The sigmoid (logistic) function is used as the likelihood: `p(y=1 | f) = σ(f) = 1/(1 + e^{−f})`.

### Algorithm 3.2 — Predictive Class Probability

Given the posterior mode `f̂` and Cholesky factor `L` from Algorithm 3.1, Algorithm 3.2 computes the **averaged predictive probability** at test inputs.

**Steps:**

1. Compute predictive mean: `μ* = K(X*, X) · a`
2. Compute predictive variance: `𝕍* = k(x*, x*) − vᵀv`, where `v = L⁻¹ (W^{1/2} K(X*, X)ᵀ)`
3. Draw `S` samples from `N(μ*, 𝕍*)` and average through the sigmoid:
   `p̄(y*=1 | x*) ≈ (1/S) Σ σ(f_s)`

This Monte Carlo integral over the Gaussian posterior approximation handles the fact that `∫ σ(f) N(f; μ*, 𝕍*) df` has no closed form.

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~91% |
| AUC-ROC | ~0.97 |

---

## Understanding Predictive Uncertainty

One of the key strengths of GP models is that they provide **calibrated uncertainty estimates** alongside predictions — not just a class label or a point estimate.

### Where uncertainty is high

- **Near the decision boundary** (`f̂ ≈ 0`): the model is uncertain whether a sample is malignant or benign, producing `p̄ ≈ 0.5`.
- **Far from training data**: test points in regions of feature space not covered by training samples receive high predictive variance. The GP reverts toward its prior, reflecting genuine ignorance.
- **Overlapping classes**: even with enough training data, if two classes share similar feature distributions in some subregion, the latent function `f` will be small there, resulting in high uncertainty.

### Where uncertainty is low

- **Confidently classified samples**: points far from the decision boundary (large `|f̂|`) receive `p̄` close to 0 or 1, and low predictive standard deviation. The model is sure.
- **Dense training regions**: when test points fall near many training points, the posterior variance is pulled down by the covariance structure of the kernel.

### Why this matters

Unlike a logistic regression or decision tree that gives only a hard prediction, the GP yields a full predictive distribution. A model that says "I'm 51% confident this is malignant" should be treated very differently from one that says "I'm 99% confident." This is especially critical in medical applications where **acting on uncertain predictions** carries real risk.

The histogram of predictive standard deviations and the predictive probability plot both reveal that the model correctly identifies borderline cases — a property that makes GPs particularly valuable in safety-sensitive domains.

---

## Setup & Requirements

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

Then open either notebook in Jupyter or Google Colab and run all cells.

---

## Reference

Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.  
Available free online: [gaussianprocess.org/gpml](http://www.gaussianprocess.org/gpml/)
