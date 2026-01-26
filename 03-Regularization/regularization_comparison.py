"""
Regularization Comparison
==========================
Side-by-side comparison: No Reg vs Ridge vs Lasso vs Elastic Net
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("REGULARIZATION COMPARISON")
print("=" * 60)

# ============================================
# Generate Data
# ============================================
np.random.seed(42)
n_samples = 30
n_features = 5

X = np.random.randn(n_samples, n_features)
true_weights = np.array([3.0, 1.5, 0.5, 0.1, 0.0])
y = X @ true_weights + np.random.randn(n_samples) * 0.5

X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"\nData: {n_samples} samples, {n_features} features")
print(f"True weights: {true_weights}")

# ============================================
# Implementation Functions
# ============================================
def no_regularization(X, y, lr=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    for _ in range(iterations):
        y_pred = X @ w
        dw = (1/m) * (X.T @ (y_pred - y))
        w -= lr * dw
    return w, np.mean((y - X @ w)**2)

def ridge(X, y, lam=0.1, lr=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    for _ in range(iterations):
        y_pred = X @ w
        dw = (1/m) * (X.T @ (y_pred - y)) + 2 * lam * w
        w -= lr * dw
    return w, np.mean((y - X @ w)**2)

def soft_threshold(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

def lasso(X, y, lam=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    for _ in range(iterations):
        for j in range(n):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual
            w[j] = soft_threshold(rho, lam * m) / (X[:, j] @ X[:, j])
    return w, np.mean((y - X @ w)**2)

def elastic_net(X, y, l1_ratio=0.5, lam=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    l1 = l1_ratio * lam
    l2 = (1 - l1_ratio) * lam
    for _ in range(iterations):
        for j in range(n):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual
            z = X[:, j] @ X[:, j] + 2 * l2 * m
            w[j] = np.sign(rho) * max(abs(rho) - l1 * m, 0) / z
    return w, np.mean((y - X @ w)**2)

# ============================================
# Run All Methods
# ============================================
lam = 0.1
results = {
    'No Reg': no_regularization(X, y),
    'Ridge (L2)': ridge(X, y, lam),
    'Lasso (L1)': lasso(X, y, lam),
    'Elastic Net': elastic_net(X, y, 0.5, lam)
}

print(f"\n{'='*60}")
print(f"RESULTS (λ = {lam})")
print(f"{'='*60}")
print(f"\n{'Method':<15} {'MSE':<10} {'Non-Zero':<10} {'Weights'}")
print("-" * 70)

for name, (w, mse) in results.items():
    non_zero = np.sum(np.abs(w) >= 0.01)
    w_str = ', '.join([f'{wi:5.2f}' for wi in w])
    print(f"{name:<15} {mse:<10.4f} {non_zero:<10} [{w_str}]")

print(f"\nTrue weights:                          [{', '.join([f'{wi:5.2f}' for wi in true_weights])}]")

# ============================================
# Visualization
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Weight comparison
ax1 = axes[0]
methods = list(results.keys())
x_pos = np.arange(n_features)
width = 0.2
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

for idx, (name, (w, mse)) in enumerate(results.items()):
    offset = (idx - 1.5) * width
    ax1.bar(x_pos + offset, w, width, label=name, color=colors[idx], alpha=0.8)

# Add true weights as line
for i, tw in enumerate(true_weights):
    ax1.hlines(tw, i - 0.4, i + 0.4, colors='black', linestyles='--', linewidth=2)

ax1.set_xlabel('Feature', fontsize=12)
ax1.set_ylabel('Weight Value', fontsize=12)
ax1.set_title('Learned Weights by Method', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'w{i+1}\n(true={true_weights[i]})' for i in range(n_features)])
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Summary metrics
ax2 = axes[1]
method_names = list(results.keys())
mse_values = [results[m][1] for m in method_names]
non_zeros = [np.sum(np.abs(results[m][0]) >= 0.01) for m in method_names]

x = np.arange(len(method_names))
width = 0.35

bars1 = ax2.bar(x - width/2, mse_values, width, label='MSE', color='#3498db')
ax2.set_ylabel('MSE', fontsize=12, color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')

ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, non_zeros, width, label='Non-Zero', color='#e74c3c')
ax2_twin.set_ylabel('Non-Zero Weights', fontsize=12, color='#e74c3c')
ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax2_twin.set_ylim(0, n_features + 1)

ax2.set_xlabel('Method', fontsize=12)
ax2.set_title('MSE vs Sparsity', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(method_names, rotation=15)
ax2.grid(True, alpha=0.3, axis='y')

fig.legend([bars1, bars2], ['MSE', 'Non-Zero Weights'], loc='upper center', 
           ncol=2, bbox_to_anchor=(0.75, 0.95))

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'regularization_comparison.png'")
plt.show()

print(f"\n{'='*60}")
print("SUMMARY: WHEN TO USE WHAT")
print(f"{'='*60}")
print("""
┌─────────────┬────────────────────┬─────────────────────────┐
│ Method      │ Best For           │ Key Property            │
├─────────────┼────────────────────┼─────────────────────────┤
│ No Reg      │ Small, clean data  │ May overfit             │
│ Ridge (L2)  │ All features useful│ Shrinks, never zeros    │
│ Lasso (L1)  │ Feature selection  │ Sets weights to zero    │
│ Elastic Net │ Correlated features│ Best of both worlds     │
└─────────────┴────────────────────┴─────────────────────────┘
""")
