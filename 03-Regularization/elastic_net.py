"""
Elastic Net Regression
=======================
Combines L1 (Lasso) and L2 (Ridge) regularization.
Loss = MSE + λ₁*Σ|w| + λ₂*Σ(w²)
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("ELASTIC NET REGRESSION")
print("=" * 50)

# ============================================
# Generate Multi-Feature Data
# ============================================
np.random.seed(42)
n_samples = 30
n_features = 5

X = np.random.randn(n_samples, n_features)
true_weights = np.array([3.0, 1.5, 0.5, 0.1, 0.0])
y = X @ true_weights + np.random.randn(n_samples) * 0.5

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"\nSamples: {n_samples}, Features: {n_features}")
print(f"True weights: {true_weights}")

# ============================================
# Elastic Net with Coordinate Descent
# ============================================
def elastic_net(X, y, l1_ratio, lambda_reg, iterations=1000):
    """
    l1_ratio: 0 = pure Ridge, 1 = pure Lasso
    lambda_reg: overall regularization strength
    """
    m, n = X.shape
    w = np.zeros(n)
    
    lambda_l1 = l1_ratio * lambda_reg
    lambda_l2 = (1 - l1_ratio) * lambda_reg
    
    for _ in range(iterations):
        for j in range(n):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual
            
            # Soft thresholding with L2 in denominator
            z = X[:, j] @ X[:, j] + 2 * lambda_l2 * m
            w[j] = np.sign(rho) * max(abs(rho) - lambda_l1 * m, 0) / z
    
    mse = np.mean((y - X @ w)**2)
    return w, mse

# ============================================
# Compare Different Ratios
# ============================================
lambda_reg = 0.1
l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
results = {}

print(f"\n{'='*50}")
print(f"RESULTS (λ = {lambda_reg})")
print(f"{'='*50}")
print(f"\n{'L1 Ratio':<12} {'Type':<15} {'MSE':<10} {'Zeros':<8} {'Weights'}")
print("-" * 75)

for ratio in l1_ratios:
    w, mse = elastic_net(X, y, ratio, lambda_reg)
    results[ratio] = {'weights': w, 'mse': mse}
    
    zeros = np.sum(np.abs(w) < 0.01)
    w_str = ', '.join([f'{wi:.2f}' for wi in w])
    
    if ratio == 0:
        type_name = "Pure Ridge"
    elif ratio == 1:
        type_name = "Pure Lasso"
    else:
        type_name = "Elastic Net"
    
    print(f"{ratio:<12} {type_name:<15} {mse:<10.4f} {zeros:<8} [{w_str}]")

print(f"\n💡 Elastic Net gives you the best of both worlds:")
print("   - Sparsity from L1 (feature selection)")
print("   - Stability from L2 (grouped features)")

# ============================================
# Visualization
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Weight values by ratio
ax1 = axes[0]
weight_matrix = np.array([results[r]['weights'] for r in l1_ratios])
x_pos = np.arange(n_features)
width = 0.15

for idx, ratio in enumerate(l1_ratios):
    offset = (idx - 2) * width
    ax1.bar(x_pos + offset, results[ratio]['weights'], width, 
            label=f'L1={ratio}', alpha=0.8)

ax1.set_xlabel('Feature', fontsize=12)
ax1.set_ylabel('Weight Value', fontsize=12)
ax1.set_title('Elastic Net: Effect of L1/L2 Ratio', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'w{i+1}' for i in range(n_features)])
ax1.legend(title='L1 Ratio')
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Plot 2: Sparsity vs Ratio
ax2 = axes[1]
non_zeros = [np.sum(np.abs(results[r]['weights']) >= 0.01) for r in l1_ratios]
colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22', '#e74c3c']

bars = ax2.bar(range(len(l1_ratios)), non_zeros, color=colors, edgecolor='black')
ax2.set_xticks(range(len(l1_ratios)))
ax2.set_xticklabels(['0\n(Ridge)', '0.25', '0.5', '0.75', '1\n(Lasso)'])
ax2.set_xlabel('L1 Ratio', fontsize=12)
ax2.set_ylabel('Non-Zero Weights', fontsize=12)
ax2.set_title('Sparsity Increases with L1 Ratio', fontsize=13, fontweight='bold')
ax2.set_ylim(0, n_features + 0.5)
ax2.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, non_zeros):
    ax2.text(bar.get_x() + bar.get_width()/2, count + 0.1, 
             str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('elastic_net.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'elastic_net.png'")
plt.show()

print(f"\n{'='*50}")
print("ELASTIC NET MATH")
print(f"{'='*50}")
print("""
Loss = MSE + λ₁*Σ|w| + λ₂*Σ(w²)

Or equivalently with ratio α:
Loss = MSE + λ * [α*Σ|w| + (1-α)*Σ(w²)]

Properties:
- α = 0: Pure Ridge (L2)
- α = 1: Pure Lasso (L1)
- 0 < α < 1: Best of both

When to use:
- Many correlated features (Ridge struggles)
- Want feature selection (Lasso too aggressive)
- Default: start with α = 0.5
""")
