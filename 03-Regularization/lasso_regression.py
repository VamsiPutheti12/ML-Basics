"""
Lasso Regression (L1 Regularization)
======================================
Adds penalty on absolute weights - can set weights to exactly zero!
Loss = MSE + λ * Σ|w|
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("LASSO REGRESSION (L1 REGULARIZATION)")
print("=" * 50)

# ============================================
# Generate Multi-Feature Data
# ============================================
np.random.seed(42)
n_samples = 30
n_features = 5

# Create features with some being more important than others
X = np.random.randn(n_samples, n_features)
true_weights = np.array([3.0, 1.5, 0.5, 0.1, 0.0])  # Feature 5 is irrelevant!
y = X @ true_weights + np.random.randn(n_samples) * 0.5

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"\nSamples: {n_samples}, Features: {n_features}")
print(f"True weights: {true_weights}")
print(f"Note: Feature 5 has weight = 0 (irrelevant)")

# ============================================
# Lasso with Coordinate Descent
# ============================================
def soft_threshold(x, threshold):
    """Soft thresholding operator for L1"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def lasso_regression(X, y, lambda_reg, iterations=1000):
    """Train lasso using coordinate descent"""
    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(iterations):
        for j in range(n):
            # Compute residual without feature j
            residual = y - X @ w + X[:, j] * w[j]
            
            # Update weight j
            rho = X[:, j] @ residual
            w[j] = soft_threshold(rho, lambda_reg * m) / (X[:, j] @ X[:, j])
    
    mse = np.mean((y - X @ w)**2)
    return w, mse

# ============================================
# Train with Different Lambda Values
# ============================================
lambdas = [0, 0.01, 0.05, 0.1, 0.5]
results = {}

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"\n{'Lambda':<10} {'MSE':<10} {'Weights (zeros highlighted)'}")
print("-" * 65)

for lam in lambdas:
    w, mse = lasso_regression(X, y, lam)
    results[lam] = {'weights': w, 'mse': mse}
    
    # Highlight zeros
    w_parts = []
    for wi in w:
        if abs(wi) < 0.01:
            w_parts.append("  0  ")
        else:
            w_parts.append(f"{wi:5.2f}")
    w_str = ', '.join(w_parts)
    
    zeros = np.sum(np.abs(w) < 0.01)
    print(f"{lam:<10} {mse:<10.4f} [{w_str}] ({zeros} zeros)")

print(f"\n🎯 As λ increases, some weights become EXACTLY ZERO!")
print("   This is automatic FEATURE SELECTION - L1's superpower!")

# ============================================
# Visualization
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Coefficient paths
ax1 = axes[0]
weight_matrix = np.array([results[lam]['weights'] for lam in lambdas])
for i in range(n_features):
    ax1.plot(lambdas, weight_matrix[:, i], 'o-', linewidth=2, 
             label=f'w{i+1} (true={true_weights[i]})')

ax1.set_xlabel('Lambda (λ)', fontsize=12)
ax1.set_ylabel('Weight Value', fontsize=12)
ax1.set_title('Lasso: Coefficients Go to Zero', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 2: Number of non-zero features
ax2 = axes[1]
non_zeros = [np.sum(np.abs(results[lam]['weights']) >= 0.01) for lam in lambdas]
ax2.bar(range(len(lambdas)), non_zeros, color='steelblue', edgecolor='black')
ax2.set_xticks(range(len(lambdas)))
ax2.set_xticklabels([str(l) for l in lambdas])
ax2.set_xlabel('Lambda (λ)', fontsize=12)
ax2.set_ylabel('Number of Non-Zero Weights', fontsize=12)
ax2.set_title('Feature Selection with Lasso', fontsize=13, fontweight='bold')
ax2.set_ylim(0, n_features + 0.5)
ax2.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for i, count in enumerate(non_zeros):
    ax2.text(i, count + 0.1, str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('lasso_regression.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'lasso_regression.png'")
plt.show()

print(f"\n{'='*50}")
print("L1 REGULARIZATION MATH")
print(f"{'='*50}")
print("""
Loss = MSE + λ * Σ|w|

Uses soft-thresholding:
w = sign(w) * max(|w| - λ, 0)

Properties:
- Can set weights to EXACTLY ZERO
- Automatic feature selection
- Creates SPARSE models
- Good when many features are irrelevant
""")
