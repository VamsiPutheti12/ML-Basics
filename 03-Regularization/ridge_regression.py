"""
Ridge Regression (L2 Regularization)
=====================================
Adds penalty on squared weights to prevent overfitting.
Loss = MSE + λ * Σ(w²)
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("RIDGE REGRESSION (L2 REGULARIZATION)")
print("=" * 50)

# ============================================
# Generate Multi-Feature Data
# ============================================
np.random.seed(42)
n_samples = 30
n_features = 5

# Create features with some being more important than others
X = np.random.randn(n_samples, n_features)
true_weights = np.array([3.0, 1.5, 0.5, 0.1, 0.0])  # Decreasing importance
y = X @ true_weights + np.random.randn(n_samples) * 0.5

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

print(f"\nSamples: {n_samples}, Features: {n_features}")
print(f"True weights: {true_weights}")

# ============================================
# Ridge Regression with Gradient Descent
# ============================================
def ridge_regression(X, y, lambda_reg, lr=0.1, iterations=1000):
    """Train ridge regression using gradient descent"""
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    
    for _ in range(iterations):
        y_pred = X @ w + b
        error = y_pred - y
        
        # Gradients with L2 penalty
        dw = (1/m) * (X.T @ error) + 2 * lambda_reg * w
        db = (1/m) * np.sum(error)
        
        w -= lr * dw
        b -= lr * db
    
    mse = np.mean((y - (X @ w + b))**2)
    return w, b, mse

# ============================================
# Train with Different Lambda Values
# ============================================
lambdas = [0, 0.01, 0.1, 1.0, 10.0]
results = {}

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"\n{'Lambda':<10} {'MSE':<10} {'Weights (rounded)'}")
print("-" * 60)

for lam in lambdas:
    w, b, mse = ridge_regression(X, y, lam)
    results[lam] = {'weights': w, 'mse': mse}
    w_str = ', '.join([f'{wi:.2f}' for wi in w])
    print(f"{lam:<10} {mse:<10.4f} [{w_str}]")

print(f"\n📊 As λ increases, weights shrink toward zero but never reach zero!")
print("   This is the key property of L2 regularization.")

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

ax1.set_xscale('log')
ax1.set_xlabel('Lambda (λ)', fontsize=12)
ax1.set_ylabel('Weight Value', fontsize=12)
ax1.set_title('Ridge: Coefficient Shrinkage', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Plot 2: Weight magnitudes comparison
ax2 = axes[1]
x_pos = np.arange(n_features)
width = 0.15

for idx, lam in enumerate([0, 0.1, 1.0, 10.0]):
    offset = (idx - 1.5) * width
    ax2.bar(x_pos + offset, np.abs(results[lam]['weights']), width, 
            label=f'λ={lam}', alpha=0.8)

ax2.set_xlabel('Feature', fontsize=12)
ax2.set_ylabel('|Weight|', fontsize=12)
ax2.set_title('Weight Magnitudes by Lambda', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'w{i+1}' for i in range(n_features)])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ridge_regression.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'ridge_regression.png'")
plt.show()

print(f"\n{'='*50}")
print("L2 REGULARIZATION MATH")
print(f"{'='*50}")
print("""
Loss = MSE + λ * Σ(w²)

Gradient: ∂L/∂w = ∂MSE/∂w + 2λw

Properties:
- Shrinks ALL weights toward zero
- Weights never become exactly zero
- Good when all features are useful
- Also called "Weight Decay"
""")
