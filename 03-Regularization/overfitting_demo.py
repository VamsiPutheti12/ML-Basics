"""
Overfitting Demo
=================
When a model is too complex and memorizes noise instead of learning patterns.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("OVERFITTING DEMO")
print("=" * 50)

# ============================================
# Generate Simple Data with Noise
# ============================================
np.random.seed(42)
n_samples = 20

X = np.linspace(0, 1, n_samples)
y_true = np.sin(2 * np.pi * X)  # True pattern
y = y_true + np.random.randn(n_samples) * 0.3  # Add noise

# Split into train/test
X_train, X_test = X[:15], X[15:]
y_train, y_test = y[:15], y[15:]

print(f"\nData: y = sin(2πx) + noise")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================
# Fit Polynomial Models of Different Degrees
# ============================================
def fit_polynomial(X_train, y_train, X_test, degree):
    """Fit polynomial and return predictions + errors"""
    # Create polynomial features
    X_poly_train = np.column_stack([X_train**i for i in range(degree+1)])
    X_poly_test = np.column_stack([X_test**i for i in range(degree+1)])
    
    # Fit using normal equation
    weights = np.linalg.lstsq(X_poly_train, y_train, rcond=None)[0]
    
    # Predictions
    y_train_pred = X_poly_train @ weights
    y_test_pred = X_poly_test @ weights
    
    # Errors
    train_mse = np.mean((y_train - y_train_pred)**2)
    test_mse = np.mean((y_test - y_test_pred)**2)
    
    return weights, train_mse, test_mse

# Fit different degrees
degrees = [1, 3, 15]
results = {}

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"\n{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Status'}")
print("-" * 55)

for deg in degrees:
    weights, train_mse, test_mse = fit_polynomial(X_train, y_train, y_test, deg)
    results[deg] = {'weights': weights, 'train_mse': train_mse, 'test_mse': test_mse}
    
    if deg == 1:
        status = "UNDERFIT"
    elif deg == 3:
        status = "GOOD FIT"
    else:
        status = "OVERFIT"
    
    print(f"{deg:<10} {train_mse:<15.4f} {test_mse:<15.4f} {status}")

print(f"\n⚠️  Degree 15 has LOW train error but HIGH test error!")
print("   This is OVERFITTING - model memorized the noise.")

# ============================================
# Visualization
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
titles = ['❌ UNDERFIT (Degree 1)', '✓ GOOD FIT (Degree 3)', '❌ OVERFIT (Degree 15)']
colors = ['red', 'green', 'purple']

X_smooth = np.linspace(0, 1, 100)

for idx, deg in enumerate(degrees):
    ax = axes[idx]
    weights = results[deg]['weights']
    
    # Predict on smooth curve
    X_poly_smooth = np.column_stack([X_smooth**i for i in range(deg+1)])
    y_smooth = X_poly_smooth @ weights
    
    # Plot
    ax.scatter(X_train, y_train, color='blue', s=60, alpha=0.7, label='Train', zorder=3)
    ax.scatter(X_test, y_test, color='orange', s=80, marker='s', alpha=0.7, label='Test', zorder=3)
    ax.plot(X_smooth, y_smooth, color=colors[idx], linewidth=2, label=f'Degree {deg}')
    ax.plot(X_smooth, np.sin(2*np.pi*X_smooth), 'k--', alpha=0.3, label='True')
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f"{titles[idx]}\nTrain MSE: {results[deg]['train_mse']:.3f}, Test MSE: {results[deg]['test_mse']:.3f}", 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_demo.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'overfitting_demo.png'")
plt.show()

print(f"\n{'='*50}")
print("KEY TAKEAWAY")
print(f"{'='*50}")
print("""
Overfitting occurs when:
- Model is too complex for the data
- LOW training error but HIGH test error
- Model has HIGH VARIANCE

Solution: Use regularization (L1, L2) or simpler model
""")
