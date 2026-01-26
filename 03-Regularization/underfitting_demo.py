"""
Underfitting Demo
==================
When a model is too simple to capture the underlying pattern.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("UNDERFITTING DEMO")
print("=" * 50)

# ============================================
# Generate Non-Linear Data (Quadratic)
# ============================================
np.random.seed(42)
X = np.linspace(-3, 3, 30)
y_true = X**2  # True pattern is a parabola
y = y_true + np.random.randn(30) * 1.5  # Add noise

print(f"\nData: y = x² + noise")
print(f"Samples: {len(X)}")

# ============================================
# Fit a Linear Model (Too Simple!)
# ============================================
# Using normal equation for linear fit: y = wx + b
X_design = np.column_stack([np.ones(len(X)), X])
weights = np.linalg.lstsq(X_design, y, rcond=None)[0]
b, w = weights[0], weights[1]

y_pred_linear = w * X + b

# ============================================
# Fit a Quadratic Model (Just Right)
# ============================================
X_quad = np.column_stack([np.ones(len(X)), X, X**2])
weights_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
y_pred_quad = weights_quad[0] + weights_quad[1]*X + weights_quad[2]*X**2

# ============================================
# Calculate Errors
# ============================================
mse_linear = np.mean((y - y_pred_linear)**2)
mse_quad = np.mean((y - y_pred_quad)**2)

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"\nLinear Model (Underfit):")
print(f"  Equation: y = {w:.2f}x + {b:.2f}")
print(f"  MSE: {mse_linear:.2f}")

print(f"\nQuadratic Model (Good Fit):")
print(f"  Equation: y = {weights_quad[2]:.2f}x² + {weights_quad[1]:.2f}x + {weights_quad[0]:.2f}")
print(f"  MSE: {mse_quad:.2f}")

print(f"\n⚠️  The linear model has {mse_linear/mse_quad:.1f}x higher error!")
print("   This is UNDERFITTING - model too simple for the data.")

# ============================================
# Visualization
# ============================================
plt.figure(figsize=(12, 5))

# Plot 1: Underfit
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', s=60, alpha=0.7, label='Data')
plt.plot(X, y_pred_linear, 'r-', linewidth=2, label=f'Linear (MSE={mse_linear:.2f})')
plt.plot(X, y_true, 'g--', linewidth=2, alpha=0.5, label='True Pattern (y=x²)')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('❌ UNDERFITTING\n(Model too simple)', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Good Fit
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', s=60, alpha=0.7, label='Data')
plt.plot(X, y_pred_quad, 'g-', linewidth=2, label=f'Quadratic (MSE={mse_quad:.2f})')
plt.plot(X, y_true, 'g--', linewidth=2, alpha=0.5, label='True Pattern')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('✓ GOOD FIT\n(Model matches complexity)', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('underfitting_demo.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'underfitting_demo.png'")
plt.show()

print(f"\n{'='*50}")
print("KEY TAKEAWAY")
print(f"{'='*50}")
print("""
Underfitting occurs when:
- Model is too simple for the data
- High training error AND high test error
- Model has HIGH BIAS

Solution: Use a more complex model
""")
