"""
Simple Logistic Regression Demo
================================
A minimal example showing binary classification with gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Generate synthetic binary classification data
# ============================================
np.random.seed(42)

# Class 0: Study hours 1-4 (fail)
# Class 1: Study hours 6-10 (pass)
X_class0 = np.random.uniform(1, 4, 8)   # 8 samples for class 0
X_class1 = np.random.uniform(6, 10, 10)  # 10 samples for class 1

X = np.concatenate([X_class0, X_class1])
y = np.concatenate([np.zeros(8), np.ones(10)])

# Shuffle the data
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Number of samples
m = len(X)

print("=" * 50)
print("SIMPLE LOGISTIC REGRESSION DEMO")
print("=" * 50)
print(f"\nTraining data: {m} samples")
print(f"Class 0 (Fail): {int(np.sum(y == 0))} samples")
print(f"Class 1 (Pass): {int(np.sum(y == 1))} samples\n")

# ============================================
# Sigmoid function
# ============================================
def sigmoid(z):
    """Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))

# ============================================
# Initialize parameters
# ============================================
w = 0.0  # weight
b = 0.0  # bias
learning_rate = 0.5
iterations = 1000

print(f"Learning rate: {learning_rate}")
print(f"Iterations: {iterations}\n")

# ============================================
# Gradient Descent Training
# ============================================
print("Training...")
for i in range(iterations):
    # 1. Forward pass: compute probabilities
    z = w * X + b
    y_pred = sigmoid(z)
    
    # 2. Compute gradients
    dw = (1/m) * np.sum((y_pred - y) * X)
    db = (1/m) * np.sum(y_pred - y)
    
    # 3. Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # 4. Compute and print loss (binary cross-entropy)
    if i % 200 == 0:
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        print(f"Iteration {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# ============================================
# Final Results
# ============================================
print(f"\n{'='*50}")
print("TRAINING COMPLETE!")
print(f"{'='*50}")
print(f"Final weight: {w:.4f}")
print(f"Final bias: {b:.4f}")

# Decision boundary: where sigmoid(wx + b) = 0.5 → wx + b = 0 → x = -b/w
decision_boundary = -b / w
print(f"Decision boundary: x = {decision_boundary:.2f}")

# Make predictions (probability > 0.5 → class 1)
y_pred_final = sigmoid(w * X + b)
y_pred_class = (y_pred_final >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(y_pred_class == y) * 100
print(f"Accuracy: {accuracy:.1f}%")

# ============================================
# Visualization
# ============================================
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X[y == 0], y[y == 0], color='red', s=100, label='Class 0 (Fail)', alpha=0.7)
plt.scatter(X[y == 1], y[y == 1], color='green', s=100, label='Class 1 (Pass)', alpha=0.7)

# Plot sigmoid curve
X_line = np.linspace(0, 11, 100)
y_line = sigmoid(w * X_line + b)
plt.plot(X_line, y_line, color='blue', linewidth=2, label='Sigmoid Curve')

# Plot decision boundary
plt.axvline(x=decision_boundary, color='orange', linestyle='--', linewidth=2, 
            label=f'Decision Boundary (x={decision_boundary:.2f})')
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Probability of Passing', fontsize=12)
plt.title('Logistic Regression: Pass/Fail Classification', fontsize=14, fontweight='bold')
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 11)
plt.ylim(-0.1, 1.1)

plt.savefig('simple_demo_result.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'simple_demo_result.png'")
plt.show()

print(f"\n{'='*50}")
print("DEMO COMPLETE!")
print(f"{'='*50}")
