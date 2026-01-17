"""
Multiclass Classification with Softmax Regression
===================================================
Classify Iris flowers into 3 species using One-vs-All approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ============================================
# Load and Prepare Data
# ============================================
print("=" * 50)
print("MULTICLASS CLASSIFICATION (IRIS DATASET)")
print("=" * 50)

# Load Iris dataset
iris = load_iris()
X_full = iris.data[:, [0, 2]]  # Sepal length & Petal length (for visualization)
y = iris.target  # 0=Setosa, 1=Versicolor, 2=Virginica
class_names = iris.target_names

print(f"\nDataset: Iris (3 classes)")
print(f"Features: '{iris.feature_names[0]}', '{iris.feature_names[2]}'")
print(f"Total samples: {len(y)}")
for i, name in enumerate(class_names):
    print(f"  Class {i} ({name}): {np.sum(y == i)} samples")

# ============================================
# Normalize Features
# ============================================
X_min = X_full.min(axis=0)
X_max = X_full.max(axis=0)
X = (X_full - X_min) / (X_max - X_min)

# ============================================
# Train/Test Split
# ============================================
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# One-hot encode labels
n_classes = 3
y_train_onehot = np.eye(n_classes)[y_train]
y_test_onehot = np.eye(n_classes)[y_test]

# ============================================
# Softmax Function
# ============================================
def softmax(z):
    """Softmax: converts logits to probabilities for multiclass"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ============================================
# Initialize Parameters
# ============================================
n_features = X_train.shape[1]
W = np.zeros((n_features, n_classes))  # Shape: (2, 3)
b = np.zeros(n_classes)                 # Shape: (3,)

learning_rate = 1.0
iterations = 2000
m = len(X_train)

print(f"\nLearning rate: {learning_rate}")
print(f"Iterations: {iterations}\n")

# ============================================
# Training with Gradient Descent
# ============================================
print("Training...")
for i in range(iterations):
    # Forward pass
    z = np.dot(X_train, W) + b  # Shape: (m, 3)
    y_pred = softmax(z)
    
    # Compute gradients
    error = y_pred - y_train_onehot  # Shape: (m, 3)
    dW = (1/m) * np.dot(X_train.T, error)
    db = (1/m) * np.sum(error, axis=0)
    
    # Update parameters
    W -= learning_rate * dW
    b -= learning_rate * db
    
    # Print progress
    if i % 400 == 0:
        # Cross-entropy loss
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(np.sum(y_train_onehot * np.log(y_pred_clipped), axis=1))
        print(f"Iteration {i}: Loss = {loss:.4f}")

# ============================================
# Evaluation
# ============================================
print(f"\n{'='*50}")
print("TRAINING COMPLETE!")
print(f"{'='*50}")

# Test predictions
z_test = np.dot(X_test, W) + b
y_test_prob = softmax(z_test)
y_test_pred = np.argmax(y_test_prob, axis=1)

# Accuracy
accuracy = np.mean(y_test_pred == y_test) * 100
print(f"\nTest Accuracy: {accuracy:.1f}%")

# Per-class accuracy
print(f"\nPer-class Results:")
for i, name in enumerate(class_names):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = np.mean(y_test_pred[mask] == i) * 100
        print(f"  {name}: {class_acc:.1f}% ({np.sum(mask)} samples)")

# ============================================
# Visualization
# ============================================
plt.figure(figsize=(12, 5))

# Plot 1: Data points
plt.subplot(1, 2, 1)
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']
for i, name in enumerate(class_names):
    mask = y_test == i
    plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                c=colors[i], marker=markers[i], s=80, 
                alpha=0.7, label=name, edgecolors='black')

plt.xlabel(f'{iris.feature_names[0]} (normalized)', fontsize=11)
plt.ylabel(f'{iris.feature_names[2]} (normalized)', fontsize=11)
plt.title('Iris Classification - Test Data', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Decision regions
plt.subplot(1, 2, 2)
h = 0.01
x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

z_grid = np.dot(grid_points, W) + b
probs_grid = softmax(z_grid)
predictions_grid = np.argmax(probs_grid, axis=1).reshape(xx.shape)

plt.contourf(xx, yy, predictions_grid, alpha=0.3, 
             levels=[-0.5, 0.5, 1.5, 2.5], colors=['red', 'green', 'blue'])
plt.contour(xx, yy, predictions_grid, levels=[0.5, 1.5], colors='black', linewidths=2)

for i, name in enumerate(class_names):
    mask = y_test == i
    plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                c=colors[i], marker=markers[i], s=80, 
                alpha=0.8, label=name, edgecolors='black')

plt.xlabel(f'{iris.feature_names[0]} (normalized)', fontsize=11)
plt.ylabel(f'{iris.feature_names[2]} (normalized)', fontsize=11)
plt.title('Decision Regions', fontsize=13, fontweight='bold')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('multiclass_iris_result.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'multiclass_iris_result.png'")
plt.show()

print(f"\n{'='*50}")
print("MULTICLASS CLASSIFICATION COMPLETE!")
print(f"{'='*50}")
