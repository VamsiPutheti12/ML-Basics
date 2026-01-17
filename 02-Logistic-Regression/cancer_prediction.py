"""
Cancer Prediction using Logistic Regression
=============================================
Binary classification: Malignant (0) vs Benign (1)
Uses sklearn's breast cancer dataset with 2 features for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# ============================================
# Load and Prepare Data
# ============================================
print("=" * 50)
print("BREAST CANCER PREDICTION")
print("=" * 50)

# Load the breast cancer dataset
data = load_breast_cancer()
feature_names = data.feature_names

# Use only 2 features for easy visualization
# Feature 0: mean radius, Feature 1: mean texture
X_full = data.data[:, [0, 1]]  # Shape: (569, 2)
y = data.target  # 0 = malignant, 1 = benign

print(f"\nDataset: {data.DESCR.split(chr(10))[0]}")
print(f"Total samples: {len(y)}")
print(f"Features used: '{feature_names[0]}', '{feature_names[1]}'")
print(f"Malignant (0): {np.sum(y == 0)} samples")
print(f"Benign (1): {np.sum(y == 1)} samples")

# ============================================
# Normalize features (Min-Max Scaling)
# ============================================
X_min = X_full.min(axis=0)
X_max = X_full.max(axis=0)
X = (X_full - X_min) / (X_max - X_min)

print(f"\nFeatures normalized to [0, 1] range")

# ============================================
# Train/Test Split (80/20)
# ============================================
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================
# Sigmoid Function
# ============================================
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# ============================================
# Initialize Parameters
# ============================================
w = np.zeros(2)  # weights for 2 features
b = 0.0          # bias
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
    z = np.dot(X_train, w) + b
    y_pred = sigmoid(z)
    
    # Compute gradients
    error = y_pred - y_train
    dw = (1/m) * np.dot(X_train.T, error)
    db = (1/m) * np.sum(error)
    
    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print progress
    if i % 400 == 0:
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_train * np.log(y_pred_clipped) + (1 - y_train) * np.log(1 - y_pred_clipped))
        print(f"Iteration {i}: Loss = {loss:.4f}")

# ============================================
# Evaluation
# ============================================
print(f"\n{'='*50}")
print("TRAINING COMPLETE!")
print(f"{'='*50}")
print(f"Final weights: w1 = {w[0]:.4f}, w2 = {w[1]:.4f}")
print(f"Final bias: b = {b:.4f}")

# Predictions on test set
y_test_prob = sigmoid(np.dot(X_test, w) + b)
y_test_pred = (y_test_prob >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(y_test_pred == y_test) * 100
print(f"\nTest Accuracy: {accuracy:.1f}%")

# Confusion Matrix
TP = np.sum((y_test == 1) & (y_test_pred == 1))
TN = np.sum((y_test == 0) & (y_test_pred == 0))
FP = np.sum((y_test == 0) & (y_test_pred == 1))
FN = np.sum((y_test == 1) & (y_test_pred == 0))

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"              Malignant  Benign")
print(f"Actual Malignant   {TN:3d}      {FP:3d}")
print(f"       Benign      {FN:3d}      {TP:3d}")

# ============================================
# Visualization
# ============================================
plt.figure(figsize=(10, 8))

# Plot training data
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
            c='red', marker='o', s=50, alpha=0.6, label='Malignant (Train)')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
            c='green', marker='o', s=50, alpha=0.6, label='Benign (Train)')

# Plot test data
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], 
            c='red', marker='x', s=80, alpha=0.9, label='Malignant (Test)')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
            c='green', marker='x', s=80, alpha=0.9, label='Benign (Test)')

# Plot decision boundary: w1*x1 + w2*x2 + b = 0  →  x2 = -(w1*x1 + b) / w2
x1_range = np.linspace(0, 1, 100)
x2_boundary = -(w[0] * x1_range + b) / w[1]
plt.plot(x1_range, x2_boundary, 'b-', linewidth=2, label='Decision Boundary')

plt.xlabel(f'{feature_names[0]} (normalized)', fontsize=12)
plt.ylabel(f'{feature_names[1]} (normalized)', fontsize=12)
plt.title(f'Breast Cancer Classification (Accuracy: {accuracy:.1f}%)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.savefig('cancer_prediction_result.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'cancer_prediction_result.png'")
plt.show()

print(f"\n{'='*50}")
print("PREDICTION COMPLETE!")
print(f"{'='*50}")
