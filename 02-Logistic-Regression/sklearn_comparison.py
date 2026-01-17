"""
Scratch vs Sklearn Comparison
==============================
Side-by-side comparison of our implementation vs sklearn's LogisticRegression.
"""

import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import StandardScaler

# ============================================
# Load and Prepare Data
# ============================================
print("=" * 60)
print("LOGISTIC REGRESSION: SCRATCH vs SKLEARN")
print("=" * 60)

# Load breast cancer dataset
data = load_breast_cancer()
X_full = data.data[:, [0, 1]]  # mean radius, mean texture
y = data.target

# Normalize (StandardScaler for fair sklearn comparison)
scaler = StandardScaler()
X = scaler.fit_transform(X_full)

# Train/test split
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nDataset: Breast Cancer ({len(y)} samples)")
print(f"Features: 'mean radius', 'mean texture'")
print(f"Train/Test split: {len(X_train)}/{len(X_test)}")

# ============================================
# OUR IMPLEMENTATION (From Scratch)
# ============================================
print(f"\n{'='*60}")
print("1. OUR IMPLEMENTATION (FROM SCRATCH)")
print(f"{'='*60}")

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Initialize
w_scratch = np.zeros(2)
b_scratch = 0.0
learning_rate = 0.1
iterations = 5000
m = len(X_train)

# Train
start_time = time.time()
for i in range(iterations):
    z = np.dot(X_train, w_scratch) + b_scratch
    y_pred = sigmoid(z)
    error = y_pred - y_train
    w_scratch -= learning_rate * (1/m) * np.dot(X_train.T, error)
    b_scratch -= learning_rate * (1/m) * np.sum(error)
scratch_time = time.time() - start_time

# Predictions
y_test_pred_scratch = (sigmoid(np.dot(X_test, w_scratch) + b_scratch) >= 0.5).astype(int)
accuracy_scratch = np.mean(y_test_pred_scratch == y_test) * 100

print(f"\nTraining time: {scratch_time*1000:.2f} ms")
print(f"Iterations: {iterations}")
print(f"Weights: [{w_scratch[0]:.4f}, {w_scratch[1]:.4f}]")
print(f"Bias: {b_scratch:.4f}")
print(f"Test Accuracy: {accuracy_scratch:.2f}%")

# ============================================
# SKLEARN IMPLEMENTATION
# ============================================
print(f"\n{'='*60}")
print("2. SKLEARN IMPLEMENTATION")
print(f"{'='*60}")

# Train sklearn model
start_time = time.time()
sklearn_model = SklearnLR(max_iter=5000, solver='lbfgs', C=1e10)  # C=1e10 ≈ no regularization
sklearn_model.fit(X_train, y_train)
sklearn_time = time.time() - start_time

# Get weights
w_sklearn = sklearn_model.coef_[0]
b_sklearn = sklearn_model.intercept_[0]

# Predictions
y_test_pred_sklearn = sklearn_model.predict(X_test)
accuracy_sklearn = np.mean(y_test_pred_sklearn == y_test) * 100

print(f"\nTraining time: {sklearn_time*1000:.2f} ms")
print(f"Weights: [{w_sklearn[0]:.4f}, {w_sklearn[1]:.4f}]")
print(f"Bias: {b_sklearn:.4f}")
print(f"Test Accuracy: {accuracy_sklearn:.2f}%")

# ============================================
# COMPARISON
# ============================================
print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")

print(f"\n{'Metric':<25} {'Scratch':<18} {'Sklearn':<18}")
print("-" * 60)
print(f"{'Weight 1:':<25} {w_scratch[0]:>12.4f}      {w_sklearn[0]:>12.4f}")
print(f"{'Weight 2:':<25} {w_scratch[1]:>12.4f}      {w_sklearn[1]:>12.4f}")
print(f"{'Bias:':<25} {b_scratch:>12.4f}      {b_sklearn:>12.4f}")
print(f"{'Test Accuracy:':<25} {accuracy_scratch:>11.2f}%      {accuracy_sklearn:>11.2f}%")
print(f"{'Training Time:':<25} {scratch_time*1000:>11.2f}ms      {sklearn_time*1000:>11.2f}ms")

# Prediction agreement
agreement = np.mean(y_test_pred_scratch == y_test_pred_sklearn) * 100
print(f"{'Prediction Agreement:':<25} {agreement:>41.1f}%")

# Weight difference
w_diff = np.abs(w_scratch - w_sklearn)
print(f"{'Weight Difference:':<25} [{w_diff[0]:.4f}, {w_diff[1]:.4f}]")

# ============================================
# Interpretation
# ============================================
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")

if agreement >= 95:
    print("\n✅ EXCELLENT! Our implementation closely matches sklearn.")
    print("   The math is correct!")
elif agreement >= 85:
    print("\n✓ GOOD! Results are similar with minor differences.")
    print("   Differences likely due to optimization methods.")
else:
    print("\n⚠ Results differ significantly.")
    print("   May need more iterations or learning rate tuning.")

print(f"""
Key Differences:
- Sklearn uses LBFGS optimizer (2nd order), we use vanilla gradient descent
- Sklearn includes regularization by default (C parameter)
- Sklearn may converge faster due to advanced optimization

Both implementations learn the same underlying pattern!
""")

print(f"{'='*60}")
print("COMPARISON COMPLETE!")
print(f"{'='*60}")
