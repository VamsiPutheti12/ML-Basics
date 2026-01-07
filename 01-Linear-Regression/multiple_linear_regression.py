"""
Multiple Linear Regression with California Housing Dataset
===========================================================
Real-world example: Predicting house prices using multiple features

Dataset: California Housing (from sklearn)
Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
Target: Median house value
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION - CALIFORNIA HOUSING DATASET")
print("=" * 70)

# Load the California Housing dataset
print("\n📊 Loading California Housing Dataset...")
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Create DataFrame for better visualization
feature_names = housing.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y

print(f"✓ Dataset loaded successfully!")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"  {i+1}. {name}: {housing.feature_names[i]}")

print(f"\nTarget: Median house value (in $100,000s)")
print(f"\nDataset Statistics:")
print(df.describe())

# Split the data
print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Feature Normalization (important for gradient descent!)
print("\n📏 Normalizing features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features normalized (mean=0, std=1)")

# Multiple Linear Regression Implementation
print("\n" + "=" * 70)
print("TRAINING MULTIPLE LINEAR REGRESSION")
print("=" * 70)

# Initialize parameters
n_features = X_train_scaled.shape[1]
weights = np.zeros(n_features)  # One weight per feature
bias = 0.0
learning_rate = 0.01
iterations = 1000
m = len(X_train_scaled)

# Store cost history
cost_history = []

print(f"\nHyperparameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Iterations: {iterations}")
print(f"  Features: {n_features}")
print(f"\nTraining...")

# Gradient Descent
for i in range(iterations):
    # 1. Predictions: y_pred = X @ weights + bias
    y_pred = np.dot(X_train_scaled, weights) + bias
    
    # 2. Calculate gradients
    dw = (1/m) * np.dot(X_train_scaled.T, (y_pred - y_train))
    db = (1/m) * np.sum(y_pred - y_train)
    
    # 3. Update parameters
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # 4. Calculate cost (MSE)
    cost = (1/(2*m)) * np.sum((y_pred - y_train)**2)
    cost_history.append(cost)
    
    # Print progress
    if i % 200 == 0:
        print(f"Iteration {i:4d}: Cost = {cost:.6f}")

print(f"\n✓ Training complete!")
print(f"Final cost: {cost_history[-1]:.6f}")

# Display learned weights
print(f"\n📊 Learned Parameters:")
print(f"Bias (intercept): {bias:.4f}")
print(f"\nWeights (coefficients):")
for i, (name, weight) in enumerate(zip(feature_names, weights)):
    print(f"  {name:12s}: {weight:8.4f}")

# Make predictions
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_train_pred = np.dot(X_train_scaled, weights) + bias
y_test_pred = np.dot(X_test_scaled, weights) + bias

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, rmse, r2

train_mse, train_rmse, train_r2 = calculate_metrics(y_train, y_train_pred)
test_mse, test_rmse, test_r2 = calculate_metrics(y_test, y_test_pred)

print(f"\nTraining Set:")
print(f"  MSE:  {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f} ($100k)")
print(f"  R²:   {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  MSE:  {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f} ($100k)")
print(f"  R²:   {test_r2:.4f}")

# Sample predictions
print(f"\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)
print(f"\nShowing first 5 test samples:")
print(f"{'Actual':>10s} {'Predicted':>10s} {'Error':>10s}")
print("-" * 35)
for i in range(5):
    actual = y_test[i]
    predicted = y_test_pred[i]
    error = actual - predicted
    print(f"${actual*100:8.1f}k  ${predicted*100:8.1f}k  ${error*100:8.1f}k")

# Visualizations
print(f"\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Cost History
ax1 = plt.subplot(2, 3, 1)
ax1.plot(cost_history, color='purple', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost (MSE)')
ax1.set_title('Cost Function Over Training', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Predictions vs Actual (Training)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_train, y_train_pred, alpha=0.3, s=10, color='blue')
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax2.set_xlabel('Actual Price ($100k)')
ax2.set_ylabel('Predicted Price ($100k)')
ax2.set_title(f'Training Set (R² = {train_r2:.3f})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Predictions vs Actual (Test)
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax3.set_xlabel('Actual Price ($100k)')
ax3.set_ylabel('Predicted Price ($100k)')
ax3.set_title(f'Test Set (R² = {test_r2:.3f})', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Residuals (Training)
ax4 = plt.subplot(2, 3, 4)
residuals_train = y_train - y_train_pred
ax4.scatter(y_train_pred, residuals_train, alpha=0.3, s=10, color='blue')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Price ($100k)')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals - Training Set', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Residuals (Test)
ax5 = plt.subplot(2, 3, 5)
residuals_test = y_test - y_test_pred
ax5.scatter(y_test_pred, residuals_test, alpha=0.5, s=20, color='green')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Price ($100k)')
ax5.set_ylabel('Residuals')
ax5.set_title('Residuals - Test Set', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Feature Importance (Weights)
ax6 = plt.subplot(2, 3, 6)
colors = ['green' if w > 0 else 'red' for w in weights]
bars = ax6.barh(feature_names, weights, color=colors, alpha=0.7)
ax6.set_xlabel('Weight (Coefficient)')
ax6.set_title('Feature Importance', fontweight='bold')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('multiple_linear_regression_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiple_linear_regression_results.png")
plt.show()

# Additional visualization: Feature correlation with price
fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    axes[i].scatter(df[feature], df['Price'], alpha=0.3, s=5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Price ($100k)')
    axes[i].set_title(f'{feature} vs Price')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_correlations.png")
plt.show()

print("\n" + "=" * 70)
print("✅ MULTIPLE LINEAR REGRESSION COMPLETE!")
print("=" * 70)
print("\nKey Takeaways:")
print(f"  • Used {n_features} features to predict house prices")
print(f"  • Achieved R² score of {test_r2:.3f} on test set")
print(f"  • RMSE: ${test_rmse*100:.1f}k (average prediction error)")
print(f"  • Feature normalization was crucial for convergence")
print(f"  • Most important features: {feature_names[np.argmax(np.abs(weights))]}")
print("\nVisualization files saved!")
