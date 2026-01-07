"""
Multiple Linear Regression with Recent USA Housing Dataset
===========================================================
Real-world example: Predicting house prices using recent data (2015-2024)

This version uses a synthetic but realistic USA housing dataset
that represents modern housing market conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("MULTIPLE LINEAR REGRESSION - RECENT USA HOUSING DATA (2015-2024)")
print("=" * 70)

# Create a realistic synthetic dataset based on recent housing market trends
# This simulates real data from 2015-2024 with modern price ranges
print("\n📊 Generating Recent Housing Dataset (2015-2024)...")
np.random.seed(42)

n_samples = 5000

# Generate realistic features based on 2015-2024 market
data = {
    'SquareFeet': np.random.normal(2000, 600, n_samples).clip(800, 5000),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Bathrooms': np.random.uniform(1, 4, n_samples),
    'YearBuilt': np.random.randint(1960, 2024, n_samples),
    'LotSize': np.random.normal(8000, 3000, n_samples).clip(2000, 30000),
    'GarageSpaces': np.random.randint(0, 4, n_samples),
    'SchoolRating': np.random.uniform(3, 10, n_samples),
    'CrimeRate': np.random.uniform(0.5, 8, n_samples),
}

df = pd.DataFrame(data)

# Calculate realistic house prices (2015-2024 range: $150k - $1.5M)
# Price formula based on realistic market factors
base_price = 100000
price = (
    base_price +
    df['SquareFeet'] * 150 +  # $150 per sq ft
    df['Bedrooms'] * 25000 +   # $25k per bedroom
    df['Bathrooms'] * 15000 +  # $15k per bathroom
    (2024 - df['YearBuilt']) * -800 +  # Newer = more expensive
    df['LotSize'] * 8 +  # $8 per sq ft of lot
    df['GarageSpaces'] * 20000 +  # $20k per garage space
    df['SchoolRating'] * 15000 +  # $15k per rating point
    df['CrimeRate'] * -10000 +  # Lower crime = higher price
    np.random.normal(0, 50000, n_samples)  # Random variation
)

df['Price'] = price.clip(150000, 1500000)  # Realistic price range

print(f"✓ Dataset created successfully!")
print(f"  Samples: {len(df)}")
print(f"  Features: {len(df.columns) - 1}")
print(f"  Time Period: 2015-2024 (modern housing market)")
print(f"\nFeatures:")
for i, col in enumerate(df.columns[:-1], 1):
    print(f"  {i}. {col}")

print(f"\nTarget: House Price (USD)")
print(f"\nDataset Statistics:")
print(df.describe())

print(f"\nPrice Range:")
print(f"  Minimum: ${df['Price'].min():,.0f}")
print(f"  Maximum: ${df['Price'].max():,.0f}")
print(f"  Median:  ${df['Price'].median():,.0f}")
print(f"  Mean:    ${df['Price'].mean():,.0f}")

# Prepare features and target
X = df.drop('Price', axis=1).values
y = df['Price'].values
feature_names = df.columns[:-1].tolist()

# Split the data
print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Feature Normalization
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
weights = np.zeros(n_features)
bias = 0.0
learning_rate = 0.01
iterations = 1000
m = len(X_train_scaled)

cost_history = []

print(f"\nHyperparameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Iterations: {iterations}")
print(f"  Features: {n_features}")
print(f"\nTraining...")

# Gradient Descent
for i in range(iterations):
    # Predictions
    y_pred = np.dot(X_train_scaled, weights) + bias
    
    # Calculate gradients
    dw = (1/m) * np.dot(X_train_scaled.T, (y_pred - y_train))
    db = (1/m) * np.sum(y_pred - y_train)
    
    # Update parameters
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # Calculate cost
    cost = (1/(2*m)) * np.sum((y_pred - y_train)**2)
    cost_history.append(cost)
    
    if i % 200 == 0:
        print(f"Iteration {i:4d}: Cost = {cost:.2e}")

print(f"\n✓ Training complete!")
print(f"Final cost: {cost_history[-1]:.2e}")

# Display learned weights
print(f"\n📊 Learned Parameters:")
print(f"Bias (intercept): ${bias:,.0f}")
print(f"\nWeights (coefficients):")
for name, weight in zip(feature_names, weights):
    print(f"  {name:15s}: {weight:12,.2f}")

# Evaluation
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_train_pred = np.dot(X_train_scaled, weights) + bias
y_test_pred = np.dot(X_test_scaled, weights) + bias

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, rmse, r2, mae

train_mse, train_rmse, train_r2, train_mae = calculate_metrics(y_train, y_train_pred)
test_mse, test_rmse, test_r2, test_mae = calculate_metrics(y_test, y_test_pred)

print(f"\nTraining Set:")
print(f"  R²:   {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:,.0f}")
print(f"  MAE:  ${train_mae:,.0f}")

print(f"\nTest Set:")
print(f"  R²:   {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:,.0f}")
print(f"  MAE:  ${test_mae:,.0f}")

# Sample predictions
print(f"\n" + "=" * 70)
print("SAMPLE PREDICTIONS (2024 Market Prices)")
print("=" * 70)
print(f"\nShowing first 5 test samples:")
print(f"{'Actual':>15s} {'Predicted':>15s} {'Error':>15s}")
print("-" * 50)
for i in range(5):
    actual = y_test[i]
    predicted = y_test_pred[i]
    error = actual - predicted
    print(f"${actual:13,.0f}  ${predicted:13,.0f}  ${error:13,.0f}")

# Visualizations
print(f"\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

# 1. Cost History
ax1 = plt.subplot(2, 3, 1)
ax1.plot(cost_history, color='purple', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')
ax1.set_title('Cost Function Over Training', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Predictions vs Actual (Training)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_train/1000, y_train_pred/1000, alpha=0.3, s=10, color='blue')
min_val = min(y_train.min(), y_train_pred.min()) / 1000
max_val = max(y_train.max(), y_train_pred.max()) / 1000
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax2.set_xlabel('Actual Price ($1000s)')
ax2.set_ylabel('Predicted Price ($1000s)')
ax2.set_title(f'Training Set (R² = {train_r2:.3f})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Predictions vs Actual (Test)
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test/1000, y_test_pred/1000, alpha=0.5, s=20, color='green')
min_val = min(y_test.min(), y_test_pred.min()) / 1000
max_val = max(y_test.max(), y_test_pred.max()) / 1000
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax3.set_xlabel('Actual Price ($1000s)')
ax3.set_ylabel('Predicted Price ($1000s)')
ax3.set_title(f'Test Set (R² = {test_r2:.3f})', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Residuals (Training)
ax4 = plt.subplot(2, 3, 4)
residuals_train = (y_train - y_train_pred) / 1000
ax4.scatter(y_train_pred/1000, residuals_train, alpha=0.3, s=10, color='blue')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Price ($1000s)')
ax4.set_ylabel('Residuals ($1000s)')
ax4.set_title('Residuals - Training Set', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Residuals (Test)
ax5 = plt.subplot(2, 3, 5)
residuals_test = (y_test - y_test_pred) / 1000
ax5.scatter(y_test_pred/1000, residuals_test, alpha=0.5, s=20, color='green')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Price ($1000s)')
ax5.set_ylabel('Residuals ($1000s)')
ax5.set_title('Residuals - Test Set', fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Feature Importance
ax6 = plt.subplot(2, 3, 6)
colors = ['green' if w > 0 else 'red' for w in weights]
bars = ax6.barh(feature_names, weights, color=colors, alpha=0.7)
ax6.set_xlabel('Weight (Coefficient)')
ax6.set_title('Feature Importance', fontweight='bold')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('recent_housing_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: recent_housing_results.png")
plt.show()

print("\n" + "=" * 70)
print("✅ MULTIPLE LINEAR REGRESSION COMPLETE!")
print("=" * 70)
print("\nKey Takeaways:")
print(f"  • Used {n_features} features to predict modern house prices")
print(f"  • Achieved R² score of {test_r2:.3f} on test set")
print(f"  • RMSE: ${test_rmse:,.0f} (average prediction error)")
print(f"  • Data represents 2015-2024 housing market")
print(f"  • Price range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
print(f"  • Most important feature: {feature_names[np.argmax(np.abs(weights))]}")
print("\nVisualization file saved!")
