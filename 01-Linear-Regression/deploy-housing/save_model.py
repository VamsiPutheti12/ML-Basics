"""
Save Trained Model Parameters
=============================
This script trains the model and saves the learned parameters
(weights, bias, scaler) for deployment.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("TRAINING AND SAVING MODEL FOR DEPLOYMENT")
print("=" * 60)

# Generate the same dataset (using same seed for reproducibility)
np.random.seed(42)
n_samples = 5000

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

# Calculate prices
base_price = 100000
price = (
    base_price +
    df['SquareFeet'] * 150 +
    df['Bedrooms'] * 25000 +
    df['Bathrooms'] * 15000 +
    (2024 - df['YearBuilt']) * -800 +
    df['LotSize'] * 8 +
    df['GarageSpaces'] * 20000 +
    df['SchoolRating'] * 15000 +
    df['CrimeRate'] * -10000 +
    np.random.normal(0, 50000, n_samples)
)
df['Price'] = price.clip(150000, 1500000)

# Prepare data
X = df.drop('Price', axis=1).values
y = df['Price'].values
feature_names = df.columns[:-1].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
n_features = X_train_scaled.shape[1]
weights = np.zeros(n_features)
bias = 0.0
learning_rate = 0.01
iterations = 1000
m = len(X_train_scaled)

print("\nTraining model...")
for i in range(iterations):
    y_pred = np.dot(X_train_scaled, weights) + bias
    dw = (1/m) * np.dot(X_train_scaled.T, (y_pred - y_train))
    db = (1/m) * np.sum(y_pred - y_train)
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Calculate R² on test set
y_test_pred = np.dot(X_test_scaled, weights) + bias
ss_tot = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - y_test_pred)**2)
r2 = 1 - (ss_res / ss_tot)

print(f"✓ Training complete! Test R²: {r2:.4f}")

# Save model parameters
model_data = {
    'weights': weights,
    'bias': bias,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'feature_names': feature_names,
    'r2_score': r2
}

with open('housing_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to 'housing_model.pkl'")
print(f"\nSaved parameters:")
print(f"  - Weights: {len(weights)} values")
print(f"  - Bias: ${bias:,.0f}")
print(f"  - Scaler mean: {len(scaler.mean_)} values")
print(f"  - Scaler scale: {len(scaler.scale_)} values")
print(f"  - Feature names: {feature_names}")
print(f"\n✓ Ready for Streamlit deployment!")
