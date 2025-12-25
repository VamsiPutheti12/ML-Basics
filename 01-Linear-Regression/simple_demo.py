"""
Simple Linear Regression Demo
==============================
A minimal example showing gradient descent and best fit line plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

# Sample data: Hours studied vs Exam score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 4, 5, 4, 5, 7, 8, 9])

# Initialize parameters
w = 0.0  # weight (slope)
b = 0.0  # bias (intercept)
learning_rate = 0.01
iterations = 1000

# Number of data points
m = len(X)

print("=" * 50)
print("SIMPLE LINEAR REGRESSION DEMO")
print("=" * 50)
print(f"\nTraining data: {m} samples")
print(f"Learning rate: {learning_rate}")
print(f"Iterations: {iterations}\n")

# Gradient Descent
print("Training...")
for i in range(iterations):
    # 1. Predictions
    y_pred = w * X + b
    
    # 2. Calculate gradients
    dw = (1/m) * np.sum((y_pred - y) * X)
    db = (1/m) * np.sum(y_pred - y)
    
    # 3. Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Print progress
    if i % 200 == 0:
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        print(f"Iteration {i}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

# Final results
print(f"\n{'='*50}")
print("TRAINING COMPLETE!")
print(f"{'='*50}")
print(f"Final equation: y = {w:.2f}x + {b:.2f}")
print(f"Final weight (slope): {w:.4f}")
print(f"Final bias (intercept): {b:.4f}")

# Calculate R² score
y_pred_final = w * X + b
ss_tot = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_pred_final)**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R² Score: {r2:.4f}")

# Plot the best fit line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, alpha=0.6, label='Actual Data')
plt.plot(X, y_pred_final, color='red', linewidth=2, label=f'Best Fit Line: y = {w:.2f}x + {b:.2f}')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.title('Linear Regression: Hours Studied vs Exam Score', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('simple_demo.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'simple_demo.png'")
plt.show()

print(f"\n{'='*50}")
print("DEMO COMPLETE!")
print(f"{'='*50}")
