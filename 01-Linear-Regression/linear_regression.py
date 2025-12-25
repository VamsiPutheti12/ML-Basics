"""
Linear Regression Implementation from Scratch
==============================================

This module implements Simple Linear Regression using gradient descent.
Perfect for learning the fundamentals of machine learning!

Author: ML-Basics Learning Repository
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class LinearRegression:
    """
    Simple Linear Regression using Gradient Descent
    
    Finds the best fit line: y = wx + b
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = None  # Slope (w)
        self.bias = None    # Intercept (b)
        self.cost_history = []  # Track cost over iterations
        
    def fit(self, X, y):
        """
        Train the model using gradient descent
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Training data (input features)
        y : array-like, shape (n_samples,)
            Target values
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get number of training examples
        m = len(X)
        
        # Initialize parameters to zero
        self.weight = 0.0
        self.bias = 0.0
        
        # Gradient Descent
        for iteration in range(self.n_iterations):
            # 1. Make predictions with current parameters
            y_predicted = self.weight * X + self.bias
            
            # 2. Calculate the error
            error = y_predicted - y
            
            # 3. Calculate gradients (partial derivatives)
            dw = (1/m) * np.sum(error * X)  # Gradient for weight
            db = (1/m) * np.sum(error)       # Gradient for bias
            
            # 4. Update parameters
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 5. Calculate and store cost (MSE)
            cost = (1/(2*m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.4f}, w = {self.weight:.4f}, b = {self.bias:.4f}")
        
        print(f"\nTraining Complete!")
        print(f"Final Parameters: w = {self.weight:.4f}, b = {self.bias:.4f}")
        print(f"Final Cost: {self.cost_history[-1]:.4f}")
        
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        predictions : array
            Predicted values
        """
        X = np.array(X)
        return self.weight * X + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            True values
            
        Returns:
        --------
        r2_score : float
            R² score (1.0 is perfect prediction)
        """
        y = np.array(y)
        y_pred = self.predict(X)
        
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Residual sum of squares
        ss_res = np.sum((y - y_pred) ** 2)
        
        # R² score
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def mean_squared_error(self, X, y):
        """
        Calculate Mean Squared Error
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            True values
            
        Returns:
        --------
        mse : float
            Mean squared error
        """
        y = np.array(y)
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse
    
    def root_mean_squared_error(self, X, y):
        """
        Calculate Root Mean Squared Error
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            True values
            
        Returns:
        --------
        rmse : float
            Root mean squared error
        """
        return np.sqrt(self.mean_squared_error(X, y))


def create_visualizations(model, X_train, y_train, X_test, y_test):
    """
    Create comprehensive visualizations for the linear regression model
    
    Parameters:
    -----------
    model : LinearRegression
        Trained model
    X_train, y_train : arrays
        Training data
    X_test, y_test : arrays
        Test data
    """
    # Create visualizations directory
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Best Fit Line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=50, label='Training Data')
    plt.scatter(X_test, y_test, color='green', alpha=0.6, s=50, label='Test Data')
    
    # Plot the regression line
    X_line = np.linspace(X_train.min(), X_train.max(), 100)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label=f'Best Fit Line: y = {model.weight:.2f}x + {model.bias:.2f}')
    
    plt.xlabel('X (Input Feature)', fontsize=12)
    plt.ylabel('y (Target)', fontsize=12)
    plt.title('Linear Regression: Best Fit Line', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / 'best_fit_line.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_dir / 'best_fit_line.png'}")
    plt.close()
    
    # 2. Cost Function History
    plt.figure(figsize=(10, 6))
    plt.plot(model.cost_history, color='purple', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (MSE)', fontsize=12)
    plt.title('Cost Function Over Training Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / 'cost_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_dir / 'cost_history.png'}")
    plt.close()
    
    # 3. Predictions vs Actual
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_pred_train, color='blue', alpha=0.6, s=50, label='Training Data')
    plt.scatter(y_test, y_pred_test, color='green', alpha=0.6, s=50, label='Test Data')
    
    # Perfect prediction line
    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / 'predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_dir / 'predictions_vs_actual.png'}")
    plt.close()
    
    # 4. Residuals Plot
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, residuals_train, color='blue', alpha=0.6, s=50, label='Training Residuals')
    plt.scatter(X_test, residuals_test, color='green', alpha=0.6, s=50, label='Test Residuals')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    plt.xlabel('X (Input Feature)', fontsize=12)
    plt.ylabel('Residuals (Error)', fontsize=12)
    plt.title('Residuals Plot', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / 'residuals.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {viz_dir / 'residuals.png'}")
    plt.close()
    
    print("\n✅ All visualizations created successfully!")


def main():
    """
    Main function to demonstrate Linear Regression
    """
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)
    print()
    
    # Generate sample data
    print("📊 Generating sample data...")
    np.random.seed(42)
    
    # Create a linear relationship with some noise
    X = np.linspace(0, 10, 100)
    y = 2.5 * X + 5 + np.random.randn(100) * 2  # y = 2.5x + 5 + noise
    
    # Split into train and test sets (80-20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Create and train the model
    print("🚀 Training Linear Regression model...")
    print("-" * 60)
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    print()
    
    # Evaluate the model
    print("📈 Model Evaluation:")
    print("-" * 60)
    
    # Training metrics
    train_r2 = model.score(X_train, y_train)
    train_mse = model.mean_squared_error(X_train, y_train)
    train_rmse = model.root_mean_squared_error(X_train, y_train)
    
    print(f"Training Set:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print()
    
    # Test metrics
    test_r2 = model.score(X_test, y_test)
    test_mse = model.mean_squared_error(X_test, y_test)
    test_rmse = model.root_mean_squared_error(X_test, y_test)
    
    print(f"Test Set:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print()
    
    # Make some predictions
    print("🔮 Making Predictions:")
    print("-" * 60)
    test_inputs = [2.0, 5.0, 8.0]
    for x_val in test_inputs:
        prediction = model.predict([x_val])[0]
        print(f"  Input: {x_val:.1f} → Prediction: {prediction:.2f}")
    print()
    
    # Create visualizations
    print("📊 Creating Visualizations:")
    print("-" * 60)
    create_visualizations(model, X_train, y_train, X_test, y_test)
    print()
    
    print("=" * 60)
    print("✅ LINEAR REGRESSION COMPLETE!")
    print("=" * 60)
    print("\nCheck the 'visualizations' folder for plots!")
    print("Read README.md for detailed theory and explanations.")


if __name__ == "__main__":
    main()
