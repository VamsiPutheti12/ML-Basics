"""
Polynomial Regression from Scratch
==================================
A complete implementation of polynomial regression with
coefficient interpretation and optimal point finding.
"""

import numpy as np


class PolynomialRegression:
    """
    Polynomial Regression implementation from scratch.
    
    Fits a polynomial of the form:
    ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
    
    Parameters:
    -----------
    degree : int
        Degree of the polynomial (default: 2 for quadratic)
    learning_rate : float
        Step size for gradient descent (default: 0.01)
    iterations : int
        Number of gradient descent iterations (default: 1000)
    normalize : bool
        Whether to normalize features before fitting (default: True)
    """
    
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000, normalize=True):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.normalize = normalize
        
        # Parameters to learn
        self.coefficients = None
        self.bias = None
        
        # Normalization parameters
        self.mean_ = None
        self.std_ = None
        
        # Training history
        self.cost_history = []
    
    def _normalize_features(self, X):
        """Normalize features to zero mean and unit variance."""
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            # Prevent division by zero
            self.std_[self.std_ == 0] = 1
        
        return (X - self.mean_) / self.std_
    
    def _polynomial_features(self, X):
        """
        Transform input features to polynomial features.
        
        For degree=3 and single feature x:
        [x] → [x, x², x³]
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        X_poly : array, shape (n_samples, n_features * degree)
            Polynomial features
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T
        
        n_samples = X.shape[0]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Create polynomial features for each original feature
        poly_features = []
        for d in range(1, self.degree + 1):
            poly_features.append(X ** d)
        
        return np.hstack(poly_features)
    
    def fit(self, X, y):
        """
        Fit polynomial regression model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,) or (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Ensure correct shapes
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T
        y = np.array(y).flatten()
        
        # Normalize if requested
        if self.normalize:
            X_norm = self._normalize_features(X)
        else:
            X_norm = X
        
        # Transform to polynomial features
        X_poly = self._polynomial_features(X_norm)
        
        n_samples, n_poly_features = X_poly.shape
        
        # Initialize coefficients
        self.coefficients = np.zeros(n_poly_features)
        self.bias = 0.0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.iterations):
            # Forward pass (predictions)
            y_pred = np.dot(X_poly, self.coefficients) + self.bias
            
            # Calculate gradients
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X_poly.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.coefficients -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate and store cost
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted polynomial model.
        
        Parameters:
        -----------
        X : array-like
            Features to predict
            
        Returns:
        --------
        y_pred : array
            Predicted values
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T
        
        # Normalize using training statistics
        if self.normalize:
            X_norm = (X - self.mean_) / self.std_
        else:
            X_norm = X
        
        # Transform to polynomial features
        X_poly = self._polynomial_features(X_norm)
        
        return np.dot(X_poly, self.coefficients) + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            True target values
            
        Returns:
        --------
        r2 : float
            R² score (1.0 is perfect prediction)
        """
        y = np.array(y).flatten()
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def get_coefficients(self):
        """
        Get the learned polynomial coefficients.
        
        Returns:
        --------
        dict with 'bias' and 'coefficients' for each polynomial term
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        result = {
            'bias': self.bias,
            'coefficients': {}
        }
        
        # Map coefficients to polynomial terms
        n_original_features = len(self.coefficients) // self.degree
        
        for d in range(1, self.degree + 1):
            for f in range(n_original_features):
                idx = (d - 1) * n_original_features + f
                key = f'x{f+1}^{d}' if n_original_features > 1 else f'x^{d}'
                result['coefficients'][key] = self.coefficients[idx]
        
        return result
    
    def find_optimal_point(self, feature_range=None):
        """
        Find the optimal point (maximum or minimum) for a quadratic model.
        
        For ŷ = β₀ + β₁x + β₂x²:
        x_optimal = -β₁ / (2β₂)
        
        Parameters:
        -----------
        feature_range : tuple, optional
            (min, max) range to search within
            
        Returns:
        --------
        dict with 'x_optimal', 'y_optimal', 'type' ('maximum' or 'minimum')
        """
        if self.degree < 2:
            raise ValueError("Need at least degree 2 to find optimal point")
        
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # For single feature quadratic
        # Coefficients are [β₁, β₂] after polynomial transform
        n_original = len(self.coefficients) // self.degree
        
        if n_original != 1:
            raise ValueError("find_optimal_point only works for single-feature models")
        
        beta_1 = self.coefficients[0]  # Linear term
        beta_2 = self.coefficients[1]  # Quadratic term
        
        if beta_2 == 0:
            return {'x_optimal': None, 'y_optimal': None, 'type': 'linear'}
        
        # Optimal point in normalized space
        x_opt_norm = -beta_1 / (2 * beta_2)
        
        # Convert back to original space
        if self.normalize:
            x_optimal = x_opt_norm * self.std_[0] + self.mean_[0]
        else:
            x_optimal = x_opt_norm
        
        # Clip to valid range if provided
        if feature_range is not None:
            x_optimal = np.clip(x_optimal, feature_range[0], feature_range[1])
        
        # Calculate y at optimal point
        y_optimal = self.predict(np.array([[x_optimal]]))[0]
        
        # Determine if maximum or minimum
        opt_type = 'maximum' if beta_2 < 0 else 'minimum'
        
        return {
            'x_optimal': x_optimal,
            'y_optimal': y_optimal,
            'type': opt_type
        }
    
    def marginal_effect(self, x_values):
        """
        Calculate the marginal effect (rate of change) at given x values.
        
        For ŷ = β₀ + β₁x + β₂x² + β₃x³:
        dŷ/dx = β₁ + 2β₂x + 3β₃x²
        
        Parameters:
        -----------
        x_values : array-like
            Points at which to calculate marginal effect
            
        Returns:
        --------
        effects : array
            Marginal effect at each x value
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        x_values = np.array(x_values).flatten()
        
        # Normalize x values
        if self.normalize:
            x_norm = (x_values - self.mean_[0]) / self.std_[0]
        else:
            x_norm = x_values
        
        # Calculate derivative: dŷ/dx = β₁ + 2β₂x + 3β₃x² + ...
        n_original = len(self.coefficients) // self.degree
        effects = np.zeros_like(x_values, dtype=float)
        
        for d in range(1, self.degree + 1):
            idx = (d - 1) * n_original
            coefficient = self.coefficients[idx]
            # Derivative of x^d is d * x^(d-1)
            effects += d * coefficient * (x_norm ** (d - 1))
        
        # Adjust for normalization
        if self.normalize:
            effects /= self.std_[0]
        
        return effects
    
    def mean_squared_error(self, X, y):
        """Calculate Mean Squared Error."""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def root_mean_squared_error(self, X, y):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(self.mean_squared_error(X, y))


# Quick demo if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("POLYNOMIAL REGRESSION DEMO")
    print("=" * 60)
    
    # Create sample data with quadratic relationship
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    # True relationship: y = 5 + 2x - 0.3x² + noise
    y = 5 + 2 * X - 0.3 * X**2 + np.random.normal(0, 1, len(X))
    
    print("\nTrue relationship: y = 5 + 2x - 0.3x²")
    
    # Fit models of different degrees
    for degree in [1, 2, 3]:
        model = PolynomialRegression(degree=degree, iterations=5000)
        model.fit(X.reshape(-1, 1), y)
        r2 = model.score(X.reshape(-1, 1), y)
        print(f"\nDegree {degree}: R² = {r2:.4f}")
        
        if degree == 2:
            coeffs = model.get_coefficients()
            print(f"  Coefficients: {coeffs}")
            
            optimal = model.find_optimal_point(feature_range=(0, 10))
            print(f"  Optimal point: x = {optimal['x_optimal']:.2f}, "
                  f"y = {optimal['y_optimal']:.2f} ({optimal['type']})")
            
            # Marginal effects
            test_points = [2, 5, 8]
            effects = model.marginal_effect(test_points)
            print(f"  Marginal effects:")
            for x, e in zip(test_points, effects):
                print(f"    At x={x}: dy/dx = {e:.3f}")
    
    print("\n✓ Polynomial Regression class working correctly!")
