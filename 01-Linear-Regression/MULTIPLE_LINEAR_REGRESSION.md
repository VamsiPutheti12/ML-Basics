# Multiple Linear Regression with California Housing Dataset

## 📊 Overview

This example demonstrates **Multiple Linear Regression** using real-world data - the California Housing dataset. Unlike simple linear regression (one feature), this uses **8 features** to predict house prices.

## 🎯 What You'll Learn

- How to handle multiple features (multiple linear regression)
- Feature normalization and why it's important
- Working with real-world datasets
- Interpreting feature importance
- Model evaluation with multiple metrics

---

## 📁 Dataset: California Housing

**Source**: Built into scikit-learn  
**Samples**: 20,640 houses  
**Target**: Median house value (in $100,000s)

### Features (8 total):

| Feature | Description |
|---------|-------------|
| **MedInc** | Median income in block group |
| **HouseAge** | Median house age in block group |
| **AveRooms** | Average number of rooms per household |
| **AveBedrms** | Average number of bedrooms per household |
| **Population** | Block group population |
| **AveOccup** | Average number of household members |
| **Latitude** | Block group latitude |
| **Longitude** | Block group longitude |

### ⚠️ Important: Dataset Time Period

**Data Collection**: 1990 U.S. Census  
**Year**: 1990 (over 30 years old)  
**Source**: U.S. Census Bureau

> **Note**: This dataset uses **1990 prices**, which are significantly lower than current California housing prices. In 1990, the median California home was ~$207,000. Today (2026), it's over $800,000. This is a historical dataset used for educational purposes.

**Why use old data?**
- ✅ Clean and well-preprocessed
- ✅ Built into sklearn (easy access)
- ✅ Widely used in ML education
- ✅ Good for learning fundamentals

**Want recent data?** See [recent_housing_regression.py](recent_housing_regression.py) for a 2015-2024 housing example with modern prices!

---

## 🔢 The Math

### Multiple Linear Regression Equation

For **n features**:

```
ŷ = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b
```

Or in vector form:
```
ŷ = W^T · X + b
```

Where:
- **ŷ** = predicted house price
- **W** = weights vector [w₁, w₂, ..., wₙ]
- **X** = features vector [x₁, x₂, ..., xₙ]
- **b** = bias (intercept)

### Gradient Descent for Multiple Features

**Update rules:**
```
W = W - α · (1/m) · X^T · (ŷ - y)
b = b - α · (1/m) · Σ(ŷ - y)
```

Where:
- **α** = learning rate
- **m** = number of training samples
- **X^T** = transpose of feature matrix

---

## 🔑 Key Concepts

### 1. Feature Normalization

**Why it's crucial:**
- Features have different scales (e.g., Latitude: -124 to -114, MedInc: 0 to 15)
- Without normalization, gradient descent converges very slowly or not at all
- Large-scale features dominate the cost function

**Solution: StandardScaler**
```
x_normalized = (x - mean) / std_deviation
```

This transforms all features to have:
- Mean = 0
- Standard deviation = 1

### 2. Feature Importance

The **magnitude of weights** tells us feature importance:
- **Large positive weight**: Feature strongly increases price
- **Large negative weight**: Feature strongly decreases price
- **Small weight**: Feature has little impact

### 3. Model Evaluation

**R² Score (Coefficient of Determination)**
- Range: 0 to 1 (higher is better)
- Interpretation: % of variance explained by the model
- Example: R² = 0.59 means model explains 59% of price variation

**RMSE (Root Mean Squared Error)**
- Average prediction error in same units as target
- Example: RMSE = 0.73 means average error of $73,000

---

## 📈 Results

### Model Performance

**Training Set:**
- R² Score: ~0.590
- RMSE: ~$73k

**Test Set:**
- R² Score: ~0.567
- RMSE: ~$74k

### Feature Importance (Learned Weights)

Based on the trained model:

1. **MedInc** (Median Income) - **Most Important** ✅
   - Strong positive correlation with price
   - Higher income areas = higher prices

2. **Latitude/Longitude** - Geographic location matters
   - Coastal areas tend to be more expensive

3. **AveOccup** - Negative correlation
   - More crowded = lower prices

4. **HouseAge** - Moderate positive effect
   - Older houses in California can be valuable

---

## 📊 Visualizations Generated

### 1. **multiple_linear_regression_results.png**

Six-panel visualization showing:

**Panel 1: Cost Function**
- Shows gradient descent convergence
- Cost decreases from ~2.5 to ~0.27

**Panel 2 & 3: Predictions vs Actual**
- Training set (blue) and Test set (green)
- Points near diagonal line = good predictions
- R² scores displayed

**Panel 4 & 5: Residuals**
- Should be randomly distributed around zero
- Patterns indicate model issues

**Panel 6: Feature Importance**
- Bar chart of learned weights
- Green = positive impact, Red = negative impact
- MedInc has the largest weight

### 2. **feature_correlations.png**

Eight scatter plots showing each feature vs price:
- **MedInc**: Clear positive correlation
- **Latitude/Longitude**: Geographic patterns visible
- **HouseAge**: Slight positive trend
- **Population**: Scattered, weak correlation

---

## 🚀 How to Run

```bash
# Navigate to the folder
cd 01-Linear-Regression

# Run the script
python multiple_linear_regression.py
```

**Requirements:**
```bash
pip install numpy matplotlib scikit-learn pandas
```

---

## 💡 Key Takeaways

✅ **Multiple features improve predictions**
- Using 8 features vs 1 gives much better accuracy

✅ **Feature normalization is essential**
- Without it, gradient descent won't converge properly

✅ **Not all features are equally important**
- Median income is the strongest predictor
- Some features (like AveBedrms) have minimal impact

✅ **Real-world data is messy**
- R² of 0.57 is decent but not perfect
- Many factors affect house prices beyond these 8 features

✅ **Visualization helps understanding**
- Residual plots show model limitations
- Feature correlations reveal relationships

---

## 🔍 Comparison with Simple Linear Regression

| Aspect | Simple | Multiple |
|--------|--------|----------|
| **Features** | 1 | 8 |
| **Equation** | y = wx + b | y = W^T·X + b |
| **Complexity** | Low | Medium |
| **Accuracy** | Lower | Higher |
| **Interpretability** | Easy | Moderate |
| **Normalization** | Optional | Essential |

---

## 🎓 Next Steps

1. **Experiment with hyperparameters**
   - Try different learning rates (0.001, 0.1)
   - Increase iterations (2000, 5000)

2. **Feature engineering**
   - Create new features (e.g., rooms per bedroom)
   - Try polynomial features

3. **Compare with sklearn**
   - Use `LinearRegression` from sklearn
   - Compare results with your implementation

4. **Try other datasets**
   - Boston Housing (deprecated but available)
   - Your own CSV data

---

## 📚 Related Files

- **[simple_demo.py](simple_demo.py)** - Simple linear regression (1 feature)
- **[linear_regression.py](linear_regression.py)** - Full implementation with class
- **[recent_housing_regression.py](recent_housing_regression.py)** - **NEW!** Recent housing data (2015-2024)
- **[README.md](README.md)** - Complete theory and mathematics

---

**Ready to dive deeper?** Try modifying the code to add more features or experiment with different datasets! 🎉
