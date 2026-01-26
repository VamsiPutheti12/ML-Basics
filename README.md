# ML-Basics: A Beginner's Journey Through Machine Learning

Welcome to **ML-Basics**! This repository is a comprehensive guide to understanding Machine Learning algorithms from the ground up. Each algorithm includes detailed theory, mathematical foundations, and clean Python implementations.

## What is Machine Learning?

**Machine Learning (ML)** is a subset of Artificial Intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed for every scenario.

### How Does Machine Learning Work?

Instead of writing specific rules, we:
1. **Feed data** to an algorithm
2. Let the algorithm **find patterns** in the data
3. Use these patterns to **make predictions** on new, unseen data

### Types of Machine Learning

**1. Supervised Learning**
- Learn from labeled data (input-output pairs)
- Examples: Linear Regression, Logistic Regression, Decision Trees
- Use cases: Price prediction, spam detection, image classification

**2. Unsupervised Learning**
- Find patterns in unlabeled data
- Examples: K-Means Clustering, PCA
- Use cases: Customer segmentation, anomaly detection

**3. Reinforcement Learning**
- Learn through trial and error with rewards/penalties
- Examples: Q-Learning, Deep Q-Networks
- Use cases: Game AI, robotics, recommendation systems

## Repository Structure

```
ML-Basics/
├── 01-Linear-Regression/
│   ├── README.md                                # Complete theory with math
│   ├── MULTIPLE_LINEAR_REGRESSION.md            # Multiple features guide
│   ├── linear_regression.py                     # Full class implementation
│   ├── simple_demo.py                           # Basic 1-feature demo
│   ├── multiple_linear_regression.py            # 1990 California housing
│   ├── recent_housing_regression.py             # 2015-2024 housing data
│   ├── deploy-housing/                          # Streamlit deployment
│   └── polynomial-semen-quality-project/        # Polynomial regression
│
├── 02-Logistic-Regression/
│   ├── README.md                                # Complete theory with math
│   ├── simple_demo.py                           # Basic binary classification
│   ├── cancer_prediction.py                     # Breast cancer classification
│   ├── multiclass_iris.py                       # Softmax regression (3 classes)
│   ├── evaluation_metrics.py                    # ROC, AUC, F1-Score
│   └── sklearn_comparison.py                    # Scratch vs sklearn validation
│
├── 03-Regularization/
│   ├── README.md                                # Complete theory with math
│   ├── underfitting_demo.py                     # Linear on quadratic data
│   ├── overfitting_demo.py                      # Polynomial degree comparison
│   ├── ridge_regression.py                      # L2 regularization
│   ├── lasso_regression.py                      # L1 regularization
│   ├── elastic_net.py                           # L1 + L2 combined
│   └── regularization_comparison.py             # Side-by-side comparison
│
├── 04-Decision-Trees/ (Coming soon)
└── ... (More algorithms)
```

## Algorithms Covered

### ✅ Implemented

1. **[Linear Regression](01-Linear-Regression/)** - Predicting continuous values
   - Simple Linear Regression (1 feature)
   - Multiple Linear Regression (8 features)
   - Historical dataset (1990 California Housing)
   - Recent dataset (2015-2024 USA Housing)
   - **Polynomial Regression** (capturing non-linear relationships)
   - Streamlit web app deployment
   - Comprehensive visualizations

2. **[Logistic Regression](02-Logistic-Regression/)** - Binary & multiclass classification
   - Simple binary classification (pass/fail demo)
   - Cancer prediction (malignant vs benign)
   - **Multiclass classification** with Softmax (Iris dataset)
   - **Evaluation metrics**: ROC Curve, AUC, Precision, Recall, F1-Score
   - Sklearn comparison to validate implementation
   - Complete math documentation (sigmoid, cross-entropy, gradients)

3. **[Regularization](03-Regularization/)** - Preventing overfitting
   - Underfitting vs Overfitting demos
   - **Ridge Regression (L2)**: Weight shrinkage
   - **Lasso Regression (L1)**: Feature selection
   - **Elastic Net**: L1 + L2 combined
   - Bias-variance tradeoff visualization
   - Complete math documentation

### 🚧 Coming Soon
4. **Decision Trees** - Tree-based decisions
5. **K-Nearest Neighbors** - Instance-based learning
5. **Support Vector Machines** - Maximum margin classification
6. **Neural Networks** - Deep learning basics
7. **K-Means Clustering** - Unsupervised clustering
8. **Principal Component Analysis (PCA)** - Dimensionality reduction

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.7+
numpy
matplotlib
scikit-learn (for datasets)
pandas (for data manipulation)
seaborn (for visualizations)
streamlit (for web app deployment)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/VamsiPutheti12/ML-Basics.git
cd ML-Basics

# Install dependencies
pip install numpy matplotlib scikit-learn pandas seaborn
```

### Quick Start
```bash
# Linear Regression demos
cd 01-Linear-Regression
python simple_demo.py
python linear_regression.py
python multiple_linear_regression.py

# Logistic Regression demos
cd ../02-Logistic-Regression
python simple_demo.py
python cancer_prediction.py
python multiclass_iris.py
python evaluation_metrics.py
python sklearn_comparison.py
```

## 🎓 Learning Path

If you're new to Machine Learning, I recommend following this order:

1. **Start with Linear Regression** - Understand the basics of supervised learning
2. **Move to Logistic Regression** - Learn classification
3. **Explore Decision Trees** - Understand non-linear models
4. **Dive into Neural Networks** - Foundation for deep learning


## 🤝 Contributing

This is a learning repository, but suggestions and improvements are welcome! Feel free to:
- Report issues
- Suggest new algorithms
- Improve explanations
- Add more examples

## 📝 License

This project is open source and available for educational purposes.

## 🌟 Acknowledgments

This repository is created as a learning journey through Machine Learning fundamentals.

---

**Happy Learning! 🚀**

*Remember: The best way to learn ML is by implementing algorithms from scratch!*
