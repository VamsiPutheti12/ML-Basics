"""
Evaluation Metrics for Classification
=======================================
ROC Curve, AUC, Precision, Recall, F1-Score, and Confusion Matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# ============================================
# Load and Prepare Data
# ============================================
print("=" * 50)
print("CLASSIFICATION EVALUATION METRICS")
print("=" * 50)

# Load breast cancer dataset
data = load_breast_cancer()
X_full = data.data[:, [0, 1]]  # mean radius, mean texture
y = data.target  # 0=malignant, 1=benign

# Normalize
X = (X_full - X_full.min(axis=0)) / (X_full.max(axis=0) - X_full.min(axis=0))

# Train/test split
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nDataset: Breast Cancer")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================
# Train Logistic Regression
# ============================================
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Initialize
w = np.zeros(2)
b = 0.0
learning_rate = 1.0
iterations = 2000
m = len(X_train)

print(f"\nTraining logistic regression...")
for i in range(iterations):
    z = np.dot(X_train, w) + b
    y_pred = sigmoid(z)
    error = y_pred - y_train
    w -= learning_rate * (1/m) * np.dot(X_train.T, error)
    b -= learning_rate * (1/m) * np.sum(error)

# Get predicted probabilities for test set
y_test_prob = sigmoid(np.dot(X_test, w) + b)
y_test_pred = (y_test_prob >= 0.5).astype(int)

print("Training complete!")

# ============================================
# Confusion Matrix Components
# ============================================
TP = np.sum((y_test == 1) & (y_test_pred == 1))  # True Positive
TN = np.sum((y_test == 0) & (y_test_pred == 0))  # True Negative
FP = np.sum((y_test == 0) & (y_test_pred == 1))  # False Positive
FN = np.sum((y_test == 1) & (y_test_pred == 0))  # False Negative

print(f"\n{'='*50}")
print("CONFUSION MATRIX")
print(f"{'='*50}")
print(f"\n                    Predicted")
print(f"                  Neg    Pos")
print(f"Actual Negative   {TN:3d}    {FP:3d}")
print(f"       Positive   {FN:3d}    {TP:3d}")

# ============================================
# Evaluation Metrics
# ============================================
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Also called Sensitivity/TPR
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\n{'='*50}")
print("EVALUATION METRICS")
print(f"{'='*50}")
print(f"\nAccuracy:    {accuracy:.4f}  (Correct / Total)")
print(f"Precision:   {precision:.4f}  (TP / Predicted Positive)")
print(f"Recall:      {recall:.4f}  (TP / Actual Positive)")
print(f"F1-Score:    {f1_score:.4f}  (Harmonic mean of P & R)")
print(f"Specificity: {specificity:.4f}  (TN / Actual Negative)")

# ============================================
# ROC Curve and AUC
# ============================================
# Generate ROC curve by varying threshold
thresholds = np.linspace(0, 1, 100)
tpr_list = []  # True Positive Rate (Recall)
fpr_list = []  # False Positive Rate (1 - Specificity)

for thresh in thresholds:
    y_pred_thresh = (y_test_prob >= thresh).astype(int)
    
    tp = np.sum((y_test == 1) & (y_pred_thresh == 1))
    fn = np.sum((y_test == 1) & (y_pred_thresh == 0))
    fp = np.sum((y_test == 0) & (y_pred_thresh == 1))
    tn = np.sum((y_test == 0) & (y_pred_thresh == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

tpr_array = np.array(tpr_list)
fpr_array = np.array(fpr_list)

# Calculate AUC using trapezoidal rule
sorted_indices = np.argsort(fpr_array)
fpr_sorted = fpr_array[sorted_indices]
tpr_sorted = tpr_array[sorted_indices]
auc = np.trapz(tpr_sorted, fpr_sorted)

print(f"\n{'='*50}")
print("ROC & AUC")
print(f"{'='*50}")
print(f"\nAUC (Area Under Curve): {auc:.4f}")
print(f"  - AUC = 1.0: Perfect classifier")
print(f"  - AUC = 0.5: Random guessing")
print(f"  - AUC < 0.5: Worse than random")

# ============================================
# Visualizations
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot 1: Confusion Matrix Heatmap
ax1 = axes[0]
cm = np.array([[TN, FP], [FN, TP]])
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Malignant', 'Benign'])
ax1.set_yticklabels(['Malignant', 'Benign'])
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('Actual', fontsize=11)
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        ax1.text(j, i, str(cm[i, j]), ha='center', va='center', 
                fontsize=16, fontweight='bold', color=color)

# Plot 2: ROC Curve
ax2 = axes[1]
ax2.plot(fpr_array, tpr_array, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random (AUC = 0.5)')
ax2.fill_between(fpr_sorted, tpr_sorted, alpha=0.2)
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)

# Plot 3: Metrics Bar Chart
ax3 = axes[2]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1_score]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = ax3.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

ax3.set_ylim(0, 1.15)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Classification Metrics', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('evaluation_metrics_result.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'evaluation_metrics_result.png'")
plt.show()

print(f"\n{'='*50}")
print("EVALUATION COMPLETE!")
print(f"{'='*50}")
