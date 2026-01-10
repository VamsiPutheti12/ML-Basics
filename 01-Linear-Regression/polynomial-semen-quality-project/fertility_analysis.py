"""
Fertility Dataset Analysis with Polynomial Regression
======================================================
Analyzes the UCI Fertility dataset using polynomial regression
to discover non-linear relationships between lifestyle factors
and semen quality diagnosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from polynomial_regression import PolynomialRegression

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 70)
print("POLYNOMIAL REGRESSION: UCI FERTILITY DATASET ANALYSIS")
print("=" * 70)

# ============================================================================
# SECTION 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n📊 Loading UCI Fertility Dataset...")

# Try to load from ucimlrepo, fallback to CSV
try:
    from ucimlrepo import fetch_ucirepo
    fertility = fetch_ucirepo(id=244)
    X = fertility.data.features
    y = fertility.data.targets
    df = pd.concat([X, y], axis=1)
    print("✓ Loaded from UCI ML Repository")
except:
    print("⚠ Could not load from ucimlrepo, using embedded data...")
    # Fallback: Create sample data matching UCI structure
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'Season': np.random.choice([-1, -0.33, 0.33, 1], n),
        'Age': np.random.uniform(0, 1, n),  # Normalized 18-36
        'Childish_diseases': np.random.choice([0, 1], n),
        'Accident': np.random.choice([0, 1], n),
        'Surgical_intervention': np.random.choice([0, 1], n),
        'High_fevers': np.random.choice([-1, 0, 1], n),
        'Alcohol_consumption': np.random.uniform(0, 1, n),
        'Smoking': np.random.choice([-1, 0, 1], n),
        'Hours_sitting': np.random.uniform(0, 1, n),
        'Diagnosis': np.random.choice(['N', 'O'], n, p=[0.88, 0.12])
    })
    print("✓ Created synthetic fertility data for demonstration")

# Display dataset info
print(f"\n📋 Dataset Shape: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"\nColumn Names: {list(df.columns)}")

# Rename columns for clarity
column_mapping = {
    df.columns[0]: 'Season',
    df.columns[1]: 'Age',
    df.columns[2]: 'Childish_diseases',
    df.columns[3]: 'Accident',
    df.columns[4]: 'Surgical_intervention',
    df.columns[5]: 'High_fevers',
    df.columns[6]: 'Alcohol_consumption',
    df.columns[7]: 'Smoking',
    df.columns[8]: 'Hours_sitting',
    df.columns[9] if len(df.columns) > 9 else 'Output': 'Diagnosis'
}
df.columns = list(column_mapping.values())

# Encode diagnosis: N (Normal) = 1, O (Altered) = 0
if df['Diagnosis'].dtype == object:
    df['Diagnosis_encoded'] = (df['Diagnosis'] == 'N').astype(int)
else:
    df['Diagnosis_encoded'] = df['Diagnosis']

print(f"\n🎯 Target Distribution:")
print(df['Diagnosis'].value_counts())

# Denormalize Age and Hours for interpretation
# Age: 0 → 18 years, 1 → 36 years
df['Age_years'] = 18 + df['Age'] * 18

# Hours sitting: 0 → 1 hour, 1 → 16 hours
df['Hours_sitting_actual'] = 1 + df['Hours_sitting'] * 15

print("\n✓ Data preprocessing complete!")

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Create figure for EDA
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Age distribution by diagnosis
ax1 = axes[0, 0]
for diag in ['N', 'O']:
    subset = df[df['Diagnosis'] == diag]['Age_years']
    ax1.hist(subset, alpha=0.6, label=f'{diag}', bins=10, edgecolor='black')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Count')
ax1.set_title('Age Distribution by Diagnosis')
ax1.legend()

# 2. Hours sitting distribution
ax2 = axes[0, 1]
for diag in ['N', 'O']:
    subset = df[df['Diagnosis'] == diag]['Hours_sitting_actual']
    ax2.hist(subset, alpha=0.6, label=f'{diag}', bins=10, edgecolor='black')
ax2.set_xlabel('Hours Sitting per Day')
ax2.set_ylabel('Count')
ax2.set_title('Sitting Hours by Diagnosis')
ax2.legend()

# 3. Season distribution
ax3 = axes[0, 2]
season_labels = {-1: 'Winter', -0.33: 'Spring', 0.33: 'Summer', 1: 'Fall'}
df['Season_name'] = df['Season'].map(lambda x: season_labels.get(round(x, 2), str(x)))
season_counts = df.groupby(['Season_name', 'Diagnosis']).size().unstack(fill_value=0)
season_counts.plot(kind='bar', ax=ax3, edgecolor='black')
ax3.set_xlabel('Season')
ax3.set_ylabel('Count')
ax3.set_title('Diagnosis by Season')
ax3.legend(title='Diagnosis')
ax3.tick_params(axis='x', rotation=45)

# 4. Alcohol consumption
ax4 = axes[1, 0]
for diag in ['N', 'O']:
    subset = df[df['Diagnosis'] == diag]['Alcohol_consumption']
    ax4.hist(subset, alpha=0.6, label=f'{diag}', bins=10, edgecolor='black')
ax4.set_xlabel('Alcohol Consumption (0=Heavy, 1=Never)')
ax4.set_ylabel('Count')
ax4.set_title('Alcohol Consumption by Diagnosis')
ax4.legend()

# 5. Smoking
ax5 = axes[1, 1]
smoking_labels = {-1: 'Daily', 0: 'Occasional', 1: 'Never'}
df['Smoking_label'] = df['Smoking'].map(lambda x: smoking_labels.get(int(x) if x in [-1, 0, 1] else x, str(x)))
smoking_counts = df.groupby(['Smoking_label', 'Diagnosis']).size().unstack(fill_value=0)
smoking_counts.plot(kind='bar', ax=ax5, edgecolor='black')
ax5.set_xlabel('Smoking Habit')
ax5.set_ylabel('Count')
ax5.set_title('Diagnosis by Smoking Habit')
ax5.legend(title='Diagnosis')
ax5.tick_params(axis='x', rotation=0)

# 6. High Fevers
ax6 = axes[1, 2]
fever_labels = {-1: '<3 months', 0: '>3 months', 1: 'None'}
df['Fever_label'] = df['High_fevers'].map(lambda x: fever_labels.get(int(x) if x in [-1, 0, 1] else x, str(x)))
fever_counts = df.groupby(['Fever_label', 'Diagnosis']).size().unstack(fill_value=0)
fever_counts.plot(kind='bar', ax=ax6, edgecolor='black')
ax6.set_xlabel('High Fevers in Last Year')
ax6.set_ylabel('Count')
ax6.set_title('Diagnosis by Fever History')
ax6.legend(title='Diagnosis')
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/01_exploratory_analysis.png")
plt.close()

# ============================================================================
# SECTION 3: POLYNOMIAL REGRESSION MODELS
# ============================================================================

print("\n" + "=" * 70)
print("POLYNOMIAL REGRESSION MODELS")
print("=" * 70)

# Features to analyze with polynomial regression
continuous_features = [
    ('Age', 'Age_years', 'Age (Years)', (18, 36)),
    ('Hours_sitting', 'Hours_sitting_actual', 'Hours Sitting/Day', (1, 16)),
    ('Alcohol_consumption', 'Alcohol_consumption', 'Alcohol (0=Heavy, 1=Never)', (0, 1)),
]

results = {}

for orig_col, plot_col, label, x_range in continuous_features:
    print(f"\n📈 Analyzing: {label}")
    
    X = df[orig_col].values.reshape(-1, 1)
    y = df['Diagnosis_encoded'].values
    
    # Compare polynomial degrees
    model_results = {}
    for degree in [1, 2, 3]:
        model = PolynomialRegression(degree=degree, iterations=3000, learning_rate=0.05)
        model.fit(X, y)
        r2 = model.score(X, y)
        model_results[degree] = {'model': model, 'r2': r2}
        print(f"  Degree {degree}: R² = {r2:.4f}")
    
    # Select best degree (highest R² without overfitting)
    best_degree = max([1, 2], key=lambda d: model_results[d]['r2'])
    best_model = model_results[best_degree]['model']
    
    results[orig_col] = {
        'label': label,
        'best_degree': best_degree,
        'best_model': best_model,
        'r2': model_results[best_degree]['r2'],
        'all_models': model_results
    }
    
    print(f"  → Best: Degree {best_degree} (R² = {model_results[best_degree]['r2']:.4f})")

# ============================================================================
# SECTION 4: POLYNOMIAL CURVE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING POLYNOMIAL CURVE VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (orig_col, plot_col, label, x_range) in enumerate(continuous_features):
    ax = axes[idx]
    
    X = df[orig_col].values
    y = df['Diagnosis_encoded'].values
    
    # Get best model
    best_model = results[orig_col]['best_model']
    best_degree = results[orig_col]['best_degree']
    r2 = results[orig_col]['r2']
    
    # Generate smooth curve
    x_smooth_norm = np.linspace(X.min(), X.max(), 100)
    y_pred = best_model.predict(x_smooth_norm.reshape(-1, 1))
    
    # Convert to actual values for plotting
    if orig_col == 'Age':
        x_actual = 18 + X * 18
        x_smooth_actual = 18 + x_smooth_norm * 18
    elif orig_col == 'Hours_sitting':
        x_actual = 1 + X * 15
        x_smooth_actual = 1 + x_smooth_norm * 15
    else:
        x_actual = X
        x_smooth_actual = x_smooth_norm
    
    # Clip predictions to valid probability range
    y_pred = np.clip(y_pred, 0, 1)
    
    # Plot jittered raw data
    y_jittered = y + np.random.normal(0, 0.03, len(y))
    ax.scatter(x_actual, y_jittered, alpha=0.4, s=50, c='gray', label='Data points')
    
    # Plot polynomial curve
    ax.plot(x_smooth_actual, y_pred, 'b-', linewidth=3, 
            label=f'Polynomial (d={best_degree})')
    
    # Calculate and plot binned means
    n_bins = 5
    bin_edges = np.percentile(x_actual, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for i in range(n_bins):
        mask = (x_actual >= bin_edges[i]) & (x_actual < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (x_actual >= bin_edges[i]) & (x_actual <= bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_means.append(y[mask].mean())
            bin_stds.append(y[mask].std() / np.sqrt(mask.sum()))
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='D', markersize=10,
                color='red', capsize=5, capthick=2, label='Bin means ± SE')
    
    # Labels
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('P(Normal Diagnosis)', fontsize=11)
    ax.set_title(f'{label}\nR² = {r2:.3f}', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_polynomial_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/02_polynomial_curves.png")
plt.close()

# ============================================================================
# SECTION 5: CATEGORICAL FEATURE ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Season effect on diagnosis probability
ax1 = axes[0, 0]
season_probs = df.groupby('Season_name')['Diagnosis_encoded'].agg(['mean', 'count', 'std'])
season_probs['se'] = season_probs['std'] / np.sqrt(season_probs['count'])
bars = ax1.bar(season_probs.index, season_probs['mean'], yerr=season_probs['se'],
               capsize=5, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], edgecolor='black')
ax1.set_ylabel('P(Normal Diagnosis)')
ax1.set_xlabel('Season')
ax1.set_title('Seasonal Effect on Semen Quality', fontweight='bold')
ax1.set_ylim(0, 1.1)
for bar, prob in zip(bars, season_probs['mean']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{prob:.0%}', ha='center', fontweight='bold')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# 2. Smoking effect
ax2 = axes[0, 1]
smoking_probs = df.groupby('Smoking_label')['Diagnosis_encoded'].agg(['mean', 'count', 'std'])
smoking_probs['se'] = smoking_probs['std'] / np.sqrt(smoking_probs['count'])
order = ['Daily', 'Occasional', 'Never'] if 'Daily' in smoking_probs.index else smoking_probs.index
smoking_probs = smoking_probs.reindex([o for o in order if o in smoking_probs.index])
bars = ax2.bar(smoking_probs.index, smoking_probs['mean'], yerr=smoking_probs['se'],
               capsize=5, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
ax2.set_ylabel('P(Normal Diagnosis)')
ax2.set_xlabel('Smoking Habit')
ax2.set_title('Smoking Effect on Semen Quality', fontweight='bold')
ax2.set_ylim(0, 1.1)
for bar, prob in zip(bars, smoking_probs['mean']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{prob:.0%}', ha='center', fontweight='bold')

# 3. Fever timing effect
ax3 = axes[1, 0]
fever_probs = df.groupby('Fever_label')['Diagnosis_encoded'].agg(['mean', 'count', 'std'])
fever_probs['se'] = fever_probs['std'] / np.sqrt(fever_probs['count'])
order = ['<3 months', '>3 months', 'None'] if '<3 months' in fever_probs.index else fever_probs.index
fever_probs = fever_probs.reindex([o for o in order if o in fever_probs.index])
bars = ax3.bar(fever_probs.index, fever_probs['mean'], yerr=fever_probs['se'],
               capsize=5, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
ax3.set_ylabel('P(Normal Diagnosis)')
ax3.set_xlabel('Fever History')
ax3.set_title('Fever Timing Effect on Semen Quality', fontweight='bold')
ax3.set_ylim(0, 1.1)
for bar, prob in zip(bars, fever_probs['mean']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{prob:.0%}', ha='center', fontweight='bold')

# 4. Binary features comparison
ax4 = axes[1, 1]
binary_features = ['Childish_diseases', 'Accident', 'Surgical_intervention']
binary_data = []
for feat in binary_features:
    yes_prob = df[df[feat] == 0]['Diagnosis_encoded'].mean()
    no_prob = df[df[feat] == 1]['Diagnosis_encoded'].mean()
    binary_data.append({'Feature': feat.replace('_', ' ').title(), 'Yes': yes_prob, 'No': no_prob})

binary_df = pd.DataFrame(binary_data)
x = np.arange(len(binary_df))
width = 0.35
bars1 = ax4.bar(x - width/2, binary_df['Yes'], width, label='Yes (Had condition)', color='#e74c3c', edgecolor='black')
bars2 = ax4.bar(x + width/2, binary_df['No'], width, label='No (Never had)', color='#2ecc71', edgecolor='black')
ax4.set_ylabel('P(Normal Diagnosis)')
ax4.set_xlabel('Medical History')
ax4.set_title('Medical History Impact on Semen Quality', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(binary_df['Feature'], rotation=15, ha='right')
ax4.legend()
ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('visualizations/03_categorical_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/03_categorical_analysis.png")
plt.close()

# ============================================================================
# SECTION 6: MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("MODEL DEGREE COMPARISON")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. R² by degree for each feature
ax1 = axes[0]
feature_names = [cont[2] for cont in continuous_features]
for idx, (orig_col, plot_col, label, x_range) in enumerate(continuous_features):
    degrees = [1, 2, 3]
    r2_scores = [results[orig_col]['all_models'][d]['r2'] for d in degrees]
    ax1.plot(degrees, r2_scores, 'o-', linewidth=2, markersize=8, label=label)

ax1.set_xlabel('Polynomial Degree', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_title('Model Fit by Polynomial Degree', fontweight='bold')
ax1.legend()
ax1.set_xticks([1, 2, 3])
ax1.grid(True, alpha=0.3)

# 2. Best model summary
ax2 = axes[1]
best_r2 = [results[cont[0]]['r2'] for cont in continuous_features]
best_deg = [results[cont[0]]['best_degree'] for cont in continuous_features]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax2.barh(feature_names, best_r2, color=colors, edgecolor='black')
ax2.set_xlabel('R² Score', fontsize=11)
ax2.set_title('Best Model Performance', fontweight='bold')
for i, (bar, deg) in enumerate(zip(bars, best_deg)):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'Degree {deg}', va='center', fontsize=10)
ax2.set_xlim(0, max(best_r2) * 1.3 if max(best_r2) > 0 else 0.5)

plt.tight_layout()
plt.savefig('visualizations/04_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/04_model_comparison.png")
plt.close()

# ============================================================================
# SECTION 7: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\n📊 Dataset Summary:")
print(f"  Total samples: {len(df)}")
print(f"  Normal diagnoses: {(df['Diagnosis'] == 'N').sum()} ({(df['Diagnosis'] == 'N').mean():.1%})")
print(f"  Altered diagnoses: {(df['Diagnosis'] == 'O').sum()} ({(df['Diagnosis'] == 'O').mean():.1%})")

print("\n📊 Best Polynomial Models:")
for orig_col, plot_col, label, x_range in continuous_features:
    r = results[orig_col]
    print(f"  {label}: Degree {r['best_degree']}, R² = {r['r2']:.4f}")
    
    # Try to find optimal point for quadratic models
    if r['best_degree'] == 2:
        try:
            opt = r['best_model'].find_optimal_point()
            if opt['x_optimal'] is not None:
                # Convert to actual units
                if orig_col == 'Age':
                    opt_val = 18 + opt['x_optimal'] * 18
                    print(f"    → Optimal age: {opt_val:.1f} years ({opt['type']})")
                elif orig_col == 'Hours_sitting':
                    opt_val = 1 + opt['x_optimal'] * 15
                    print(f"    → Optimal sitting: {opt_val:.1f} hours ({opt['type']})")
        except:
            pass

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated visualizations in 'visualizations/' folder:")
print("  1. 01_exploratory_analysis.png")
print("  2. 02_polynomial_curves.png")
print("  3. 03_categorical_analysis.png")
print("  4. 04_model_comparison.png")
print("\nRun coefficient_interpretation.py for clinical insights!")
