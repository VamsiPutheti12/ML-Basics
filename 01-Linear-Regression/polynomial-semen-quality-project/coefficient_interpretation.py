"""
Clinical Interpretation of Polynomial Coefficients
===================================================
Examples of how to interpret polynomial regression coefficients
in a fertility/semen quality clinical context.
"""

import numpy as np
from polynomial_regression import PolynomialRegression

print("=" * 70)
print("CLINICAL INTERPRETATION OF POLYNOMIAL COEFFICIENTS")
print("=" * 70)
print("\nThis script demonstrates how to interpret polynomial coefficients")
print("in the context of semen quality diagnosis.\n")

# ============================================================================
# EXAMPLE 1: Age vs Sperm Concentration (Cubic Model)
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Age vs Sperm Concentration (Cubic Polynomial)")
print("=" * 70)

print("""
Model: Sperm Concentration = 150 - 2.5×Age + 0.08×Age² - 0.001×Age³

This hypothetical model shows the typical pattern of male fertility with age.
""")

# Define coefficients
beta_0 = 150   # Intercept
beta_1 = -2.5  # Linear term
beta_2 = 0.08  # Quadratic term
beta_3 = -0.001  # Cubic term

def sperm_concentration(age):
    return beta_0 + beta_1*age + beta_2*(age**2) + beta_3*(age**3)

print("📊 Coefficient Interpretation:")
print("-" * 50)
print(f"β₀ = {beta_0} (Intercept)")
print("   → Theoretical concentration at age 0 (not meaningful)")
print()
print(f"β₁ = {beta_1} (Linear term)")
print("   → Initial direction: NEGATIVE")
print("   → Suggests concentration decreases with age initially")
print("   ⚠ Cannot interpret alone with higher-order terms!")
print()
print(f"β₂ = +{beta_2} (Quadratic term)")
print("   → Positive value → U-shaped contribution")
print("   → Rate of decline SLOWS at middle ages")
print("   → Clinical: Fertility stabilizes in 30s")
print()
print(f"β₃ = {beta_3} (Cubic term)")
print("   → Negative value → Accelerated decline at extremes")
print("   → Creates S-curve with inflection point")
print("   → Clinical: Steep decline after 45")

print("\n📈 Predictions at Different Ages:")
print("-" * 50)
ages = [20, 25, 30, 35, 40, 45, 50]
for age in ages:
    conc = sperm_concentration(age)
    if conc >= 120:
        status = "✓ Excellent"
    elif conc >= 100:
        status = "○ Good"
    elif conc >= 80:
        status = "△ Moderate"
    else:
        status = "✗ Low"
    print(f"Age {age:2d}: {conc:6.1f} million/mL  {status}")

print("\n💡 Clinical Insights:")
print("-" * 50)
print("1. Gradual decline from 20s to 30s (mild effect)")
print("2. Stabilization in early 30s (quadratic compensates)")
print("3. Accelerated decline after 45 (cubic dominates)")
print("→ Male fertility relatively stable until mid-40s")

# ============================================================================
# EXAMPLE 2: BMI vs Sperm Concentration (Quadratic - Optimal Range)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: BMI vs Sperm Concentration (Quadratic - U-Shape)")
print("=" * 70)

print("""
Model: Sperm Concentration = -200 + 18×BMI - 0.35×BMI²

This shows the OPTIMAL RANGE phenomenon - both too low and too high
BMI are harmful. The negative quadratic term creates an inverted U (∩).
""")

beta_0_bmi = -200
beta_1_bmi = 18
beta_2_bmi = -0.35

def sperm_from_bmi(bmi):
    return beta_0_bmi + beta_1_bmi*bmi + beta_2_bmi*(bmi**2)

# Find optimal BMI
optimal_bmi = -beta_1_bmi / (2 * beta_2_bmi)
optimal_conc = sperm_from_bmi(optimal_bmi)

print("📊 Coefficient Interpretation:")
print("-" * 50)
print(f"β₁ = +{beta_1_bmi} (Linear term)")
print("   → Positive: concentration initially increases with BMI")
print()
print(f"β₂ = {beta_2_bmi} (Quadratic term)")
print("   → NEGATIVE: Creates INVERTED U-shape (∩)")
print("   → Key insight: There's an OPTIMAL middle range")
print()
print(f"🎯 Optimal BMI = -β₁/(2β₂) = -{beta_1_bmi}/(2×{beta_2_bmi})")
print(f"             = {optimal_bmi:.1f} kg/m²")
print(f"   → Maximum concentration: {optimal_conc:.1f} million/mL")

print("\n📈 Predictions at Different BMIs:")
print("-" * 50)
bmis = [18, 20, 22, 25, 28, 30, 35, 40]
for bmi in bmis:
    conc = max(0, sperm_from_bmi(bmi))
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    print(f"BMI {bmi:2d} ({cat:>11}): {conc:5.1f} million/mL")

print("\n💡 Clinical Insights:")
print("-" * 50)
print("• Underweight (BMI <18.5): Hormonal imbalances, nutritional deficits")
print("• Normal (BMI 18.5-25): Optimal hormonal environment")
print("• Overweight/Obese (BMI >30): ↑ Estrogen, ↓ Testosterone, scrotal heat")
print(f"• TARGET BMI for fertility: ~{optimal_bmi:.0f} kg/m²")

# ============================================================================
# EXAMPLE 3: Hours Sitting vs Sperm Motility (Accelerating Decline)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Hours Sitting vs Sperm Motility (Accelerating Decline)")
print("=" * 70)

print("""
Model: Sperm Motility (%) = 65 - 0.5×Hours - 0.15×Hours²

Both linear AND quadratic terms are NEGATIVE.
This creates an accelerating decline - each additional hour is worse!
""")

beta_0_sit = 65
beta_1_sit = -0.5
beta_2_sit = -0.15

def motility_from_sitting(hours):
    return beta_0_sit + beta_1_sit*hours + beta_2_sit*(hours**2)

def marginal_effect_sitting(hours):
    """Rate of change at given hours = β₁ + 2β₂×hours"""
    return beta_1_sit + 2*beta_2_sit*hours

print("📊 Coefficient Interpretation:")
print("-" * 50)
print(f"β₀ = {beta_0_sit} (Intercept)")
print("   → Baseline motility with minimal sitting")
print()
print(f"β₁ = {beta_1_sit} (Linear term)")
print("   → Negative: motility decreases with sitting")
print()
print(f"β₂ = {beta_2_sit} (Quadratic term)")
print("   → Also NEGATIVE: Decline ACCELERATES")
print("   → Each additional hour is MORE harmful than the last!")

print("\n📈 Predictions and Marginal Effects:")
print("-" * 60)
print(f"{'Hours':<7} {'Motility':<12} {'Rate of Change':<20} {'Status'}")
print("-" * 60)

for hours in [2, 4, 6, 8, 10, 12, 14]:
    mot = motility_from_sitting(hours)
    rate = marginal_effect_sitting(hours)
    if mot >= 50:
        status = "✓ Normal (>40%)"
    elif mot >= 40:
        status = "△ Borderline"
    else:
        status = "✗ Abnormal (<40%)"
    print(f"{hours:<7} {mot:>5.1f}%       {rate:>+.2f}%/hour           {status}")

print("\n💡 Clinical Insights:")
print("-" * 50)
print("• First few hours: Mild impact (-1% per hour)")
print("• 6-8 hours (desk job): Moderate impact (-2.5% per hour)")
print("• 10+ hours: Severe impact (-3.5%+ per hour)")
print("• RECOMMENDATION: Stand/walk breaks every 30-60 minutes")

# ============================================================================
# EXAMPLE 4: Age × BMI Interaction
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Age × BMI Interaction (Combined Risk)")
print("=" * 70)

print("""
Model: Sperm Count = 200 - 3×Age + 5×BMI - 0.05×Age² - 0.1×BMI² - 0.02×(Age×BMI)

The interaction term (-0.02) means harmful effects COMPOUND:
Age amplifies BMI damage, and vice versa.
""")

def combined_model(age, bmi):
    return (200 
            - 3*age + 5*bmi 
            - 0.05*(age**2) - 0.1*(bmi**2) 
            - 0.02*(age*bmi))

print("📊 The Interaction Term:")
print("-" * 50)
print("β_interaction = -0.02")
print("• NEGATIVE interaction → effects multiply harmfully")
print("• Young men can 'tolerate' higher BMI")
print("• Older men are MORE vulnerable to obesity's effects")
print("• Combined risk > sum of individual risks")

print("\n📈 Predictions for Different Age/BMI Combinations:")
print("-" * 65)
print(f"{'Profile':<30} {'Age':<5} {'BMI':<5} {'Sperm Count':<15} {'Risk'}")
print("-" * 65)

cases = [
    ("Young, Normal Weight", 25, 22),
    ("Young, Obese", 25, 32),
    ("Older, Normal Weight", 45, 22),
    ("Older, Obese", 45, 32),
]

for label, age, bmi in cases:
    count = combined_model(age, bmi)
    count = max(0, count)  # Can't be negative
    
    if count >= 100:
        risk = "✓ Low"
    elif count >= 50:
        risk = "△ Moderate"
    elif count >= 20:
        risk = "⚠ High"
    else:
        risk = "✗ Severe"
    
    print(f"{label:<30} {age:<5} {bmi:<5} {count:>6.1f} million    {risk}")

print("\n💡 Clinical Insights:")
print("-" * 50)
print("• Young + Normal BMI: Best outcomes")
print("• Young + Obese: Still acceptable (youth compensates)")
print("• Older + Normal BMI: Moderate decline (age effect alone)")
print("• Older + Obese: SEVERE impairment (effects compound)")
print("→ PRIORITY: Weight management becomes critical with age")

# ============================================================================
# EXAMPLE 5: Seasonal Effects
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Seasonal Effects on Semen Quality")
print("=" * 70)

print("""
Season affects spermatogenesis through temperature regulation.
The testes function optimally at 2-4°C below body temperature.
""")

print("📊 Typical Seasonal Pattern:")
print("-" * 50)

seasons = [
    ("Winter", "Cold weather", "↑ Optimal scrotal temperature", "Best"),
    ("Spring", "Warming", "○ Good conditions", "Good"),
    ("Summer", "Heat stress", "↓ Elevated scrotal temperature", "Lowest"),
    ("Fall", "Cooling", "○ Recovery begins", "Improving"),
]

for season, temp, effect, quality in seasons:
    print(f"• {season:<8}: {temp:<15} → {effect:<30} [{quality}]")

print("\n💡 Clinical Recommendations:")
print("-" * 50)
print("• Schedule fertility assessment in winter/spring if possible")
print("• Summer: Avoid hot baths, tight underwear, laptop on lap")
print("• Expect ~10-20% seasonal variation in sperm parameters")

# ============================================================================
# EXAMPLE 6: Fever Timing and Recovery
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Fever Timing and Sperm Recovery")
print("=" * 70)

print("""
High fever (>38.5°C) temporarily impairs spermatogenesis.
Sperm production cycle is ~74 days, so effects are delayed.
""")

print("📊 Recovery Timeline After Fever:")
print("-" * 50)
print("• 0-2 weeks: No visible effect yet (sperm already matured)")
print("• 2-6 weeks: Decline becomes apparent")
print("• 6-10 weeks: Maximum impact (lowest counts)")
print("• 10-14 weeks: Recovery begins")
print("• 14+ weeks: Usually full recovery")

print("\n💡 Clinical Recommendations:")
print("-" * 50)
print("• If fever in last 3 months: WAIT before fertility testing")
print("• Retest 3-4 months after high fever episode")
print("• The dataset encodes: -1 = <3 months, 0 = >3 months, 1 = None")
print("• Expected: Recent fever → lower diagnosis probability")

# ============================================================================
# SUMMARY: How to Report Polynomial Results
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: REPORTING POLYNOMIAL REGRESSION RESULTS")
print("=" * 70)

print("""
✓ DO:
  • Report overall R² and model fit metrics
  • Present CURVES, not individual coefficients
  • Calculate and report OPTIMAL POINTS
  • Show predictions at clinically relevant values
  • Include confidence intervals
  • Compare to clinical thresholds (WHO standards)

✗ DON'T:
  • "Each year increases X by β₁" (wrong with polynomials!)
  • Interpret coefficients in isolation
  • Extrapolate beyond data range
  • Ignore the sign of quadratic term

📊 Key Coefficient Patterns:
  • Positive β₂: U-shape (minimum exists)
  • Negative β₂: Inverted U (maximum exists)
  • Negative β₁ AND negative β₂: Accelerating decline
  • Negative interaction: Compound harm
""")

print("\n" + "=" * 70)
print("✅ COEFFICIENT INTERPRETATION COMPLETE!")
print("=" * 70)
print("\nThese examples demonstrate how polynomial coefficients")
print("translate to clinical insights for fertility analysis.")
