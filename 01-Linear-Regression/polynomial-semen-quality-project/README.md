# 📊 Polynomial Regression: Semen Quality Analysis

## 🎯 Project Overview

This project uses **Polynomial Regression** to analyze the [UCI Fertility Dataset](https://archive.ics.uci.edu/ml/datasets/Fertility) and discover how lifestyle factors, medical history, and demographics affect semen quality. We demonstrate how polynomial terms (x², x³) capture **non-linear relationships** that simple linear regression misses.

### Why This Dataset?

Semen quality is influenced by multiple factors in complex, non-linear ways:
- **Age**: Fertility doesn't decline linearly—it's stable until ~35, then accelerates
- **BMI**: Both underweight AND overweight harm fertility (U-shaped relationship)
- **Sitting hours**: The harm accelerates with more hours (not constant rate)

Polynomial regression captures these curved relationships that linear models cannot.

---

## 📚 Table of Contents

1. [The Dataset Variables](#the-dataset-variables)
2. [What is Polynomial Regression?](#what-is-polynomial-regression)
3. [The Mathematics](#the-mathematics)
4. [Variable Relationships & Clinical Intuitions](#variable-relationships--clinical-intuitions)
5. [Visualizations Explained](#visualizations-explained)
6. [Interpreting Polynomial Coefficients](#interpreting-polynomial-coefficients)
7. [Clinical Examples](#clinical-examples)
8. [How to Run](#how-to-run)

---

## 🔬 The Dataset Variables

### Overview

| Property | Value |
|----------|-------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 100 volunteers |
| **Features** | 9 input variables |
| **Target** | Semen Quality: Normal (N) / Altered (O) |
| **Collection** | Analyzed per WHO 2010 criteria |

---

### Feature Descriptions

#### 1️⃣ **Season** (When the semen sample was collected)

| Value | Season | Meaning |
|-------|--------|---------|
| -1 | Winter | Cold months (Dec-Feb) |
| -0.33 | Spring | Warming months (Mar-May) |
| 0.33 | Summer | Hot months (Jun-Aug) |
| 1 | Fall | Cooling months (Sep-Nov) |

**Clinical Relevance:**
- Testes function best at 2-4°C BELOW body temperature
- Summer heat can impair spermatogenesis (sperm production)
- Winter typically shows highest sperm quality
- Effect is temporary (recovers in cooler months)

**Expected Pattern:** Lower quality in summer, higher in winter

---

#### 2️⃣ **Age** (Age at time of analysis)

| Raw Value | Actual Age |
|-----------|------------|
| 0 | 18 years |
| 0.5 | 27 years |
| 1 | 36 years |

**Formula:** `Actual_Age = 18 + (Normalized_Value × 18)`

**Clinical Relevance:**
- Male fertility is relatively stable from 18-35
- After 35: gradual decline in sperm count and motility
- After 45: accelerated decline (DNA fragmentation increases)
- Unlike women, men can remain fertile into old age, but quality decreases

**Expected Pattern:** Slight decline with age, accelerating after 35 (cubic polynomial)

---

#### 3️⃣ **Childish Diseases** (History of childhood infections)

Includes: Chickenpox, Measles, Mumps, Polio

| Value | Meaning |
|-------|---------|
| 0 | YES - Had childhood disease |
| 1 | NO - Never had |

**Clinical Relevance:**
- **Mumps** is the most significant—can cause orchitis (testicular inflammation)
- Orchitis during puberty may permanently damage testicular tissue
- Chickenpox/measles have less direct impact on fertility
- Modern vaccines have reduced this risk significantly

**Expected Pattern:** History of disease (especially mumps) → lower quality

---

#### 4️⃣ **Accident/Trauma** (Serious physical injury history)

| Value | Meaning |
|-------|---------|
| 0 | YES - Had serious trauma |
| 1 | NO - Never had |

**Clinical Relevance:**
- Testicular trauma can cause:
  - Direct tissue damage
  - Blood-testis barrier disruption (autoimmune attack on sperm)
  - Varicocele formation (varicose veins in scrotum)
- Severity and location of trauma matters
- Even healed injuries may have permanent effects

**Expected Pattern:** Trauma history → higher risk of altered semen

---

#### 5️⃣ **Surgical Intervention** (Past surgeries)

| Value | Meaning |
|-------|---------|
| 0 | YES - Had surgery |
| 1 | NO - Never had |

**Clinical Relevance:**
- Relevant surgeries include:
  - Hernia repair (may damage vas deferens)
  - Undescended testicle correction
  - Varicocele repair
  - Prostate/bladder surgery
- General anesthesia may temporarily affect hormones
- Post-surgical infections can cause scarring

**Expected Pattern:** Surgical history → slightly lower quality (depends on surgery type)

---

#### 6️⃣ **High Fevers** (Fever episodes in last year)

| Value | Meaning | Timing |
|-------|---------|--------|
| -1 | YES - Recent | Less than 3 months ago |
| 0 | YES - Past | More than 3 months ago |
| 1 | NO | No fever in past year |

**Clinical Relevance:**
- High fever (>38.5°C) impairs spermatogenesis
- **Sperm production cycle is 74 days**, so:
  - Week 1-2 after fever: No visible effect (sperm already matured)
  - Week 3-8: Decline becomes apparent
  - Week 8-12: Maximum impact (lowest counts)
  - Week 12+: Recovery begins
- This is why <3 months is worse than >3 months

**Expected Pattern:** Recent fever (-1) → lowest quality; No fever (1) → highest quality

---

#### 7️⃣ **Alcohol Consumption** (Drinking frequency)

| Value | Meaning |
|-------|---------|
| 0 | Heavy (several times daily) |
| 0.2 | Daily |
| 0.4 | Several times weekly |
| 0.6 | Once weekly |
| 0.8 | Rarely |
| 1 | Never |

**Clinical Relevance:**
- Alcohol reduces testosterone production
- Heavy drinking causes liver damage → increased estrogen
- Chronic use shrinks testes and reduces sperm production
- Moderate drinking (1-2 drinks occasionally) has minimal effect
- Effect is dose-dependent and reversible with abstinence

**Expected Pattern:** Linear or slight polynomial—more alcohol → worse quality

---

#### 8️⃣ **Smoking Habit**

| Value | Meaning |
|-------|---------|
| -1 | Daily smoker |
| 0 | Occasional smoker |
| 1 | Never smoked |

**Clinical Relevance:**
- Tobacco smoke contains 7,000+ chemicals, many toxic to sperm
- Effects include:
  - Reduced sperm count (10-17% lower in smokers)
  - Lower motility (sperm swim slower)
  - Abnormal morphology (shape defects)
  - DNA damage (affects offspring health)
- Dose-dependent: More cigarettes = worse effects
- Partially reversible after quitting (3-6 months)

**Expected Pattern:** Non-smokers have best outcomes; daily smokers worst

---

#### 9️⃣ **Hours Sitting per Day** (Sedentary time)

| Raw Value | Actual Hours |
|-----------|--------------|
| 0 | 1 hour/day |
| 0.5 | 8 hours/day |
| 1 | 16 hours/day |

**Formula:** `Actual_Hours = 1 + (Normalized_Value × 15)`

**Clinical Relevance:**
- Prolonged sitting causes:
  - **Scrotal heating** (testes pressed against body)
  - Reduced blood flow to reproductive organs
  - Hormonal changes from sedentary lifestyle
- Effects are **non-linear**—each additional hour is worse than the last
- Desk workers at high risk (8+ hours)
- Recommendation: Stand/walk breaks every 30-60 minutes

**Expected Pattern:** Accelerating decline (negative quadratic)—first hours mild, later hours severe

---

### Target Variable

#### 🎯 **Diagnosis** (Semen Quality Classification)

| Value | Meaning | WHO Criteria |
|-------|---------|--------------|
| N | Normal | Meets all WHO 2010 standards |
| O | Altered | Below normal in at least one parameter |

**WHO 2010 Normal Values:**
- Sperm concentration: ≥15 million/mL
- Total motility: ≥40%
- Progressive motility: ≥32%
- Normal morphology: ≥4%
- Volume: ≥1.5 mL

For regression, we encode: **N=1, O=0** (probability of normal diagnosis)

---

## 🔍 What is Polynomial Regression?

### The Problem with Linear Regression

Linear regression assumes a straight-line relationship:

```
ŷ = β₀ + β₁x
```

But many real relationships are **curved**:
- Age vs Fertility: Peaks in mid-20s, then declines
- BMI vs Health: Optimal range exists (not too low, not too high)
- Exercise vs Performance: Diminishing returns at extremes

### The Solution: Add Polynomial Terms

**Quadratic (degree 2):**
```
ŷ = β₀ + β₁x + β₂x²
```
Creates **U-shaped** or **inverted U** curves.

**Cubic (degree 3):**
```
ŷ = β₀ + β₁x + β₂x² + β₃x³
```
Creates **S-curves** with multiple inflection points.

### Visual Comparison

| Degree | Shape | Example Use Case |
|--------|-------|------------------|
| 1 (Linear) | Straight line | Constant rate of change |
| 2 (Quadratic) | U or ∩ shaped | Optimal middle range |
| 3 (Cubic) | S-curve | Age-related decline |
| 4+ | Complex waves | Rarely needed |

---

## 📐 The Mathematics

### Polynomial Feature Transformation

Original features are transformed:
```
Original: x
Degree 2: [x, x²]
Degree 3: [x, x², x³]
```

### Finding Optimal Points

For quadratic `ŷ = β₀ + β₁x + β₂x²`:

**Optimal x = -β₁ / (2β₂)**

- If β₂ < 0: Maximum (inverted U)
- If β₂ > 0: Minimum (U-shape)

### Marginal Effect (Rate of Change)

The derivative shows how y changes with x:
```
dŷ/dx = β₁ + 2β₂x + 3β₃x²
```

The effect of x on y **depends on where you are** on the curve!

---

## 🔗 Variable Relationships & Clinical Intuitions

### 1. Age × Everything

Age **amplifies** the negative effects of other factors:
- Older + Smoker = Much worse than young smoker
- Older + Obese = Much worse than young obese
- Older + Sedentary = Much worse than young sedentary

**Why?** Age reduces the body's compensatory mechanisms.

### 2. Lifestyle Factor Clustering

Bad habits cluster together:
- Heavy drinkers often smoke
- Smokers often sit more
- Sedentary people have higher BMI

This creates **compound effects** that polynomial regression can model.

### 3. Medical History as Vulnerabilities

Prior medical issues create sensitivities:
- Childhood mumps + Current fever = Severe impact
- Past trauma + Current sitting = Blood flow issues
- Surgery + Age = Increased complications

### 4. Seasonal + Lifestyle Interaction

Summer + Sitting = Worst combination
- Already elevated ambient temperature
- Sitting adds direct scrotal heating
- Effect is multiplicative

### 5. Recovery Patterns

Some effects are reversible:
- Stop smoking → 3-6 months to improve
- Fever recovery → 3-4 months
- Weight loss → 2-3 months for hormonal normalization
- Reduce sitting → Immediate improvement

---

## 📊 Visualizations Explained

### Plot 1: Exploratory Analysis (`01_exploratory_analysis.png`)

**What it shows:**
- Distribution of each feature by diagnosis (Normal vs Altered)
- Histograms reveal imbalances and patterns

**How to interpret:**
- Look for separation between N and O groups
- Overlapping distributions = weak predictor
- Clear separation = strong predictor

**Key insights:**
- More Normal diagnoses overall (~88% N, 12% O)
- Some seasons have higher Normal rates
- Smoking shows visible separation

---

### Plot 2: Polynomial Curves (`02_polynomial_curves.png`)

**What it shows:**
- Fitted polynomial curves for continuous features
- Gray dots: Actual data (jittered for visibility)
- Blue line: Polynomial fit
- Red diamonds: Bin means with error bars

**How to interpret:**
- The CURVE shows the relationship shape
- Upward curve = increasing probability of Normal
- Downward curve = decreasing probability
- U-shape = optimal middle range
- Red diamonds validate the curve fits the data

**Key insights:**
- Age: Slight downward trend (older → lower probability)
- Sitting: U-shaped or declining trend
- Alcohol: Higher values (less drinking) → better outcomes

---

### Plot 3: Categorical Analysis (`03_categorical_analysis.png`)

**What it shows:**
- Bar charts for categorical features
- Height = Probability of Normal diagnosis
- Error bars = Standard error (uncertainty)

**How to interpret:**
- Taller bars = Better outcomes
- Overlapping error bars = Not significantly different
- Clear height difference = Significant effect

**Key insights:**
- Season effect visible (best in winter?)
- Never-smokers have highest Normal probability
- Recent fever (<3 months) has lowest probability
- Binary features (disease, trauma) show comparison

---

### Plot 4: Model Comparison (`04_model_comparison.png`)

**What it shows:**
- R² scores for different polynomial degrees
- Best model selection per feature

**How to interpret:**
- Higher R² = Better fit
- If R² increases then decreases with degree → Overfitting
- Choose degree where R² plateaus

**Key insights:**
- Most features work well with degree 2
- Going to degree 3+ rarely helps (small dataset)
- Low R² overall due to noisy binary outcome

---

## 🧮 Interpreting Polynomial Coefficients

### The Key Rule

❌ **WRONG:** "Each year increases X by β₁"  
✅ **RIGHT:** "The relationship follows a curved pattern"

You CANNOT interpret polynomial coefficients individually!

### Coefficient Meanings

| Coefficient | Name | What It Controls |
|-------------|------|-----------------|
| β₀ | Intercept | Baseline level |
| β₁ | Linear | Initial direction |
| β₂ | Quadratic | Curvature (U or ∩) |
| β₃ | Cubic | Additional complexity |

### Sign Interpretation

**Quadratic term (β₂):**
- β₂ > 0: U-shaped (minimum exists)
- β₂ < 0: Inverted U (maximum exists)

**Both β₁ and β₂ negative:**
- Accelerating decline (gets worse faster)

---

## 📈 Clinical Examples

### Example 1: Age vs Sperm Concentration

**Model:** `Concentration = 150 - 2.5×Age + 0.08×Age² - 0.001×Age³`

| Age | Concentration | Status |
|-----|---------------|--------|
| 20 | 124 million/mL | Excellent |
| 30 | 120 million/mL | Excellent |
| 40 | 114 million/mL | Good |
| 50 | 100 million/mL | Moderate |

**Insight:** Decline accelerates after age 45.

---

### Example 2: BMI Optimal Range

**Model:** `Concentration = -200 + 18×BMI - 0.35×BMI²`

**Optimal BMI:** -18 / (2 × -0.35) = **25.7 kg/m²**

**Insight:** Both underweight and obese harm fertility. Optimal is slightly above "normal."

---

### Example 3: Sitting Hours (Accelerating Harm)

**Model:** `Motility = 65 - 0.5×Hours - 0.15×Hours²`

| Hours | Decline Rate |
|-------|-------------|
| 2 | -1.1%/hour |
| 6 | -2.3%/hour |
| 10 | -3.5%/hour |

**Insight:** First hours are mild; 10+ hours is severely harmful.

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Analysis

```bash
cd polynomial-semen-quality-project

# Full analysis (generates visualizations)
python fertility_analysis.py

# Clinical interpretation examples
python coefficient_interpretation.py

# Test polynomial regression class
python polynomial_regression.py
```

### Output Files

```
visualizations/
├── 01_exploratory_analysis.png    # Distribution by diagnosis
├── 02_polynomial_curves.png       # Fitted polynomial models
├── 03_categorical_analysis.png    # Categorical feature effects
└── 04_model_comparison.png        # Model degree comparison
```

---

## 📚 Key Takeaways

1. **Polynomial regression captures curves** that linear models miss

2. **Don't interpret coefficients individually** — look at the full curve shape

3. **Quadratic terms reveal optimal ranges** (U-shapes or inverted U)

4. **Marginal effects vary** — harm accelerates at extremes

5. **Age amplifies other risk factors** — older patients are more vulnerable

6. **Lifestyle factors compound** — smoking + sitting + drinking = multiplicative harm

7. **Some effects are reversible** — stopping smoking improves quality in 3-6 months

8. **Seasonal effects are real** — consider timing of fertility assessments

---

## 📖 References

- UCI Fertility Dataset: https://archive.ics.uci.edu/ml/datasets/Fertility
- WHO Laboratory Manual for Semen Analysis (2010)
- Gil, D. & Girela, J. (2012). Predicting seminal quality with AI methods.

---

**Ready to explore?** Run `python fertility_analysis.py` to generate visualizations! 🔬
