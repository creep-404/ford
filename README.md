# Ford Car Dataset — EDA, Preprocessing & ML Pipeline

## Overview

This project performs a complete exploratory data analysis (EDA),
preprocessing pipeline, and machine learning model training on the
**Ford Car Dataset**, a used-car listings dataset containing
**17,966 records** and **9 columns**. The goal is to clean the data,
understand its structure and distributions, identify relationships
between features, and build a Linear Regression model to predict
used Ford car resale price.

**Dataset columns:** `model`, `year`, `price`, `transmission`, `mileage`,
`fuelType`, `tax`, `mpg`, `engineSize`

**Tools used:** `pandas`, `numpy`, `matplotlib`, `seaborn`,
`scikit-learn` (`StandardScaler`, `LinearRegression`, `r2_score`),
`joblib`

---

## Table of Contents

1. [Q1 — Data Loading & Initial Analysis](#q1--data-loading--initial-analysis)
2. [Q2 — Missing & Duplicate Values](#q2--missing--duplicate-values)
3. [Q3 — Statistical Summary](#q3--statistical-summary)
4. [Q4 — Histograms of Numeric Features](#q4--histograms-of-numeric-features)
5. [Q5 — Count Plots of Categorical Features](#q5--count-plots-of-categorical-features)
6. [Q6 — Correlation Heatmap](#q6--correlation-heatmap)
7. [Q7 — Feature Identification](#q7--feature-identification)
8. [Q8 — Encoding Categorical Variables](#q8--encoding-categorical-variables)
9. [Q9 — Feature Scaling](#q9--feature-scaling)
10. [Q10 — Complete Preprocessing Pipeline](#q10--complete-preprocessing-pipeline)
11. [Q11 — Preparing Features (X and Y)](#q11--preparing-features-x-and-y)
12. [Q12 — One-Hot Encoding](#q12--one-hot-encoding)
13. [Q13 — Feature Scaling (StandardScaler)](#q13--feature-scaling-standardscaler)
14. [Q14 — Train-Test Split](#q14--train-test-split)
15. [Q15 — Building Linear Regression Model](#q15--building-linear-regression-model)
16. [Q16 — Making Predictions](#q16--making-predictions)
17. [Q17 — Model Evaluation using R² Score](#q17--model-evaluation-using-r-score)
18. [Q18 — Saving the Model](#q18--saving-the-model)
19. [Project Files](#project-files)
20. [Key Takeaways](#key-takeaways)

---

## Q1 — Data Loading & Initial Analysis

Loaded the dataset with `pandas.read_csv()` and inspected its structure
using `head(10)`, `tail(5)`, `.shape`, and `.dtypes`.

**Findings:**
- Shape: **17,966 rows × 9 columns**
- Data types: 4 integer columns (`year`, `price`, `mileage`, `tax`), 2
  float columns (`mpg`, `engineSize`), 3 categorical/string columns
  (`model`, `transmission`, `fuelType`)
- No ID column is present — pandas default integer index serves as the
  row identifier
- `price` stands out as the natural target variable, with all other
  columns describing car attributes

---

## Q2 — Missing & Duplicate Values

Checked for missing values with `df.isnull().sum()` and duplicates with
`df.duplicated().sum()`.

**Findings:**
- **Missing values:** 0 across every column — the dataset required no
  imputation
- **Duplicate rows:** 154 exact duplicates found, likely from repeated
  listings in the source data
- **Handling:** Removed with `drop_duplicates()` and reset the index;
  final row count: **17,812**

---

## Q3 — Statistical Summary

Generated `df.describe()` and examined min/max/mean/median for `price`,
`mileage`, and `year`.

| Feature | Min | Max | Mean | Median |
|---|---|---|---|---|
| price | 495 | 54,995 | 12,269.56 | 11,288.00 |
| mileage | 1 | 177,644 | 23,381.15 | 18,277.00 |
| year | 1996 | 2060 | 2016.86 | 2017.00 |

**Findings:**
- `price` and `mileage` are both **right-skewed** (mean > median)
- `year` is concentrated around recent model years
- **Data quality flag:** the maximum `year` value is **2060** — an
  impossible future date, almost certainly a data-entry error that
  should be corrected or removed before modeling

---

## Q4 — Histograms of Numeric Features

Plotted histograms for `price`, `mileage`, `year`, `engineSize`, and
`mpg`.

📊 *See: `Q4_histograms.png`*

**Findings:**
- **price:** right-skewed, most listings under ~20,000 with a long tail
  toward ~55,000
- **mileage:** right-skewed, most cars under ~40,000 miles with a long
  tail to ~177,000
- **year:** concentrated in 2016–2020, with the 2060 outlier clearly
  visible as an isolated bar
- **engineSize:** multimodal, with peaks at common Ford displacements
  (1.0L, 1.5L, 2.0L)
- **mpg:** roughly unimodal around 55–65 mpg, with a thin tail past 150
  mpg likely from Hybrid/Electric models

---

## Q5 — Count Plots of Categorical Features

Created count plots for `fuelType`, `transmission`, and `model` using
seaborn.

📊 *See: `Q5_countplots.png`*

**Most common categories:**

| Column | Top category | Count |
|---|---|---|
| fuelType | Petrol | 12,081 |
| transmission | Manual | 15,383 |
| model | Fiesta | 6,509 |

**Market trend insights:**
- Petrol dominates over Diesel/Hybrid/Electric, suggesting the data
  predates widespread EV/hybrid adoption
- Manual outnumbering Automatic reflects a European/UK-style used-car
  market
- Fiesta and Focus dominate the model counts, consistent with their
  reputation as Ford best-selling, mass-market models

---

## Q6 — Correlation Heatmap

Computed the correlation matrix for all numeric features and visualized
it as an annotated seaborn heatmap.

📊 *See: `Q6_heatmap.png`*

**Correlation with price (sorted):**

| Feature | Correlation with price |
|---|---|
| year | +0.64 |
| engineSize | +0.41 |
| tax | +0.41 |
| mpg | −0.35 |
| mileage | −0.53 |

**Observations:**
- `year` is the strongest positive driver of price — newer cars cost more
- `mileage` is the strongest negative driver — higher mileage lowers
  resale value
- `year` and `mileage` are themselves strongly correlated (−0.71),
  flagging a multicollinearity risk for linear models

---

## Q7 — Feature Identification

**Independent Features (Inputs):** `model`, `year`, `transmission`,
`mileage`, `fuelType`, `tax`, `mpg`, `engineSize`

**Dependent Feature (Target):** `price`

**Justification:** `price` is the outcome we want to predict, while all
other columns describe properties of the car that are determined before
a price is set (e.g. mileage and year come from usage history, not from
price). The correlation analysis in Q6 supports treating these as causal
inputs to price.

---

## Q8 — Encoding Categorical Variables

Applied **One-Hot Encoding** via `pd.get_dummies()` on `model`,
`transmission`, and `fuelType`, using `drop_first=True` to avoid the
dummy variable trap.

**Before → After (example: `transmission`):**

| Before | → | transmission_Manual | transmission_Semi-Auto |
|---|---|---|---|
| Automatic | | False | False |
| Manual | | True | False |

**Shape change:** `(17812, 9)` → `(17812, 34)`

---

## Q9 — Feature Scaling

Applied `StandardScaler` from scikit-learn to the numeric independent
features: `year`, `mileage`, `tax`, `mpg`, `engineSize`.

**First 5 rows of scaled data:**

| year | mileage | tax | mpg | engineSize |
|---|---|---|---|---|
| 0.067 | −0.383 | 0.591 | −0.021 | −0.811 |
| 0.554 | −0.736 | 0.591 | −0.021 | −0.811 |
| 0.067 | −0.563 | 0.591 | −0.021 | −0.811 |
| 1.042 | −0.665 | 0.511 | −1.738 | 0.345 |
| 1.042 | −1.128 | 0.511 | −0.909 | −0.811 |

Scaling puts every numeric feature on a mean-0, standard-deviation-1
scale, which is essential for distance-based or gradient-based models.

---

## Q10 — Complete Preprocessing Pipeline

Combined every prior step into a single end-to-end pipeline:

1. **Load & clean** — read CSV, drop 154 duplicate rows, confirm zero
   missing values
2. **EDA** — histograms, count plots, and correlation heatmap
3. **Feature identification** — `price` as target, remaining 8 columns
   as inputs
4. **Encoding** — one-hot encode `model`, `transmission`, `fuelType`
5. **Scaling** — standard-scale the numeric input features

**Final output:**
- `X` (features): **(17,812, 33)**
- `y` (target): **(17,812,)**

The resulting `X`/`y` pair is fully cleaned, encoded, and scaled — ready
to be fed directly into a machine learning model.

---

## Q11 — Preparing Features (X and Y)

Loaded the cleaned Ford Car dataset and separated it into independent
features and the dependent target variable.

- `Y = df['price']` — target column (what we want to predict)
- `X = df.drop('price', axis=1)` — all remaining columns as inputs

**Shape of X:** `(17812, 8)` | **Shape of Y:** `(17812,)`

---

## Q12 — One-Hot Encoding

Identified all categorical columns in X using `select_dtypes(include='object')`.
Applied **One-Hot Encoding** with `pd.get_dummies()` and converted the
result to integer type using `.astype(int)`.

**Categorical columns encoded:** `model`, `transmission`, `fuelType`

**Shape after encoding:** `(17812, 36)`

First 5 rows after encoding show binary (0/1) dummy columns for each
category, replacing the original string columns.

---

## Q13 — Feature Scaling (StandardScaler)

Applied `StandardScaler` on the numerical columns: `year`, `mileage`,
`tax`, `mpg`, `engineSize`.

Each column is transformed to have **mean = 0** and **standard deviation = 1**,
ensuring no single feature dominates due to scale differences.

---

## Q14 — Train-Test Split

Split the preprocessed data into training and testing sets using
`train_test_split` with `test_size=0.33` and `random_state=42`.

| Set | X shape | y shape |
|---|---|---|
| Training | (11934, 36) | (11934,) |
| Testing | (5878, 36) | (5878,) |

67% data used for training, 33% held out for testing.

---

## Q15 — Building Linear Regression Model

Trained a `LinearRegression` model from `sklearn.linear_model` on
`X_train` and `y_train`.

- **Intercept:** `17251.61`
- Coefficients reflect the weight assigned to each feature in predicting
  car price — positive for features like `year` and `engineSize`,
  negative for `mileage` and `mpg`

---

## Q16 — Making Predictions

Used the trained model to make predictions on `X_test` and stored
the results in `y_pred`.

**First 10 Predicted vs Actual values:**

| # | Predicted (y_pred) | Actual (y_test) |
|---|---|---|
| 1 | 6,152.58 | 6,995 |
| 2 | 9,374.38 | 8,999 |
| 3 | 9,464.68 | 7,998 |
| 4 | 4,597.92 | 5,491 |
| 5 | 3,496.32 | 3,790 |
| 6 | 15,303.51 | 18,200 |
| 7 | 20,374.82 | 22,998 |
| 8 | 12,631.56 | 11,000 |
| 9 | 12,297.70 | 7,600 |
| 10 | 17,230.22 | 14,985 |

Most predictions are reasonably close to actual values, with a few
deviations in rows where car specs are atypical.

---

## Q17 — Model Evaluation using R² Score

Calculated the R² score using `r2_score(y_test, y_pred)`.

**R² Score: 0.831**

**Interpretation:** The model explains **83.1% of the variance** in car
prices using the available features. This means that year, mileage,
engine size, fuel type, and transmission together account for the
majority of price variation in the dataset.

**Performance comment:** An R² of 0.83 indicates **good model
performance** for a Linear Regression baseline. The remaining 16.9%
unexplained variance is likely due to factors not in the dataset such
as car condition, accident history, and regional pricing. A tree-based
model like Random Forest could push this further toward 0.90+.

---

## Q18 — Saving the Model

Saved all trained objects using `joblib.dump()` for future reuse without
retraining.

| File | Contents |
|---|---|
| `LR_ford_car.pkl` | Trained Linear Regression model |
| `scaler.pkl` | Fitted StandardScaler object |
| `columns.pkl` | List of feature column names (36 columns) |

All three files are needed together at prediction time — the scaler must
transform new input data the same way as training data, and the columns
list ensures the feature order matches what the model expects.

---

## Project Files

| File | Description |
|---|---|
| `ford_car.csv` | Raw source dataset |
| `ford_assignment.py` | Full annotated script covering Q1–Q10 (EDA & Preprocessing) |
| `ford_ml.py` | ML pipeline script covering Q11–Q18 (Modeling & Evaluation) |
| `pipeline.py` | Standalone, minimal end-to-end pipeline script |
| `LR_ford_car.pkl` | Saved Linear Regression model |
| `scaler.pkl` | Saved StandardScaler object |
| `columns.pkl` | Saved list of feature columns |
| `Q4_histograms.png` | Histograms of numeric features |
| `Q5_countplots.png` | Count plots of categorical features |
| `Q6_heatmap.png` | Correlation heatmap |

---

## Key Takeaways

- The dataset was clean aside from 154 duplicate rows and one clearly
  erroneous `year` value (2060)
- `year` and `mileage` are the dominant predictors of `price`, with
  `engineSize`, `tax`, and `mpg` contributing secondary signal
- The categorical mix (Petrol, Manual, Fiesta/Focus) reflects a
  mainstream used-car market rather than a premium or electrified one
- Linear Regression achieved an **R² of 0.831** — a strong baseline
  that explains 83% of price variance
- All preprocessing objects (scaler, columns) are saved alongside the
  model to ensure consistent predictions on new data

---

## Functions Used

| Function | Purpose |
|---|---|
| `pd.read_csv()` | Load dataset |
| `df.drop_duplicates()` | Remove duplicate rows |
| `df.isnull().sum()` | Check missing values |
| `df.describe()` | Statistical summary |
| `df.select_dtypes()` | Identify column types |
| `pd.get_dummies()` | One-Hot Encode categorical columns |
| `StandardScaler()` | Scale numeric features to mean-0, std-1 |
| `train_test_split()` | Split data into train and test sets |
| `LinearRegression()` | Build regression model |
| `model.fit()` | Train the model |
| `model.predict()` | Make predictions |
| `r2_score()` | Evaluate model performance |
| `joblib.dump()` | Save model and preprocessing objects |