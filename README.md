# Ford Car Dataset — EDA & Preprocessing Pipeline

## Overview

This project performs a complete exploratory data analysis (EDA) and
preprocessing pipeline on the **Ford Car Dataset**, a used-car listings
dataset containing **17,966 records** and **9 columns**. The goal is to
clean the data, understand its structure and distributions, identify
relationships between features, and prepare it for downstream machine
learning modeling (predicting car resale price).

**Dataset columns:** `model`, `year`, `price`, `transmission`, `mileage`,
`fuelType`, `tax`, `mpg`, `engineSize`

**Tools used:** `pandas`, `numpy`, `matplotlib`, `seaborn`,
`scikit-learn` (`StandardScaler`)

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
11. [Project Files](#project-files)

---

## Q1 — Data Loading & Initial Analysis

Loaded the dataset with `pandas.read_csv()` and inspected its structure
using `head(10)`, `tail(5)`, `.shape`, and `.dtypes`.

**Findings:**
- Shape: **17,966 rows × 9 columns**
- Data types: 4 integer columns (`year`, `price`, `mileage`, `tax`), 2
  float columns (`mpg`, `engineSize`), 3 categorical/string columns
  (`model`, `transmission`, `fuelType`)
- No ID column is present — pandas' default integer index serves as the
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
  reputation as Ford's best-selling, mass-market models
- Overall, the data reflects a mainstream, budget-to-mid-range used-car
  market rather than a premium or electrified one

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

## Project Files

| File | Description |
|---|---|
| `ford_car_dataset.csv` | Raw source dataset |
| `ford_assignment.py` | Full annotated script covering Q1–Q10 |
| `pipeline.py` | Standalone, minimal-style end-to-end pipeline script |
| `Q4_histograms.png` / `pipeline_histograms.png` | Histograms of numeric features |
| `Q5_countplots.png` / `pipeline_countplots.png` | Count plots of categorical features |
| `Q6_heatmap.png` / `pipeline_heatmap.png` | Correlation heatmap |
| `run_output.txt` | Captured console output from the full run |

---

---

## Key Takeaways

- The dataset was clean aside from 154 duplicate rows and one clearly
  erroneous `year` value (2060)
- `year` and `mileage` are the dominant predictors of `price`, with
  `engineSize`, `tax`, and `mpg` contributing secondary signal
- The categorical mix (Petrol, Manual, Fiesta/Focus) reflects a
  mainstream used-car market rather than a premium or electrified one
- The final preprocessed dataset (`X`, `y`) is ready for regression
  modeling to predict used Ford car prices