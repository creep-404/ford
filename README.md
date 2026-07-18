# Ford Car Price Predictor — Full Project README

## Overview

This project covers the complete lifecycle of a Ford used-car price
prediction system: from raw data exploration and cleaning, through
model training (two separate assignments, two different pipelines),
to building working Streamlit web app frontends for each model.

**Dataset:** `ford_car.csv` — 17,966 rows, 9 columns
(`model`, `year`, `price`, `transmission`, `mileage`, `fuelType`,
`tax`, `mpg`, `engineSize`)

---

## Table of Contents

1. [Part 1 — EDA & Preprocessing Pipeline](#part-1--eda--preprocessing-pipeline)
2. [Part 2 — Model Training (Assignment 20)](#part-2--model-training-assignment-20)
3. [Part 3 — Session 21: Building the Streamlit App](#part-3--session-21-building-the-streamlit-app)
4. [Part 4 — The LabelEncoder Discovery](#part-4--the-labelencoder-discovery)
5. [Part 5 — Building `frontend.py` for the Assignment 20 Model](#part-5--building-frontendpy-for-the-assignment-20-model)
6. [Part 6 — Debugging & Environment Issues](#part-6--debugging--environment-issues)
7. [Current Project Files](#current-project-files)
8. [Where Things Stand](#where-things-stand)

---

## Part 1 — EDA & Preprocessing Pipeline

The dataset was first explored and cleaned end-to-end (Q1–Q10 of an
earlier pipeline):

- **Q1–Q3:** Loaded the data, checked shape/dtypes, found 154 duplicate
  rows (removed, leaving 17,812 rows), confirmed zero missing values,
  and flagged a bad data point — a `year` value of **2060**.
- **Q4–Q6:** Plotted histograms, count plots, and a correlation heatmap.
  Found `price` and `mileage` right-skewed; `year` (+0.64) and
  `mileage` (−0.53) were the strongest price drivers; Petrol/Manual/
  Fiesta were the dominant categories.
- **Q7–Q9:** Identified `price` as the target and the other 8 columns
  as features; one-hot encoded the 3 categorical columns
  (`model`, `transmission`, `fuelType`) with `drop_first=True`; scaled
  the 5 numeric columns with `StandardScaler`.
- **Q10:** Combined everything into one pipeline, producing a final
  `X` of shape `(17812, 33)` and `y` of shape `(17812,)`.

This pipeline was documentation/exploration only — it wasn't the
version whose `.pkl` files ended up being used later.

---

## Part 2 — Model Training (Assignment 20)

A second, separate notebook (**Session 20 Assignment**, by Shahid Ahmad
Khan) trained the actual Linear Regression model that got saved and
reused for the Streamlit frontend work:

- **Q1:** Loaded `ford_car.csv`, split into `X` (8 features) and
  `Y = price`. Shape: `X (17966, 8)`, `Y (17966,)`.
- **Q2–Q3 (exploratory only):** Showed one-hot encoding via
  `pd.get_dummies()` producing 36 columns, then StandardScaler applied
  on top of that — **but this encoding approach was not what got used
  in the final saved model** (see Part 4).
- **Q4:** `train_test_split(test_size=0.33, random_state=42)`.
- **Q5:** Trained `LinearRegression()`. Intercept ≈ `17210.92`.
- **Q6–Q7:** Predicted on the test set; **R² score = 0.7366**.
- **Q8:** Saved the final artifacts with `joblib.dump()`:
  - `LR_ford_car.pkl` — the trained model
  - `scaler.pkl` — the fitted `StandardScaler`
  - `columns.pkl` — the list of feature column names

Critically, the **Q8 code path** switched from the one-hot approach
shown earlier in the notebook to using **`LabelEncoder`** instead —
looping over the 3 categorical columns and reassigning a single shared
`le` object each time. This is the pipeline that actually produced the
saved `.pkl` files.

---

## Part 3 — Session 21: Building the Streamlit App

Working through the **Session 21 (AIML) Assignment Questions** PDF,
`app.py` was built up incrementally, one question at a time:

| Q | What was added |
|---|---|
| Q1 | Imports: `streamlit`, `pandas`, `joblib`, with comments on each |
| Q2 | `joblib.load()` calls for `model`, `scaler`, `encoded_columns` |
| Q3 | `st.set_page_config(page_title="Ford Car Price Predictor", layout="centered")` |
| Q4 | `st.title()` + `st.write()` description |
| Q5 | `st.number_input()` fields for year, mileage, tax, mpg, engine size — ranges pulled from the real min/max/median of `ford_car.csv` |
| Q6 | `st.selectbox()` dropdowns for transmission and fuel type (verified actual category values in the CSV, including `Electric` and `Other` fuel types not mentioned in the assignment text) |
| Q7 | `st.text_input()` for car model name + `st.button("Predict Price")` |
| Q8 | Inside `if predict_clicked:` — built the input `DataFrame`, one-hot encoded with `pd.get_dummies()`, aligned to `encoded_columns` via `reindex(fill_value=0)` |
| Q9 | Scaled the 5 numeric columns with the loaded scaler, ran `model.predict()`, displayed the result with `st.success()` formatted as `£X,XXX.XX` |
| Q10 | Combined everything into one clean `app.py`, with error handling (`try/except` around file loading and prediction) and a nicer two-column layout for the number inputs |

This version assumes **one-hot encoded** training columns (matching
the Part 1 pipeline), which turned out to be a mismatch with the
actual saved `.pkl` files from Part 2.

---

## Part 4 — The LabelEncoder Discovery

When asked to build a frontend for the **Assignment 20** model
specifically (the PDF's optional task), a closer read of that PDF's
final Q8 code revealed the mismatch:

- `columns.pkl` from Assignment 20 holds just **8 plain column names**
  (`model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`,
  `mpg`, `engineSize`) — not 36 one-hot dummy columns.
- The categorical columns were converted to integers via
  `LabelEncoder`, not `pd.get_dummies()`.
- The training code never saved the fitted `LabelEncoder` itself —
  only the model, scaler, and column list — so the exact category→
  number mapping used during training wasn't stored anywhere.

Since scikit-learn's `LabelEncoder` assigns codes in **alphabetical
order** of the unique values by default, the original mapping was
reconstructed directly from `ford_car.csv`:

```python
model     -> B-MAX:0, C-MAX:1, EcoSport:2, ... Transit Tourneo:22  (23 values)
transmission -> Automatic:0, Manual:1, Semi-Auto:2
fuelType  -> Diesel:0, Electric:1, Hybrid:2, Other:3, Petrol:4
```

This confirmed the Session 21 `app.py` (one-hot based) and the
Assignment 20 model (label-encoded) are **not interchangeable** — each
needed its own frontend logic.

---

## Part 5 — Building `frontend.py` for the Assignment 20 Model

A new file, `frontend.py`, was built specifically for the Assignment 20
model:

- Loads `LR_ford_car.pkl`, `scaler.pkl`, `columns.pkl`
- Dropdowns for `model`, `transmission`, `fuelType` built from the
  reconstructed label-encoding maps above
- On predict: converts each selected category to its label-encoded
  integer, builds a single-row `DataFrame`, reorders columns to match
  `columns.pkl`, scales the 5 numeric columns, and predicts
- Written in a plainer, more student-like comment style (lowercase,
  short comments) to match the original assignment code's voice,
  rather than heavily annotated "AI-style" documentation

---

## Part 6 — Debugging & Environment Issues

A few real issues came up while testing, all resolved along the way:

- **`IndentationError: unindent amount does not match previous
  indent`** — traced to inconsistent spacing (1 space vs. 4 spaces)
  in a manually retyped copy of the Q8 code block, plus other bugs in
  that copy: scalar values passed to `pd.DataFrame()` without being
  wrapped in lists, `pd.get_dummies(..., columns=encoded_columns)`
  used incorrectly, and `model.columns` referenced on the trained
  model object (which doesn't have a `.columns` attribute).
- **Duplicated code blocks** — because the notebook was built up cell
  by cell with each new cell re-pasting everything before it, the full
  code dump included the same imports/loading/config code repeated
  6 times. Cleaned down to a single, non-duplicated script.
- **`streamlit : The term 'streamlit' is not recognized...`** in
  PowerShell — Streamlit wasn't installed or wasn't on PATH. Fix:
  `pip install streamlit`, or run via `python -m streamlit run app.py`
  if PATH still doesn't pick it up.

---

## Current Project Files

| File | Purpose |
|---|---|
| `ford_car.csv` | Raw source dataset |
| `app.py` | Streamlit frontend for the **one-hot encoded** model (Session 21, Q1–Q10) |
| `frontend.py` | Streamlit frontend for the **label-encoded** model (Assignment 20) |
| `LR_ford_car.pkl` | Trained Linear Regression model (Assignment 20 — label-encoded) |
| `scaler.pkl` | Fitted `StandardScaler` (Assignment 20) |
| `columns.pkl` | List of 8 feature column names (Assignment 20, label-encoded) |

> **Note:** `app.py` (one-hot based) will only work correctly against
> a model/scaler/columns set trained with `pd.get_dummies()` — it does
> **not** match the `.pkl` files produced by Assignment 20. `frontend.py`
> is the version that correctly matches those files.

---

## Where Things Stand

- Both frontends (`app.py` and `frontend.py`) are written and
  syntax-checked.
- The Assignment 20 `.pkl` files (`LR_ford_car.pkl`, `scaler.pkl`,
  `columns.pkl`) still need to be confirmed as present in the local
  project folder before `frontend.py` can be run end-to-end.
- Streamlit installation was flagged as missing on the local machine —
  needs `pip install streamlit` before `streamlit run frontend.py` (or
  `app.py`) will work.
- Once running, the plan (Q10 of Session 21) is to take screenshots of
  the working app for submission.