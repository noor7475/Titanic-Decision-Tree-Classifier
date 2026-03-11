# Titanic Survival Prediction – Decision Tree Classifier

## Overview

This project builds a **Decision Tree classifier** to predict whether a passenger survived the Titanic disaster using passenger attributes such as age, fare, passenger class, and family relationships.

The notebook demonstrates a complete **machine learning workflow**, including:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Handling missing values
* Addressing class imbalance
* Hyperparameter tuning and tree pruning
* Model evaluation

---

# Dataset

The Dataset contains information about passengers aboard the Titanic.

Important features used in the model include:

* `Pclass` – Passenger class
* `Age` – Passenger age
* `Fare` – Ticket fare
* `SibSp` – Number of siblings/spouses aboard
* `Parch` – Number of parents/children aboard
* `Embarked` – Port of embarkation

Target variable:

* **Survived**

  * `0` → Did not survive
  * `1` → Survived

### Removed Features

Some features were removed during preprocessing:

* **Cabin** – contained a very large number of missing values.
* **Sex** – removed intentionally because it perfectly predicts survival in the dataset (100% separation).
Including it would dominate the model and prevent learning meaningful patterns from other features such as age, fare, or passenger class.

---

# Project Pipeline

## 1. Data Cleaning

* Checked for missing values
* Removed features with excessive missing data (`Cabin`)
* Removed non-informative columns
* Removed the `Sex` feature to avoid a dominant predictor

---

## 2. Exploratory Data Analysis (EDA)

EDA was performed to understand feature distributions and relationships with survival:

* **Histograms** to inspect numerical feature distributions
* **Boxplots** to detect outliers
* **Correlation heatmap** for numerical relationships
* Survival analysis across passenger features

---

## 3. Feature Engineering

A new feature called **FamilyGroup** was created using:

```
FamilySize = SibSp + Parch
```

Passengers were grouped based on family size to capture patterns in survival among individuals traveling alone versus with family.

---

## 4. Handling Missing Values

Missing values were handled as follows:

* **Age** → filled using the **median grouped by passenger class**
* **Fare** → filled using the **median**

Median imputation was chosen because it is **robust to skewed distributions and outliers**.

---

## 5. Handling Class Imbalance

The dataset contains **more non-survivors than survivors**, creating class imbalance.

Two techniques were used:

* **Undersampling** to balance the training data
* Initializing the model with
  `class_weight='balanced'`
  to further account for imbalance during training.

---

## 6. Model Training

A **Decision Tree Classifier** from Scikit-learn was used.

To improve the model and prevent overfitting:

* **GridSearchCV** was used for **hyperparameter tuning**

* Tree **pruning parameters** were optimized, including:

* `max_depth`

* `min_samples_split`

* `min_samples_leaf`

* `criterion`

These parameters control **tree complexity**, helping the model generalize better to unseen data.

---

## Model Results

The tuned Decision Tree was evaluated on the **test dataset (126 samples)**.

### Classification Report

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Not Survived (0) | 0.78      | 0.64   | 0.71     |
| Survived (1)     | 0.57      | 0.72   | 0.64     |

### Overall Metrics

* **Accuracy:** 0.67
* **Macro Avg F1-Score:** 0.67
* **Weighted Avg F1-Score:** 0.68

### Confusion Matrix

```
[[49 27]
 [14 36]]
```

### Result Interpretation

The overall accuracy (~67%) is lower than some Titanic models because the **`Sex` feature was removed**. Including it would cause **data leakage**, as it **perfectly determines survival**.

By excluding it, the model learns patterns from other meaningful features like passenger class, age, fare, and family relationships. This ensures the model reflects **true influences on survival** rather than relying on a single perfectly predictive variable.

---

## Repository Structure

```
Titanic-Decision-Tree-Classifier
│
├── Titanic_Decision_Tree_Classifier.ipynb
├── Titanic_Data.csv
└── README.md
```

---

## The Colab Link

The **Colab link is available inside the notebook** (`Titanic_Decision_Tree_Classifier.ipynb`).

---
