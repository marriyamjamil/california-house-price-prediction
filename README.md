# California House Price Prediction

An end-to-end machine learning project that predicts median house values for districts in California. The project covers the full workflow, from data preparation and model training to an interactive web application built with Streamlit.

## Overview

Accurately estimating property values is a core problem in real estate. This project builds a regression model that predicts the median house value of a district from census features such as location, housing age, population and median income.

The workflow includes:

- **Stratified sampling** on income category, so the test set reflects the real income distribution
- **A reusable preprocessing pipeline** that handles missing values, scaling and categorical encoding
- **A Random Forest regressor**, evaluated on a held-out test set
- **A Streamlit web app** that loads the trained model and returns predictions from user input

## Dataset

The project uses the **California Housing** dataset, derived from the 1990 California census. Each record represents one district.

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic location of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Number of residents |
| `households` | Number of households |
| `median_income` | Median household income (in tens of thousands of USD) |
| `ocean_proximity` | Location relative to the ocean (`<1H OCEAN`, `INLAND`, `NEAR BAY`, `NEAR OCEAN`, `ISLAND`) |
| `median_house_value` | **Target variable:** median house value in USD |

## Methodology

### 1. Train/test split
The data is split 80/20 using `StratifiedShuffleSplit` on an income category feature. Median income is a strong predictor of house value, so stratifying on it keeps both sets representative.

### 2. Preprocessing

| Feature type | Transformation |
|---|---|
| Numerical | Missing values filled with the median (`SimpleImputer`), then standardized (`StandardScaler`) |
| Categorical | One-hot encoded (`OneHotEncoder`), with unknown categories ignored at prediction time |

Both steps are combined in a single `ColumnTransformer`, so exactly the same transformations are applied during training and prediction.

### 3. Model
A `RandomForestRegressor` is trained on the processed features. Random Forest handles non-linear relationships well and is robust to outliers, which makes it a strong baseline for tabular data.

## Results

The model is evaluated on the held-out test set, which it never sees during training.

| Model | Test RMSE |
|---|---|
| Random Forest Regressor | $47,198 |

On average, predictions are within about $47,000 of the actual median house value. Values in the dataset reach about $500,000.

## Getting started

### Prerequisites
- Python 3.10 or later

### Installation

```bash
git clone https://github.com/marriyamjamil/california-house-price-prediction.git
cd california-house-price-prediction
pip install -r requirements.txt
```

### Train the model

```bash
python main.py
```

This trains the model, prints the test RMSE, and saves the model and preprocessing pipeline to the `models/` directory.

### Run the web app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, enter the district details and click **Predict price**.

## Tech stack

| Category | Tools |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Web app | Streamlit |

## Future improvements

- Compare additional models such as Gradient Boosting and XGBoost
- Tune hyperparameters with cross-validation (`GridSearchCV` / `RandomizedSearchCV`)
- Engineer features such as rooms per household and population per household
- Deploy the app to Streamlit Community Cloud

## Notes

Trained model files are not included in this repository because of their size. Run `python main.py` to generate them locally.
