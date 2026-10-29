# California House Price Prediction

A machine learning project that predicts median house values for California districts using a Random Forest regression model, with a Streamlit web app for interactive predictions.

## Overview

1. Stratified train/test split based on income category
2. Preprocessing with a scikit-learn pipeline: median imputation, feature scaling and one-hot encoding
3. Random Forest regression model, evaluated with RMSE on the held-out test set
4. Streamlit app that loads the saved model and predicts prices from user input

## Dataset

The project uses the **California Housing** dataset (`data/housing.csv`), based on the 1990 California census. Each row describes one district.

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Location of the district |
| `housing_median_age` | Median age of houses |
| `total_rooms`, `total_bedrooms` | Total rooms and bedrooms |
| `population`, `households` | Population and number of households |
| `median_income` | Median household income (in tens of thousands of USD) |
| `ocean_proximity` | Distance category to the ocean |
| `median_house_value` | Target: median house value (USD) |

## Project structure

```
.
├── data/
│   └── housing.csv         # Dataset
├── src/
│   ├── config.py           # Paths and settings
│   ├── data.py             # Data loading and stratified split
│   ├── pipeline.py         # Preprocessing pipeline
│   ├── train.py            # Model training, evaluation and saving
│   └── predict.py          # Model loading and prediction
├── app.py                  # Streamlit web app
├── main.py                 # Entry point for training
├── requirements.txt
└── README.md
```

Trained files are saved to `models/`, which is created when you run `main.py`.

## Preprocessing pipeline

| Feature type | Steps |
|---|---|
| Numerical | `SimpleImputer(strategy="median")`, then `StandardScaler` |
| Categorical (`ocean_proximity`) | `OneHotEncoder(handle_unknown="ignore")` |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/marriyamjamil/california-house-price-prediction.git
cd california-house-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

This trains the model, prints the test RMSE, and saves `model.pkl` and `pipeline.pkl` to `models/`.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter the district details and click **Predict price**.

## Tech stack

- Python
- pandas, NumPy
- scikit-learn
- joblib
- Streamlit

## Notes

Trained model files are not included in the repository because of their size. Run `python main.py` to generate them.
