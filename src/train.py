import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src.config import CAT_ATTRIBS, MODEL_FILE, MODELS_DIR, PIPELINE_FILE, RANDOM_STATE
from src.data import load_data, split_data, split_features_target
from src.pipeline import build_pipeline


def train():
    """Train the model, save it, and return the RMSE on the test set."""
    train_set, test_set = split_data(load_data())
    X_train, y_train = split_features_target(train_set)
    X_test, y_test = split_features_target(test_set)

    num_attribs = X_train.drop(CAT_ATTRIBS, axis=1).columns.tolist()
    pipeline = build_pipeline(num_attribs, CAT_ATTRIBS)

    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(pipeline.fit_transform(X_train), y_train)

    rmse = root_mean_squared_error(y_test, model.predict(pipeline.transform(X_test)))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    return rmse
