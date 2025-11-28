import argparse
import os
import socket
import pandas as pd
import numpy as np
import pickle
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from ml_model.data_prep import get_features_and_target, get_preprocessor
from ml_model.custom_transformers import FeatureEngineerAndCleaner
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_and_prepare_data(data_path):
    print('load_and_prepare_data begin')
    df = pd.read_csv(data_path)
    print('file loaded')

    df = df.dropna(subset=['selling_price', 'year'])
    X, Y_scale, power_trans = get_features_and_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y_scale, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val, power_trans


def train_model(X_train, y_train):
    estimators = [
        ('dt', DecisionTreeRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('svr', SVR()),
        ('knn', KNeighborsRegressor()),
        ('lasso', Lasso(random_state=42))
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression()
    )

    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineerAndCleaner(use_statistic_method=True)),
        ('preprocessor_scaler', get_preprocessor()),
        ('regressor', stacking_regressor)
    ])

    params = {
        'regressor__dt__max_depth': [5],
        #'regressor__rf__n_estimators': [50, 100],
    }

    clf = GridSearchCV(full_pipeline, params, cv=3, n_jobs=-1, verbose=1)
    print("Начало обучения и подбора гиперпараметров...")
    clf.fit(X_train, y_train.ravel())  # .ravel() вместо reshape

    best_model = clf.best_estimator_
    print("Обучение завершено. Лучшие параметры:", clf.best_params_)
    return best_model, clf.best_params_, clf.best_score_


# def save_artifacts(best_model, power_trans, model_path, transformer_path):
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     with open(model_path, 'wb') as f:
#         pickle.dump(best_model, f)
#     with open(transformer_path, 'wb') as f:
#         pickle.dump(power_trans, f)
#     print(f"Модель сохранена: {model_path}")
#     print(f"Трансформер сохранён: {transformer_path}")


def evaluate_model(best_model, X_val, y_val, power_trans):
    y_pred = best_model.predict(X_val)
    y_val_real = power_trans.inverse_transform(y_val.reshape(-1, 1))
    y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1))
    rmse, mae, r2 = eval_metrics(y_val_real, y_pred_real)

    print("-" * 30)
    print(f"Metrics on Validation Set:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--skip-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--mlflow-experiment", type=str, default="Car_Price_Prediction_DVC")

    args = parser.parse_args()
    if os.getenv("GITHUB_ACTIONS"):
        mlflow_uri = "http://host.docker.internal:5000"
    else:
        mlflow_uri = "http://127.0.0.1:5000"

    if not args.skip_mlflow:
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)

        except Exception as e:
            print(f"⚠️ MLflow недоступен: {e}. Пропускаем логирование.")
            args.skip_mlflow = True

    X_train, X_val, y_train, y_val, power_trans = load_and_prepare_data(args.data_path)
    best_model, best_params, best_score = train_model(X_train, y_train)
    evaluate_model(best_model, X_val, y_val, power_trans)

    y_pred = best_model.predict(X_val)
    y_val_real = power_trans.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    rmse, mae, r2 = eval_metrics(y_val_real, y_pred_real)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("artifacts/power_trans.pkl", "wb") as f:
        pickle.dump(power_trans, f)

    if not args.skip_mlflow:
        with mlflow.start_run() as run:
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2, "best_cv_score": best_score})
            mlflow.log_params(best_params)
            mlflow.sklearn.log_model(best_model, "model")

            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                registered_model_name="CarPriceRegressor"
            )

    print(f"✅ Модель зарегистрирована в MLflow как 'CarPriceRegressor'")
    print(f"✅ Run ID: {run.info.run_id}")


if __name__ == '__main__':
    main()