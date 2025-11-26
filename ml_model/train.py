import pandas as pd
import numpy as np
import pickle
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature
from ml_model.data_prep import get_features_and_target, get_preprocessor
from ml_model.custom_transformers import FeatureEngineerAndCleaner
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Car_Price_Prediction_Pipeline")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
TRANSFORMER_PATH = os.path.join(ARTIFACTS_DIR, 'power_trans.pkl')

def eval_metrics(actual, pred):
    """
    Рассчитывает метрики регрессии.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_and_prepare_data():
    """
    Загружает сырые данные, выполняет масштабирование Y и разбивает на выборки.
    Возвращает данные для обучения и трансформер Y.
    """
    print('load_and_prepare_data begin')
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/Deshp666/cars_dataset/refs/heads/main/cars.csv', delimiter = ',')
        print('file dowloaded')
    except FileNotFoundError:
        print("Ошибка: Файл не найден.")
        raise

    df = df.dropna(subset=['selling_price', 'year'])

    X, Y_scale, power_trans = get_features_and_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y_scale, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, power_trans


def train_model(X_train, y_train):
    """
    Создает полный ML Pipeline, выполняет GridSearchCV и обучает модель.
    Возвращает лучший обученный Pipeline.
    """

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
        ('preprocessor_scaler', get_preprocessor()),  # ColumnTransformer со StandardScaler
        ('regressor', stacking_regressor)
    ])

    params = {
        'regressor__dt__max_depth': [5, 10],
        'regressor__rf__n_estimators': [50, 100],
    }

    clf = GridSearchCV(full_pipeline, params, cv=3, n_jobs=-1, verbose=1)

    print("Начало обучения и подбора гиперпараметров...")
    clf.fit(X_train, y_train.reshape(-1))

    best_model = clf.best_estimator_
    print("Обучение завершено. Лучшие параметры:", clf.best_params_)

    return best_model, clf.best_params_, clf.best_score_


def evaluate_and_save_artifacts(best_model, X_val, y_val, power_trans, best_params, best_score):
    """
    Оценивает модель, логирует метрики в MLflow и сохраняет .pkl файлы.
    """
    y_pred = best_model.predict(X_val)

    y_val_real = power_trans.inverse_transform(y_val)
    y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1))

    (rmse, mae, r2) = eval_metrics(y_val_real, y_pred_real)

    print("-" * 30)
    print(f"Metrics on Validation Set:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    print("-" * 30)

    with mlflow.start_run(run_name=best_model.steps[-1][0]):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("best_cv_score", best_score)

        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        predictions = best_model.predict(X_val)
        signature = infer_signature(X_val, predictions)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Модель ({MODEL_PATH}) сохранена.")

    with open(TRANSFORMER_PATH, 'wb') as file:
        pickle.dump(power_trans, file)
    print(f"Трансформер ({TRANSFORMER_PATH}) сохранен.")


if __name__ == '__main__':
    print('init...')
    X_train, X_val, y_train, y_val, power_trans = load_and_prepare_data()
    print('load_and_prepare_data is done')

    best_model, best_params, best_score = train_model(X_train, y_train)
    print('train_model is done')

    evaluate_and_save_artifacts(best_model, X_val, y_val, power_trans, best_params, best_score)
    print('evaluate_and_save_artifacts is done')