import os
import sys
import pickle
import pandas as pd
import mlflow
from mlflow.artifacts import download_artifacts
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Подключаемся к MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Загружаем Production-модель
model = mlflow.pyfunc.load_model("models:/CarPriceRegressor@production")

# Загружаем трансформер (из последнего run)
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("Car_Price_Prediction_DVC")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)
latest_run_id = runs[0].info.run_id

# Скачиваем power_trans.pkl
artifact_path = download_artifacts(
    run_id=latest_run_id,
    artifact_path="transformers/power_trans.pkl"
)
with open(artifact_path, "rb") as f:
    power_trans = pickle.load(f)

# Добавляем корень в sys.path для кастомных трансформеров
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_model import custom_transformers
sys.modules['custom_transformers'] = custom_transformers

# FastAPI
app = FastAPI(title="Car Price Prediction")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


class CarInput(BaseModel):
    name: str
    year: float
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str      # "21.14 kmpl"
    engine: str       # "1248 CC"
    max_power: str    # "74 bhp"
    seats: float       # новое поле


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(car: CarInput):
    try:
        input_df = pd.DataFrame([car.dict()])
        pred_scaled = model.predict(input_df)
        pred_real = power_trans.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        return {"predicted_price": round(float(pred_real), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}