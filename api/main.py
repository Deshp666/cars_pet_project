import os
import sys
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_model import custom_transformers
sys.modules['custom_transformers'] = custom_transformers

# Загрузка модели и трансформера из локальных файлов
MODEL_PATH = os.path.join(project_root, "artifacts", "model.pkl")
TRANSFORMER_PATH = os.path.join(project_root, "artifacts", "power_trans.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(TRANSFORMER_PATH, "rb") as f:
    power_trans = pickle.load(f)

from ml_model import custom_transformers
sys.modules['custom_transformers'] = custom_transformers

# === FastAPI ===
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
    seats: float


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
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}