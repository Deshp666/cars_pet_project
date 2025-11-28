docker run -d --name mlflow-temp -p 5000:5000 car-price-mlflow

Start-Sleep -Seconds 10

python -m ml_model.train --data-path data/cars.csv

docker stop mlflow-temp
docker rm mlflow-temp