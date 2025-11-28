FROM python:3.10-slim

RUN pip install mlflow==2.11.3

RUN mkdir -p /mlflow && chmod -R 777 /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts"]
