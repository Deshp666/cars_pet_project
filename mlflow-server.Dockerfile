FROM python:3.10-slim

RUN pip install mlflow==2.11.3

RUN mkdir -p /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow", \
     "--serve-artifacts"]