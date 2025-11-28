FROM python:3.10-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY ml_model/ ./ml_model/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

RUN python -c "from ml_model import custom_transformers; print('✅ Custom transformer OK')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]