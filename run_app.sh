#!/bin/bash
set -e

echo "🚀 Starting full pipeline: train → serve"

echo "📚 Training model..."
python -m ml_model.train

echo "🌐 Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000