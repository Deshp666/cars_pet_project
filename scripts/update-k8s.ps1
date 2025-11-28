Write-Host "Pulling latest image..."
minikube image pull ghcr.io/deshp666/car-price-api:latest

Write-Host "Updating deployment..."
kubectl set image deployment/car-price-api api=ghcr.io/deshp666/car-price-api:latest

Write-Host "Waiting for rollout..."
kubectl rollout status deployment/car-price-api

Write-Host "✅ Deployment updated successfully!"
