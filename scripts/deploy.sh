#!/bin/bash

REPO_URI="692828329557.dkr.ecr.us-east-1.amazonaws.com/frontend"

echo "Logging into ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REPO_URI


echo "Pulling latest image..."
docker pull $REPO_URI:latest

echo "Stopping old container..."
docker stop frontend || true

echo "Removing old container..."
docker rm frontend || true

echo "Starting new container..."
docker run -d -p 80:5000 --restart always --name frontend $REPO_URI:latest

echo "Deployment complete."
