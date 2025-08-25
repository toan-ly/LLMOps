# LLM Backend Service

A FastAPI application that provides structured API endpoints for interacting with LLM models through LangChain

## Features
-  **Sentiment Analysis**: Analyze text sentiment using VSF-LORA model
- **Medical QA**: Answer medical multiple choice questions using MEDQA-LORA model
- **Instrumentation**: Prometheus metrics for monitoring
- **Tracing**: LangSmith integration for LLM observability

## API Endpoints
- `GET /`
- `GET /health`: Health check endpoint
- `POST /v1/sentiment`: Analyze sentiment of text input
- `POST /v1/medical-qa`: Answer medical multiple choice question

## Usage
#### Use Docker Compose:
```bash
docker compose up -d
```
The API will be available at `http://localhost:8001`

