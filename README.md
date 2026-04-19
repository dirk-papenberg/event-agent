# event-agent

Runs a FastAPI service on port `8000` and exposes `POST /events`.

## Local run

```bash
uv sync
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Docker deploy

The GitHub Actions workflow in `.github/workflows/deploy.yml` deploys this service
onto a self-hosted runner labeled `local`. It creates or reuses the shared
`recipes-network` Docker network and starts the container through
`docker compose up --build -d`.

Required GitHub secrets:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `TAVILY_API_KEY`
