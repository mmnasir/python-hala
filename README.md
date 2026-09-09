# Chat Backend

Small FastAPI service that forwards chat messages from a frontend to an LLM.

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Set the API key

```bash
cp .env.example .env
```

Open `.env` and set:

```bash
LLM_API_KEY=your-api-key-here
```

## 3. Start FastAPI

```bash
uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## 4. Open Swagger

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

## 5. Test `/chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG"}'
```

Or use the `/chat` endpoint in Swagger.

## Tests

```bash
pytest
```
