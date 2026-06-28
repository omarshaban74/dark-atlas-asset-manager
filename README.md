# DarkAtlas Asset Management API

LangChain-powered asset management module for the DarkAtlas Attack Surface Monitoring platform. Built for the Buguard AI Applications internship track.

---

## Setup

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `DATABASE_URL` | `postgresql://user:password@db:5432/darkatlas` |
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |
| `POSTGRES_USER` | PostgreSQL username |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `POSTGRES_DB` | Database name |

---

## Run with Docker (Recommended)

```bash
docker compose up --build
```

API: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

```bash
# Stop
docker compose down

# Stop and wipe database
docker compose down -v
```

---

## Run Locally

**Prerequisites:** Python 3.11+, PostgreSQL running locally.

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create the database (SQLAlchemy creates tables automatically)
createdb darkatlas

# 4. Start the API
uvicorn main:app --reload
```

### Key Libraries

| Library | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | Server that runs FastAPI |
| `sqlalchemy` | ORM for PostgreSQL |
| `psycopg2-binary` | PostgreSQL driver |
| `pydantic` | Request/response validation |
| `langchain` + `langchain-groq` | LLM chain framework + Groq provider |
| `python-dotenv` | Loads `.env` variables |

---

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/assets/import` | Bulk import assets. Idempotent — re-importing updates `last_seen` and merges metadata, no duplicates. Malformed records are skipped gracefully. |
| `POST` | `/assets/enrich/batch` | Finds all unenriched assets and runs AI on them in one batch. Writes risk score, environment, criticality, summary, and enriched metadata back to DB. |
| `POST` | `/assets/enrich/{asset_id}` | Same as batch but for one specific asset by ID. |
| `GET` | `/assets/report` | Reads the full inventory and generates a CISO-ready risk report. LLM is grounded strictly in DB data — cannot invent assets. |
| `GET` | `/assets/search?q=` | Translate plain English into DB filters and return matching assets. Example: `?q=show me expired certificates on production subdomains` |

---

## Design Decisions

- **Structured LLM output** — all chains use `with_structured_output()` bound to Pydantic schemas. No free-text parsing.
- **Feedback loop** — on validation failure, the error is injected back into the prompt and retried up to 3 times before returning a 500.
- **Hallucination guard** — `/report` and `/search` are grounded in real DB data. The LLM never generates asset data directly.
- **Metadata merge** — on re-import, new metadata fields are added without overwriting existing ones.

## What I'd Do Next

- Auto-trigger enrichment on import
- Pagination on search results
- Split into `models.py`, `chains.py`, `routes.py`
- API key authentication on write operations
- Turn analysis layer into a LangGraph agent
