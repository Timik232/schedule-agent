# Schedule AI Agent

AI-powered agent system for querying university schedules using LangGraph, LangChain, and deterministic optimization tools.

## Architecture Overview

This system uses a hybrid approach:
- **LLM as Router/Planner**: Extracts entities, routes queries, and formats responses
- **Deterministic Tools**: SQL templates and OR-Tools for actual data retrieval and optimization
- **LangGraph**: Stateful orchestration with explicit control flow
- **Structured Output**: JSON schemas prevent hallucinations

## Features

- 📅 Natural language schedule queries
- 🎯 Entity extraction (groups, teachers, rooms, dates)
- 🔍 Optimized SQL queries with minimal context
- 🧮 Meeting slot optimization using CP-SAT solver
- 🛡️ Input validation and security guardrails
- 💬 Conversational context management
- 🚀 Docker deployment ready

## Project Structure

```
schedule-ai-agent/
├── src/
│   ├── graph/          # LangGraph state machine
│   ├── tools/          # SQL templates, text-to-SQL, optimizer
│   ├── models/         # Pydantic schemas and entities
│   ├── utils/          # Prompts, validators, guardrails
│   └── api/            # FastAPI server
└── tests/
```

## Setup

### Prerequisites
- Python 3.11-3.12
- PostgreSQL database with schedule schema
- OpenAI API key (or other LLM provider)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd schedule-ai-agent

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/schedule_db

# Security
MAX_SQL_ROWS=1000
QUERY_TIMEOUT_SECONDS=30
```

## Usage

### Start Development Server

```bash
poetry run uvicorn src.api.server:app --reload --port 8080
```

### Example Queries

```python
import httpx

# Query for first class
response = httpx.post("http://localhost:8080/query", json={
    "message": "Какая завтра первая пара у группы ИКМО-05-21?"
})

# Find meeting slots
response = httpx.post("http://localhost:8080/query", json={
    "message": "Когда преподаватель Иванов может встретиться с группами ИКМО-05-21 и ИКМО-06-21?"
})
```

```bash
# First class lookup
curl -X POST http://localhost:8080/query \
    -H "Content-Type: application/json" \
    -d '{"message": "Какая первая пара у группы ИКМО-05-21 14.11.2025?"}'

# Teacher schedule
curl -X POST http://localhost:8080/query \
    -H "Content-Type: application/json" \
    -d '{"message": "Покажи расписание преподавателя Иванов с 14.11.2025 по 15.11.2025"}'

# Meeting slots
curl -X POST http://localhost:8080/query \
    -H "Content-Type: application/json" \
    -d '{"message": "Найди окно для встречи у Иванова и групп ИКМО-05-21, ИКМО-06-21 на следующей неделе"}'
```

### API Response Shape

```json
{
    "response": "Краткий ответ для пользователя",
    "intent": "find_meeting_slot",
    "tool_calls": [
        {"tool": "find_common_free_slots", "parameters": {"teacher_norm": "иванов"}}
    ],
    "data": {
        "slots": [
            {
                "start": "2025-11-14T11:30:00",
                "end": "2025-11-14T12:30:00",
                "duration_minutes": 60,
                "confidence_score": 1.0
            }
        ],
        "requested_groups": ["икмо-05-21"],
        "requested_teacher": "иванов"
    },
    "error": null
}
```

## Docker Deployment

```bash
# Build and run both API and Postgres services
docker compose up --build

# Access API at http://localhost:8080
# Postgres exposed at localhost:5432 (user/password: schedule/schedule)
```

The compose stack mounts `min (1).db` into the Postgres container as an initialization asset. Ensure this file contains a compatible SQL dump; it will be executed on first startup.

## Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=html
```

## Deployment Notes

- Ensure PostgreSQL is reachable with the expected schema (see `specs/main/data-model.md`).
- Provide an OpenAI-compatible API key (or adjust `LLM_MODEL` to your provider).
- Configure `LOG_LEVEL` and external logging sinks for production observability.
- Use the supplied Docker Compose file or build the image manually for container deployments.

## Development with GitHub Copilot

See [COPILOT_GUIDE.md](COPILOT_GUIDE.md) for detailed prompts and implementation steps.

## License

MIT
