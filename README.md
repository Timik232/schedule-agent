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
- Python 3.11+ (tested on 3.11, 3.12, 3.13)
- PostgreSQL database (can use Docker Compose)
- SQLite seed database file (`min (1).db` or similar)
- LLM API key (GitHub Copilot or LM Studio)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd schedule-ai-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Generate pinned requirements (optional)
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=copilot  # or lm_studio
LLM_API_BASE_URL=https://api.githubcopilot.com  # or http://localhost:1234/v1 for LM Studio
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o-mini-copilot
LLM_TEMPERATURE=0.0

# Database Configuration
DATABASE_URL=postgresql+asyncpg://schedule:schedule@localhost:5432/schedule
SQLITE_SEED_PATH="./min (1).db"

# Security
MAX_SQL_ROWS=1000
QUERY_TIMEOUT_SECONDS=30
```

**Important**: Wrap `SQLITE_SEED_PATH` in quotes if the path contains spaces.

### Database Setup

**Required**: You must have a SQLite seed database file (e.g., `min (1).db`) containing schedule data.

1. **Place the SQLite file** in your project root or any accessible location
2. **Update `.env`** to point to it: `SQLITE_SEED_PATH="./min (1).db"`
3. **Start PostgreSQL** (via Docker Compose or standalone)
4. **Run the server** - migration happens automatically on startup:

```bash
# Using Docker Compose (recommended)
docker compose up -d database

# Or start standalone PostgreSQL and create database manually
# Then verify migration:
source venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8080
```

The first startup will:
- Create all required PostgreSQL tables
- Migrate data from SQLite to PostgreSQL (273k+ lessons, 2k+ teachers, etc.)
- Log "PostgreSQL seed completed successfully" when done

To verify migration:
```bash
docker exec schedule-agent-db psql -U schedule -d schedule -c "SELECT COUNT(*) FROM lesson;"
```

## Usage

### Start Development Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start server with auto-reload
uvicorn src.api.server:app --reload --port 8080
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

**Prerequisites**: Ensure `min (1).db` exists in the project root before running Docker Compose.

```bash
# Build and run both API and Postgres services
docker compose up --build

# Access API at http://localhost:8080
# Postgres exposed at localhost:5432 (user/password: schedule/schedule)
```

The FastAPI app automatically seeds PostgreSQL from the SQLite database on first startup:
- Docker Compose mounts `./min (1).db` → `/app/data/min.db` in the container
- The app reads `SQLITE_SEED_PATH=/app/data/min.db` from the container environment
- Migration runs automatically if PostgreSQL is empty
- Check logs for "PostgreSQL seed completed successfully"

If you rename or relocate the seed file, update both:
1. The volume mount in `docker-compose.yml`
2. The `SQLITE_SEED_PATH` environment variable

## Testing

```bash
# Activate virtual environment
source venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html
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
