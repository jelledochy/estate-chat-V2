.PHONY: help backend api frontend ui dev lint test check

PYTHON := $(CURDIR)/.venv/bin/python
STREAMLIT := $(CURDIR)/.venv/bin/streamlit

HOST ?= 0.0.0.0
API_PORT ?= 8001
UI_PORT ?= 8510
BACKEND_API_URL ?= http://localhost:$(API_PORT)

PYTHONPATH := $(CURDIR):$(CURDIR)/backend
API_APP := backend.app.main:app
UI_APP := streamlit_app/app.py

help:
	@echo "Available commands:"
	@echo "  make backend   Run FastAPI backend on http://localhost:$(API_PORT)"
	@echo "  make frontend  Run Streamlit frontend on http://localhost:$(UI_PORT)"
	@echo "  make dev       Run backend and frontend together"
	@echo "  make lint      Run Ruff checks"
	@echo "  make test      Run pytest"
	@echo "  make check     Run lint and tests"

backend:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn $(API_APP) --host $(HOST) --port $(API_PORT) --reload

api: backend

frontend:
	BACKEND_API_URL=$(BACKEND_API_URL) PYTHONPATH=$(PYTHONPATH) $(STREAMLIT) run $(UI_APP) --server.address $(HOST) --server.port $(UI_PORT)

ui: frontend

dev:
	@echo "Backend:  http://localhost:$(API_PORT)"
	@echo "Frontend: http://localhost:$(UI_PORT)"
	@if curl -fsS http://localhost:$(API_PORT)/health >/dev/null 2>&1; then \
		echo "Backend already running on http://localhost:$(API_PORT); starting frontend only."; \
		BACKEND_API_URL=$(BACKEND_API_URL) PYTHONPATH=$(PYTHONPATH) $(STREAMLIT) run $(UI_APP) --server.address $(HOST) --server.port $(UI_PORT); \
	else \
		trap 'kill 0' INT TERM EXIT; \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn $(API_APP) --host $(HOST) --port $(API_PORT) --reload & \
		BACKEND_API_URL=$(BACKEND_API_URL) PYTHONPATH=$(PYTHONPATH) $(STREAMLIT) run $(UI_APP) --server.address $(HOST) --server.port $(UI_PORT); \
	fi

lint:
	$(CURDIR)/.venv/bin/ruff check backend streamlit_app

test:
	$(CURDIR)/.venv/bin/pytest -q

check: lint test
