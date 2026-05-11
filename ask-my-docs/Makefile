VENV     := backend/.rag
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
UVICORN  := $(VENV)/bin/uvicorn

.PHONY: install dev test eval docker-up docker-down

install:
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm install

dev:
	@echo "Starting backend and frontend in parallel..."
	@trap 'kill 0' SIGINT; \
	cd backend && $(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000 & \
	cd frontend && npm run dev & \
	wait

test:
	cd backend && $(PYTEST) tests/unit/ -v --cov=app --cov-report=term-missing

eval:
	cd backend && $(PYTHON) -m app.evaluation.ragas_runner

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
