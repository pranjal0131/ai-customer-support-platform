.PHONY: install install-ml dev backend frontend test lint format build seed datasets train-baselines docker-up docker-down

install:
	python -m pip install -r requirements.txt
	cd frontend && npm install

install-ml:
	python -m pip install -r requirements-ml.txt

dev:
	docker compose up --build

backend:
	uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	pytest
	cd frontend && npm run test -- --run

lint:
	ruff check backend ml tests
	cd frontend && npm run lint

format:
	ruff format backend ml tests
	cd frontend && npm run format

build:
	cd frontend && npm run build

seed:
	python -m ml.scripts.seed_demo_tickets

datasets:
	python -m ml.scripts.download_datasets --smoke

train-baselines:
	python -m ml.scripts.train_intent_baselines --smoke
	python -m ml.scripts.train_urgency

docker-up:
	docker compose up --build

docker-down:
	docker compose down
