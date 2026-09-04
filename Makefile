.PHONY: help up down ingest

help:
	@echo "  make up         - Start container"
	@echo "  make down       - Stop Docker"
	@echo "  make ingest     - Ingest dataset"

up:
	docker-compose up -d

down:
	docker-compose down -v

ingest: up
	.\.venv\Scripts\activate
	python -m src.data.ingest

.DEFAULT_GOAL := help