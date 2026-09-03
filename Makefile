.PHONY: help up down ingest connect

help:
    @echo "  make up         - Start container"
    @echo "  make down       - Stop Docker"
    @echo "  make ingest     - Ingest dataset"

up:
    docker-compose up -d

down:
    docker-compose down -v

ingest: up
    python src/data/ingest.py

.DEFAULT_GOAL := help