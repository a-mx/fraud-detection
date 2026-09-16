SHELL := /bin/bash
API_URL ?= http://localhost:8000
MODEL ?= xgb

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.PHONY: help up down build logs ingest reload predict health train promote deploy

help:
	@echo "Available commands:"
	@echo "  make up                - start the full stack"
	@echo "  make down              - stop the stack"
	@echo "  make build             - rebuild images"
	@echo "  make logs              - tail API logs"
	@echo ""
	@echo "  make ingest     		- ingest dataset"
	@echo ""
	@echo "  make train MODEL=xgb   - train a model (baseline|xgb|mlp)"
	@echo "  make promote MODEL=xgb - train and promote to production"
	@echo "  make reload            - reload model in API"
	@echo ""
	@echo "  make health            - check API status"
	@echo "  make predict           - send a test request"

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python -m src.data.ingest

build:
	docker compose build

logs:
	docker compose logs -f api

train:
	python -m src.main --model $(MODEL)

promote:
	python -m src.main --model $(MODEL) --promote

reload:
ifeq ($(RELOAD_TOKEN),)
	@echo "Error: RELOAD_TOKEN is not set in .env"
	@exit 1
endif
	@curl -fsS -X POST $(API_URL)/reload \
		-H "X-Reload-Token: $(RELOAD_TOKEN)" | python -m json.tool

health:
	@curl -fsS $(API_URL)/health | python -m json.tool

predict:
	@curl -fsS -X POST $(API_URL)/predict \
		-H "Content-Type: application/json" \
		-d @test_payload.json | python -m json.tool

deploy:
	$(MAKE) promote MODEL=$(MODEL)
	$(MAKE) reload
	$(MAKE) health
