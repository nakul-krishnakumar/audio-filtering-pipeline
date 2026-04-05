.PHONY: install download stop start_head start_worker run dashboard

DIR ?= ./data
ADDR ?= 127.0.0.1
PORT ?= 6379

install:
	uv sync

download:
	uv run python scripts/setup_dataset.py $(DIR)
	uv run python scripts/generate_test_manifest.py

stop:
	uv run ray stop --force || true
	rm -rf /tmp/ray

start_head:
	uv run ray stop --force || true
	rm -rf /tmp/ray
	uv run ray start --head --port=$(PORT)

start_worker:
	uv run ray stop --force || true
	rm -rf /tmp/ray
	uv run ray start --address="$(ADDR):$(PORT)"

run:
	uv run --active python main.py

run_head:
	uv run ray stop --force || true
	rm -rf /tmp/ray
	uv run ray start --head --port=$(PORT)
	uv run --active python main.py

dashboard:
	uv run python dashboard/app.py