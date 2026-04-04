#!bin/bash

rm -rf /tmp/ray
uv run ray stop --force
uv run ray start --head
uv run --active python main.py