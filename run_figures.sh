#!/bin/bash
# Generate all paper figures using uv
set -e

echo "Generating paper figures..."
uv run python scripts/generate_all_figures.py "$@"
