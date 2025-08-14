#!/bin/bash
# Code formatting script for the RAG chatbot project

set -e

echo "🔧 Running code quality checks and formatting..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

# Install dev dependencies if needed
echo "📦 Ensuring dev dependencies are installed..."
uv sync --extra dev

# Format code with black
echo "🖤 Formatting code with black..."
uv run black backend/

# Sort imports with isort
echo "📋 Sorting imports with isort..."
uv run isort backend/

# Run linting with flake8
echo "🔍 Running linting with flake8..."
uv run flake8 backend/

echo "✅ Code quality checks completed successfully!"