#!/bin/bash
# Code quality check script (dry-run) for the RAG chatbot project

set -e

echo "🔍 Running code quality checks (dry-run)..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

# Install dev dependencies if needed
echo "📦 Ensuring dev dependencies are installed..."
uv sync --extra dev

# Check formatting with black (dry-run)
echo "🖤 Checking code formatting with black..."
if ! uv run black --check --diff backend/; then
    echo "❌ Code formatting issues found. Run 'scripts/format.sh' to fix."
    exit 1
fi

# Check import sorting with isort (dry-run)
echo "📋 Checking import sorting with isort..."
if ! uv run isort --check-only --diff backend/; then
    echo "❌ Import sorting issues found. Run 'scripts/format.sh' to fix."
    exit 1
fi

# Run linting with flake8
echo "🔍 Running linting with flake8..."
uv run flake8 backend/

echo "✅ All code quality checks passed!"