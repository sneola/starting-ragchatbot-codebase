# RAG System Testing Framework

This directory contains the enhanced testing framework for the RAG system, providing comprehensive API endpoint testing, unit tests, and shared fixtures.

## Quick Start

```bash
# Install test dependencies  
uv sync --extra test

# Run all tests
uv run pytest backend/tests/ -v

# Run specific test categories
uv run pytest backend/tests/ -m api          # API endpoint tests only
uv run pytest backend/tests/ -m unit         # Unit tests only  
uv run pytest backend/tests/ -m integration  # Integration tests only

# Run specific test files
uv run pytest backend/tests/test_api.py -v   # API tests
uv run pytest backend/tests/test_demo.py -v  # Demo tests
```

## Test Structure

### Files Overview

- `conftest.py` - Shared fixtures and test configuration
- `test_api.py` - FastAPI endpoint tests for `/api/query` and `/api/courses`
- `test_unit.py` - Unit tests for individual RAG system components  
- `test_demo.py` - Demonstration tests showing framework capabilities
- `__init__.py` - Package initialization

### Test Categories (Markers)

- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests with mocked components

## Key Features

### 1. API Endpoint Testing

The framework provides comprehensive testing for FastAPI endpoints without requiring the actual server to run:

```python
def test_query_endpoint(test_client):
    response = test_client.post("/api/query", json={"query": "test"})
    assert response.status_code == 200
    assert "answer" in response.json()
```

### 2. Shared Fixtures

#### Mock RAG System (`mock_rag_system`)
- Mocks the entire RAGSystem with realistic responses
- Pre-configured with sample course data and session management

#### Test Client (`test_client`) 
- FastAPI TestClient with mocked dependencies
- No filesystem dependencies or external API calls
- Access to mock objects via `test_client.mock_rag`

#### Sample Data (`sample_course_data`)
- Structured course data for consistent testing
- Includes course title, lessons, and metadata

#### Temporary Resources (`temp_chroma_db`)
- Temporary ChromaDB directory for tests
- Automatically cleaned up after each test

### 3. Mocked External Dependencies

#### Anthropic API Mock (`mock_anthropic_api`)
```python
# Automatically mocks Anthropic API calls
def test_with_ai(mock_anthropic_api):
    # No real API calls made
    response = mock_anthropic_api.messages.create(...)
    assert response.content[0].text == "Test response from AI"
```

#### Configuration Mock (`mock_config`)
- Mock configuration with test-appropriate settings
- Temporary ChromaDB paths and test API keys

## Example Test Patterns

### API Endpoint Testing

```python
@pytest.mark.api
def test_query_with_sources(test_client):
    # Test different source formats
    test_client.mock_rag.query.return_value = (
        "Answer", 
        [{"text": "Source", "link": "https://example.com"}]
    )
    
    response = test_client.post("/api/query", json={"query": "test"})
    sources = response.json()["sources"]
    assert sources[0]["link"] == "https://example.com"
```

### Unit Testing with Mocks

```python
@pytest.mark.unit  
def test_component(mock_vector_store):
    # Test individual components with mocked dependencies
    titles = mock_vector_store.get_existing_course_titles()
    assert "Test Course" in titles
```

### Error Handling

```python
def test_error_handling(test_client):
    # Mock exceptions for error testing
    test_client.mock_rag.query.side_effect = Exception("Test error")
    
    response = test_client.post("/api/query", json={"query": "fail"})
    assert response.status_code == 500
```

## Configuration

The testing framework is configured via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = ["-v", "--tb=short", "--disable-warnings"]
markers = [
    "unit: Unit tests", 
    "integration: Integration tests",
    "api: API endpoint tests"
]
asyncio_mode = "auto"
```

## Benefits

1. **No External Dependencies**: Tests run without requiring actual databases, API keys, or file systems
2. **Fast Execution**: Mocked components provide instant responses  
3. **Isolated Testing**: Each test runs in isolation with fresh mocks
4. **Comprehensive Coverage**: API endpoints, unit components, and integration scenarios
5. **Easy Debugging**: Clear test structure with detailed assertions and error messages

## Adding New Tests

### For API Endpoints:
1. Add test to `test_api.py`
2. Use `@pytest.mark.api` marker
3. Use `test_client` fixture for HTTP requests
4. Access mocks via `test_client.mock_rag`

### For Unit Tests:
1. Add test to `test_unit.py` or create new file
2. Use `@pytest.mark.unit` marker  
3. Use specific component fixtures (e.g., `mock_vector_store`)

### For Integration Tests:
1. Use `@pytest.mark.integration` marker
2. Combine multiple fixtures for complex scenarios
3. Test interactions between components

## Troubleshooting

- **Import errors**: Check that you're running tests from the project root
- **Mock not working**: Verify fixture dependencies and ensure proper patching
- **Async issues**: Tests are configured for async mode automatically
- **Fixture scope issues**: Most fixtures use function scope for isolation