"""
API endpoint tests for the RAG system FastAPI application.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import status


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint."""

    def test_query_with_new_session(self, test_client):
        """Test querying without providing a session_id."""
        response = test_client.post(
            "/api/query", json={"query": "What is this course about?"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)

        # Verify the mock was called correctly
        test_client.mock_rag.session_manager.create_session.assert_called_once()
        test_client.mock_rag.query.assert_called_once_with(
            "What is this course about?", "test-session-123"
        )

    def test_query_with_existing_session(self, test_client):
        """Test querying with an existing session_id."""
        session_id = "existing-session-456"

        response = test_client.post(
            "/api/query",
            json={
                "query": "Tell me more about advanced topics",
                "session_id": session_id,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["session_id"] == session_id

        # Verify session creation was not called
        assert not test_client.mock_rag.session_manager.create_session.called
        test_client.mock_rag.query.assert_called_with(
            "Tell me more about advanced topics", session_id
        )

    def test_query_response_format(self, test_client):
        """Test the structure of the query response."""
        response = test_client.post("/api/query", json={"query": "Sample query"})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify sources format
        sources = data["sources"]
        assert isinstance(sources, list)
        if sources:
            source = sources[0]
            assert "text" in source
            assert "link" in source

    def test_query_with_dict_sources(self, test_client):
        """Test query response with dict-formatted sources."""
        # Mock different source format
        test_client.mock_rag.query.return_value = (
            "Test answer",
            [{"text": "Source text", "link": "https://example.com"}],
        )

        response = test_client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        sources = data["sources"]
        assert len(sources) == 1
        assert sources[0]["text"] == "Source text"
        assert sources[0]["link"] == "https://example.com"

    def test_query_with_string_sources(self, test_client):
        """Test query response with string-formatted sources (fallback)."""
        # Mock old string format
        test_client.mock_rag.query.return_value = (
            "Test answer",
            ["String source 1", "String source 2"],
        )

        response = test_client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        sources = data["sources"]
        assert len(sources) == 2
        assert sources[0]["text"] == "String source 1"
        assert sources[0]["link"] is None
        assert sources[1]["text"] == "String source 2"
        assert sources[1]["link"] is None

    def test_query_empty_string(self, test_client):
        """Test query with empty string."""
        response = test_client.post("/api/query", json={"query": ""})

        assert response.status_code == status.HTTP_200_OK
        test_client.mock_rag.query.assert_called_with("", "test-session-123")

    def test_query_missing_field(self, test_client):
        """Test query request with missing query field."""
        response = test_client.post("/api/query", json={})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_invalid_json(self, test_client):
        """Test query with invalid JSON."""
        response = test_client.post("/api/query", data="invalid json")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_exception_handling(self, test_client):
        """Test query endpoint exception handling."""
        # Mock RAG system to raise exception
        test_client.mock_rag.query.side_effect = Exception("Test error")

        response = test_client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Test error" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""

    def test_get_courses_success(self, test_client):
        """Test successful retrieval of course statistics."""
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Test Course"]

        test_client.mock_rag.get_course_analytics.assert_called_once()

    def test_get_courses_empty(self, test_client):
        """Test courses endpoint with no courses."""
        test_client.mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_exception_handling(self, test_client):
        """Test courses endpoint exception handling."""
        test_client.mock_rag.get_course_analytics.side_effect = Exception(
            "Analytics error"
        )

        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Analytics error" in response.json()["detail"]


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test CORS and middleware functionality."""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses."""
        response = test_client.get("/api/courses")

        # Check common CORS headers
        assert "access-control-allow-origin" in response.headers

    def test_options_request(self, test_client):
        """Test OPTIONS preflight request handling."""
        response = test_client.options("/api/query")

        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]


@pytest.mark.api
class TestRequestValidation:
    """Test request validation and edge cases."""

    def test_query_request_validation(self, test_client):
        """Test various query request validation scenarios."""
        # Valid minimal request
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == status.HTTP_200_OK

        # Valid request with session_id
        response = test_client.post(
            "/api/query", json={"query": "test", "session_id": "custom-session"}
        )
        assert response.status_code == status.HTTP_200_OK

        # Valid request with None session_id (should create new)
        response = test_client.post(
            "/api/query", json={"query": "test", "session_id": None}
        )
        assert response.status_code == status.HTTP_200_OK

    def test_invalid_http_methods(self, test_client):
        """Test invalid HTTP methods on endpoints."""
        # GET on query endpoint (should be POST)
        response = test_client.get("/api/query")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        # POST on courses endpoint (should be GET)
        response = test_client.post("/api/courses", json={})
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_nonexistent_endpoints(self, test_client):
        """Test requests to non-existent endpoints."""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

        response = test_client.post("/api/invalid")
        assert response.status_code == status.HTTP_404_NOT_FOUND
