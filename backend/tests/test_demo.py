"""
Demonstration tests showing the enhanced testing framework capabilities.
"""

import pytest
from fastapi import status


@pytest.mark.api
class TestDemoAPI:
    """Demo tests showing API testing capabilities."""

    def test_api_query_basic_functionality(self, test_client):
        """Demo: Test basic query functionality."""
        response = test_client.post(
            "/api/query", json={"query": "What can you tell me about this course?"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify sources are properly formatted
        if data["sources"]:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source

        print(f"✓ API query test passed - received response: {data['answer'][:50]}...")

    def test_api_courses_analytics(self, test_client):
        """Demo: Test course analytics endpoint."""
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        print(f"✓ Course analytics test passed - {data['total_courses']} courses found")

    def test_session_continuity(self, test_client):
        """Demo: Test session continuity across multiple requests."""
        # First request
        response1 = test_client.post(
            "/api/query", json={"query": "Hello, can you help me?"}
        )

        assert response1.status_code == status.HTTP_200_OK
        session_id = response1.json()["session_id"]

        # Second request with same session
        response2 = test_client.post(
            "/api/query",
            json={"query": "What did I just ask you?", "session_id": session_id},
        )

        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["session_id"] == session_id

        print(f"✓ Session continuity test passed - session_id: {session_id}")

    def test_error_handling(self, test_client):
        """Demo: Test error handling."""
        # Test with mock exception
        test_client.mock_rag.query.side_effect = Exception("Mock error")

        response = test_client.post("/api/query", json={"query": "This should fail"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Mock error" in response.json()["detail"]

        # Reset mock for other tests
        test_client.mock_rag.query.side_effect = None
        test_client.mock_rag.query.return_value = (
            "Test response",
            [{"text": "Test source", "link": "https://example.com"}],
        )

        print("✓ Error handling test passed")


@pytest.mark.integration
class TestDemoIntegration:
    """Demo integration tests using mocked components."""

    def test_mock_configurations(self, mock_config, mock_vector_store, mock_rag_system):
        """Demo: Test that all mocks are properly configured."""
        # Test mock_config
        assert mock_config.CHROMA_PATH is not None
        assert mock_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

        # Test mock_vector_store
        course_titles = mock_vector_store.get_existing_course_titles()
        assert "Test Course" in course_titles

        # Test mock_rag_system
        response, sources = mock_rag_system.query("test", "session-123")
        assert isinstance(response, str)
        assert isinstance(sources, list)

        print("✓ All mocks configured correctly")

    def test_sample_data_structure(self, sample_course_data):
        """Demo: Test sample data structure."""
        assert "title" in sample_course_data
        assert "lessons" in sample_course_data
        assert len(sample_course_data["lessons"]) == 2

        lesson = sample_course_data["lessons"][0]
        assert "title" in lesson
        assert "link" in lesson
        assert "content" in lesson

        print(f"✓ Sample course data structure valid: {sample_course_data['title']}")


@pytest.mark.unit
class TestDemoUnit:
    """Demo unit tests showing component testing."""

    def test_temp_directory_fixture(self, temp_chroma_db):
        """Demo: Test temporary directory fixture."""
        import os

        assert os.path.exists(temp_chroma_db)
        assert os.path.isdir(temp_chroma_db)
        print(f"✓ Temporary ChromaDB directory created: {temp_chroma_db}")

    def test_anthropic_mock(self, mock_anthropic_api):
        """Demo: Test Anthropic API mock."""
        # Test that mock is configured
        assert mock_anthropic_api is not None

        # Test mock response structure
        response = mock_anthropic_api.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        assert response.content[0].text == "Test response from AI"
        assert response.stop_reason == "end_turn"

        print("✓ Anthropic API mock working correctly")
