"""
Shared test fixtures and configuration for the RAG system tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API for testing without actual API calls."""
    with patch("anthropic.Anthropic") as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        # Mock the messages.create method
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response from AI")]
        mock_response.stop_reason = "end_turn"

        mock_instance.messages.create.return_value = mock_response

        yield mock_instance


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_chroma_db):
    """Mock configuration for tests."""
    config = Mock()
    config.CHROMA_PATH = temp_chroma_db
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    return config


@pytest.fixture
def sample_course_data():
    """Sample course data for testing."""
    return {
        "title": "Test Course",
        "content": """Course Title: Test Course
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson-0
This is the introduction to the test course.

Lesson 1: Advanced Topics
Lesson Link: https://example.com/lesson-1
This covers advanced topics in the test course.
""",
        "lessons": [
            {
                "title": "Introduction",
                "link": "https://example.com/lesson-0",
                "content": "This is the introduction to the test course.",
            },
            {
                "title": "Advanced Topics",
                "link": "https://example.com/lesson-1",
                "content": "This covers advanced topics in the test course.",
            },
        ],
    }


@pytest.fixture
def mock_vector_store(mock_config, sample_course_data):
    """Mock VectorStore for testing."""
    with patch("vector_store.VectorStore") as mock_vs_class:
        mock_vs = Mock()
        mock_vs_class.return_value = mock_vs

        # Mock methods
        mock_vs.get_existing_course_titles.return_value = [sample_course_data["title"]]
        mock_vs.search_course_content.return_value = [
            {
                "content": sample_course_data["lessons"][0]["content"],
                "metadata": {
                    "course_title": sample_course_data["title"],
                    "lesson_title": sample_course_data["lessons"][0]["title"],
                    "lesson_link": sample_course_data["lessons"][0]["link"],
                },
            }
        ]
        mock_vs.get_course_outline.return_value = {
            "title": sample_course_data["title"],
            "lessons": sample_course_data["lessons"],
        }

        yield mock_vs


@pytest.fixture
def mock_rag_system(mock_config, mock_anthropic_api, mock_vector_store):
    """Mock RAGSystem for API testing."""
    with patch("rag_system.RAGSystem") as mock_rag_class:
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag

        # Mock session manager
        mock_session_manager = Mock()
        mock_session_manager.create_session.return_value = "test-session-123"
        mock_rag.session_manager = mock_session_manager

        # Mock query method
        mock_rag.query.return_value = (
            "This is a test response from the RAG system.",
            [
                {
                    "text": "Test Course - Introduction",
                    "link": "https://example.com/lesson-0",
                }
            ],
        )

        # Mock analytics method
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"],
        }

        # Mock add_course_folder method
        mock_rag.add_course_folder.return_value = (1, 2)  # 1 course, 2 chunks

        yield mock_rag


@pytest.fixture
def test_client(mock_rag_system):
    """Create a test client with simplified inline app."""
    from typing import List, Optional

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    # Create app inline
    app = FastAPI(title="Test RAG API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceCitation(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceCitation]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            formatted_sources = []
            for source in sources:
                if isinstance(source, dict):
                    formatted_sources.append(
                        SourceCitation(
                            text=source.get("text", ""), link=source.get("link")
                        )
                    )
                else:
                    formatted_sources.append(
                        SourceCitation(text=str(source), link=None)
                    )

            return QueryResponse(
                answer=answer, sources=formatted_sources, session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Create client with mock attached for test access
    client = TestClient(app)
    client.mock_rag = mock_rag_system

    return client
