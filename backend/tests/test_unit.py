"""
Unit tests for individual RAG system components.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.mark.unit
class TestVectorStore:
    """Unit tests for VectorStore functionality."""

    def test_vector_store_initialization(self, mock_config):
        """Test VectorStore initialization."""
        from vector_store import VectorStore

        with patch("chromadb.PersistentClient") as mock_client:
            vector_store = VectorStore(
                db_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS,
            )

            assert vector_store.max_results == mock_config.MAX_RESULTS
            mock_client.assert_called_once_with(path=mock_config.CHROMA_PATH)

    def test_get_existing_course_titles(self, mock_vector_store):
        """Test retrieving existing course titles."""
        titles = mock_vector_store.get_existing_course_titles()
        assert isinstance(titles, list)
        assert "Test Course" in titles

    def test_search_course_content(self, mock_vector_store):
        """Test searching course content."""
        results = mock_vector_store.search_course_content("test query")
        assert isinstance(results, list)
        if results:
            assert "content" in results[0]
            assert "metadata" in results[0]


@pytest.mark.unit
class TestDocumentProcessor:
    """Unit tests for DocumentProcessor functionality."""

    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        from document_processor import DocumentProcessor

        processor = DocumentProcessor()
        assert processor is not None

    def test_parse_course_document(self, sample_course_data):
        """Test parsing course document format."""
        from document_processor import DocumentProcessor

        processor = DocumentProcessor()

        with patch.object(processor, "process_course_file") as mock_process:
            mock_process.return_value = (
                sample_course_data["title"],
                sample_course_data["lessons"],
            )

            title, lessons = processor.process_course_file("dummy_path")

            assert title == sample_course_data["title"]
            assert len(lessons) == len(sample_course_data["lessons"])


@pytest.mark.unit
class TestSearchTools:
    """Unit tests for search tools."""

    def test_course_search_tool_initialization(self, mock_vector_store):
        """Test CourseSearchTool initialization."""
        from search_tools import CourseSearchTool

        tool = CourseSearchTool(mock_vector_store)
        assert tool.vector_store == mock_vector_store

    def test_course_search_tool_definition(self, mock_vector_store):
        """Test CourseSearchTool tool definition."""
        from search_tools import CourseSearchTool

        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["name"] == "course_search"

    def test_course_outline_tool_initialization(self, mock_vector_store):
        """Test CourseOutlineTool initialization."""
        from search_tools import CourseOutlineTool

        tool = CourseOutlineTool(mock_vector_store)
        assert tool.vector_store == mock_vector_store

    def test_course_outline_tool_definition(self, mock_vector_store):
        """Test CourseOutlineTool tool definition."""
        from search_tools import CourseOutlineTool

        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["name"] == "course_outline"

    def test_course_search_execution(self, mock_vector_store):
        """Test CourseSearchTool execution."""
        from search_tools import CourseSearchTool

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        assert isinstance(result, str)
        assert tool.last_sources is not None

    def test_course_outline_execution(self, mock_vector_store, sample_course_data):
        """Test CourseOutlineTool execution."""
        from search_tools import CourseOutlineTool

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title=sample_course_data["title"])

        assert isinstance(result, str)
        assert tool.last_sources is not None


@pytest.mark.unit
class TestAIGenerator:
    """Unit tests for AIGenerator functionality."""

    def test_ai_generator_initialization(self, mock_config, mock_anthropic_api):
        """Test AIGenerator initialization."""
        from ai_generator import AIGenerator

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            generator = AIGenerator(mock_config.ANTHROPIC_MODEL)
            assert generator.model == mock_config.ANTHROPIC_MODEL

    def test_generate_response_with_tools(self, mock_config, mock_anthropic_api):
        """Test response generation with tools."""
        from ai_generator import AIGenerator

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            generator = AIGenerator(mock_config.ANTHROPIC_MODEL)

            mock_tools = [Mock()]
            mock_tools[0].get_tool_definition.return_value = {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {},
            }

            response = generator.generate_response(
                query="Test query", tools=mock_tools, conversation_history=[]
            )

            assert isinstance(response, str)

    def test_generate_response_without_tools(self, mock_config, mock_anthropic_api):
        """Test response generation without tools."""
        from ai_generator import AIGenerator

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            generator = AIGenerator(mock_config.ANTHROPIC_MODEL)

            response = generator.generate_response(
                query="Test query", tools=[], conversation_history=[]
            )

            assert isinstance(response, str)


@pytest.mark.unit
class TestSessionManager:
    """Unit tests for SessionManager functionality."""

    def test_session_manager_initialization(self):
        """Test SessionManager initialization."""
        from session_manager import SessionManager

        manager = SessionManager(max_history=10)
        assert manager.max_history == 10

    def test_create_session(self):
        """Test session creation."""
        from session_manager import SessionManager

        manager = SessionManager()
        session_id = manager.create_session()

        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_add_and_get_conversation(self):
        """Test adding and retrieving conversation history."""
        from session_manager import SessionManager

        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_to_conversation(session_id, "user", "Hello")
        manager.add_to_conversation(session_id, "assistant", "Hi there")

        history = manager.get_conversation_history(session_id)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there"

    def test_session_history_limit(self):
        """Test session history length limiting."""
        from session_manager import SessionManager

        manager = SessionManager(max_history=2)
        session_id = manager.create_session()

        # Add more messages than the limit
        manager.add_to_conversation(session_id, "user", "Message 1")
        manager.add_to_conversation(session_id, "assistant", "Response 1")
        manager.add_to_conversation(session_id, "user", "Message 2")
        manager.add_to_conversation(session_id, "assistant", "Response 2")
        manager.add_to_conversation(session_id, "user", "Message 3")

        history = manager.get_conversation_history(session_id)

        # Should only keep the last 2 messages
        assert len(history) <= 2


@pytest.mark.unit
class TestRAGSystem:
    """Unit tests for RAGSystem integration."""

    def test_rag_system_initialization(self, mock_config):
        """Test RAGSystem initialization."""
        from rag_system import RAGSystem

        with (
            patch("rag_system.VectorStore"),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
        ):

            rag_system = RAGSystem(mock_config)
            assert rag_system.config == mock_config

    def test_query_processing(self, mock_config, mock_anthropic_api, mock_vector_store):
        """Test query processing workflow."""
        from rag_system import RAGSystem

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
        ):

            # Setup mocks
            mock_ai_instance = Mock()
            mock_ai_gen.return_value = mock_ai_instance
            mock_ai_instance.generate_response.return_value = "Test response"

            mock_session_instance = Mock()
            mock_session_mgr.return_value = mock_session_instance
            mock_session_instance.get_conversation_history.return_value = []

            rag_system = RAGSystem(mock_config)
            response, sources = rag_system.query("Test query", "test-session")

            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_get_course_analytics(self, mock_config, mock_vector_store):
        """Test course analytics retrieval."""
        from rag_system import RAGSystem

        with (
            patch("rag_system.VectorStore", return_value=mock_vector_store),
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
        ):

            rag_system = RAGSystem(mock_config)
            analytics = rag_system.get_course_analytics()

            assert "total_courses" in analytics
            assert "course_titles" in analytics
            assert isinstance(analytics["total_courses"], int)
            assert isinstance(analytics["course_titles"], list)
