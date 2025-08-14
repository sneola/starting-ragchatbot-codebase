# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

```bash
# Start the development server
./run.sh

# Manual server start (from backend directory)
uv run uvicorn app:app --reload --port 8000

# Install dependencies
uv sync
uv sync --extra dev  # Include development tools

# Code quality and formatting
./scripts/format.sh   # Format code with black and isort, run linting
./scripts/check.sh    # Check code quality without making changes

# Individual quality tools
uv run black backend/           # Format code
uv run isort backend/          # Sort imports
uv run flake8 backend/         # Lint code

# Test specific components
python3 -m py_compile <filename>  # Syntax check
```

## Architecture Overview

### RAG System Core Components

**RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates all components:
- Integrates DocumentProcessor, VectorStore, AIGenerator, SessionManager, and ToolManager
- Handles end-to-end query processing workflow
- Manages tool registration and execution

**Vector Storage Layer**:
- **VectorStore** (`vector_store.py`) - ChromaDB wrapper with two collections:
  - `course_catalog`: Course metadata and lesson structures (searchable by title)
  - `course_content`: Chunked course content with lesson associations
- **DocumentProcessor** (`document_processor.py`) - Processes course documents with structured format:
  ```
  Course Title: [title]
  Course Link: [url] 
  Course Instructor: [instructor]
  
  Lesson 0: [title]
  Lesson Link: [url]
  [content...]
  ```

**AI Integration**:
- **AIGenerator** (`ai_generator.py`) - Anthropic Claude API wrapper with tool calling capabilities
- **Tool System** (`search_tools.py`) - Two specialized tools:
  - `CourseSearchTool`: Semantic search of course content with course/lesson filtering
  - `CourseOutlineTool`: Retrieves structured course outlines with lesson lists
- **SessionManager** (`session_manager.py`) - Conversation history management

### Data Flow

1. **Document Ingestion**: Course files → DocumentProcessor → CourseChunks → VectorStore
2. **Query Processing**: User query → RAGSystem → AIGenerator → Tool execution → Vector search → Response
3. **Tool Selection**: AI automatically chooses between content search vs outline retrieval based on query type

### Web Interface

- **Frontend**: Vanilla HTML/CSS/JS in `/frontend` with real-time chat interface
- **Backend**: FastAPI app serving static files and API endpoints (`/api/query`, `/api/courses`)
- **Session Management**: Frontend maintains session_id for conversation continuity

## Key Configuration

- **Environment**: Requires `ANTHROPIC_API_KEY` in `.env` file
- **Vector Database**: ChromaDB stored in `./chroma_db/` (created automatically)
- **Course Documents**: Place in `/docs` folder - auto-loaded on server startup
- **Model Settings**: Uses `claude-sonnet-4-20250514` and `all-MiniLM-L6-v2` embeddings

## Development Notes

- Course documents must follow specific format with "Course Title:", "Course Link:", "Course Instructor:" headers
- Vector store uses course titles as unique identifiers
- Tools track sources for UI display via `last_sources` attribute
- AI system prompt configured for single tool call per query maximum
- Server startup includes automatic document processing from `../docs`