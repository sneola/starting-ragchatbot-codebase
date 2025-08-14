#!/usr/bin/env python3
"""
Quick test script for CourseOutlineTool functionality
"""

from config import config
from search_tools import CourseOutlineTool
from vector_store import VectorStore


def test_course_outline_tool():
    """Test the course outline tool with a sample course"""
    print("Testing CourseOutlineTool...")

    # Initialize vector store
    vector_store = VectorStore(
        config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
    )

    # Initialize outline tool
    outline_tool = CourseOutlineTool(vector_store)

    # Test tool definition
    tool_def = outline_tool.get_tool_definition()
    print(f"Tool name: {tool_def['name']}")
    print(f"Tool description: {tool_def['description']}")

    # Test getting existing course titles
    existing_courses = vector_store.get_existing_course_titles()
    print(f"\nExisting courses in database: {existing_courses}")

    if existing_courses:
        # Test with first available course
        test_course = existing_courses[0]
        print(f"\nTesting with course: {test_course}")

        # Execute tool
        result = outline_tool.execute(course_title=test_course)
        print(f"\nOutline result:\n{result}")

        # Test sources
        sources = outline_tool.last_sources
        print(f"\nSources: {sources}")
    else:
        print("No courses found in database. Please ensure course data is loaded.")

    # Test with partial match (if we have courses)
    if existing_courses:
        # Try with partial course name
        partial_name = existing_courses[0].split()[0]  # First word of first course
        print(f"\nTesting with partial name: '{partial_name}'")
        result = outline_tool.execute(course_title=partial_name)
        print(f"Partial match result:\n{result}")


if __name__ == "__main__":
    test_course_outline_tool()
