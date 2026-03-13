import pytest
from unittest.mock import patch, MagicMock
from js_ai import LibraryAI

# ==========================================
# FIXTURES (The Mock Setup)
# ==========================================
# We use @patch to intercept the heavy ML and DB calls.
# When LibraryAI tries to load Chroma or Ollama, it gets a lightweight fake instead.

@pytest.fixture
@patch('js_ai.Chroma')
@patch('js_ai.HuggingFaceEmbeddings')
@patch('js_ai.OllamaLLM')
@patch('js_ai.DuckDuckGoSearchRun')
def ai_app(mock_ddg, mock_llm, mock_embeddings, mock_chroma):
    """Creates a LibraryAI instance with all ML and DB components mocked out."""
    # Initialize the app with a fake directory
    app = LibraryAI("./fake_test_dir")
    return app

# ==========================================
# TEST SUITE
# ==========================================

def test_command_routing_intercepts_system_commands(ai_app):
    """Tests if the router correctly identifies database commands vs chat queries."""
    
    # Should return True (it is a system command)
    assert ai_app.process_system_command("-info my name is Alice") == True
    assert ai_app.process_system_command("-facts") == True
    assert ai_app.process_system_command("-clear") == True
    
    # Should return False (it is a standard RAG chat query)
    assert ai_app.process_system_command("What is a pointer in C++?") == False
    assert ai_app.process_system_command("-strict explain RAG") == False # Routing flag, not a system command

def test_parse_search_flags_strict_mode(ai_app):
    """Tests if the -strict flag correctly disables memory search."""
    clean_query, use_lib, use_mem, force_web = ai_app.parse_search_flags("-strict how does routing work?")
    
    assert clean_query == "how does routing work?"
    assert use_lib == True
    assert use_mem == False
    assert force_web == False

def test_parse_search_flags_chat_mode(ai_app):
    """Tests if the -chat flag correctly disables library search."""
    clean_query, use_lib, use_mem, force_web = ai_app.parse_search_flags("-chat what did we just talk about?")
    
    assert clean_query == "what did we just talk about?"
    assert use_lib == False
    assert use_mem == True
    assert force_web == False

def test_parse_search_flags_web_mode(ai_app):
    """Tests if the -web flag forces live internet and disables local DBs."""
    clean_query, use_lib, use_mem, force_web = ai_app.parse_search_flags("-web weather in Boston")
    
    assert clean_query == "weather in Boston"
    assert use_lib == False
    assert use_mem == False
    assert force_web == True

def test_parse_search_flags_default_mode(ai_app):
    """Tests if a standard query leaves all local searches enabled."""
    clean_query, use_lib, use_mem, force_web = ai_app.parse_search_flags("explain binary search")
    
    assert clean_query == "explain binary search"
    assert use_lib == True
    assert use_mem == True
    assert force_web == False

def test_short_term_buffer_truncation(ai_app):
    """Tests if the RAM buffer strictly caps at 6 items to save LLM tokens."""
    
    # Add 7 items to the buffer
    ai_app._update_short_term_buffer("Message 1")
    ai_app._update_short_term_buffer("Message 2")
    ai_app._update_short_term_buffer("Message 3")
    ai_app._update_short_term_buffer("Message 4")
    ai_app._update_short_term_buffer("Message 5")
    ai_app._update_short_term_buffer("Message 6")
    ai_app._update_short_term_buffer("Message 7")
    
    # Assert it only kept 6
    assert len(ai_app.recent_chat_buffer) == 6
    # Assert it dropped Message 1 and kept Message 7
    assert ai_app.recent_chat_buffer[0] == "Message 2"
    assert ai_app.recent_chat_buffer[-1] == "Message 7"

def test_info_command_saves_to_memory(ai_app):
    """Tests the explicit fact saving logic."""
    # Mock the memory_db's add_texts method so we can spy on it
    ai_app.memory_db.add_texts = MagicMock()
    
    # Execute the command
    ai_app.process_system_command("-info I code in C#")
    
    # Assert the database write function was called with the exact right metadata
    ai_app.memory_db.add_texts.assert_called_once()
    called_args, called_kwargs = ai_app.memory_db.add_texts.call_args
    assert called_kwargs['texts'] == ["User provided an explicit fact to remember: I code in C#"]
    assert called_kwargs['metadatas'] == [{"role": "user", "type": "explicit_fact"}]