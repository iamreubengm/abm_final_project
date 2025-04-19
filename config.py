# config.py
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# API Keys and configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model configuration
DEFAULT_MODEL = "claude-3-opus-20240229"
DEFAULT_MAX_TOKENS = 4096

# Agent configuration
AGENT_TYPES = [
    "budget",
    "investment",
    "debt",
    "savings",
    "tax"
]

# RAG configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "data/vector_db"

# Data paths
FINANCIAL_KB_PATH = "data/financial_kb"
USER_DATA_PATH = "data/user_data"
SYSTEM_PROMPTS_PATH = "prompts/system_prompts"
PROMPT_TEMPLATES_PATH = "prompts/templates"

# Streamlit configuration
STREAMLIT_PAGE_TITLE = "AI Personal Financial Portal"
STREAMLIT_PAGE_ICON = "💰"
STREAMLIT_LAYOUT = "wide"

def get_anthropic_client():
    """Initialize and return the Anthropic client."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    
    return Anthropic(api_key=ANTHROPIC_API_KEY)

# Configuration for agent interactions
AGENT_INTERACTION_SETTINGS = {
    "voting_threshold": 0.6,  # Minimum percentage of agents that must agree for consensus
    "debate_rounds": 2,       # Number of rounds in debate-based cooperation
    "multi_path_options": 3,  # Number of alternative strategies to generate
    "human_feedback_weight": 1.5  # Weight multiplier for human feedback
}