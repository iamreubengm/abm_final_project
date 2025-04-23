"""
CSS styles for the application.
"""

MAIN_CSS = """
    .main-header {
        font-size: 2.5rem;
        color: #3366CC;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #3366CC;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3366CC;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .agent-response {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f1f7ff;
        margin-bottom: 1rem;
        border-left: 4px solid #3366CC;
    }
"""

def apply_css():
    """Apply the main CSS styles to the application."""
    import streamlit as st
    st.markdown(f"<style>{MAIN_CSS}</style>", unsafe_allow_html=True) 