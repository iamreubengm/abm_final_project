"""
Common UI components for the application.
"""
import streamlit as st
import re
import plotly.graph_objs as go
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

def display_header(title, level=1):
    """Display a header with consistent styling."""
    if level == 1:
        st.markdown(f"<h1 class='main-header'>{title}</h1>", unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f"<h2 class='sub-header'>{title}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h{level}>{title}</h{level}>", unsafe_allow_html=True)

def display_metric_card(title: str, value: str, delta: Optional[str] = None) -> None:
    """Display a metric card with consistent styling.
    
    Args:
        title: Card title
        value: Main value to display
        delta: Optional delta value to show change
    """
    st.metric(title, value, delta)

def display_card(content):
    """Display content in a card with consistent styling."""
    st.markdown(f"""
        <div class='card' style='white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%;'>
            {content}
        </div>
    """, unsafe_allow_html=True)

def display_agent_response(response: str, title: str) -> None:
    """Display an AI agent's response with consistent styling.
    
    Args:
        response: Response text from the agent
        title: Title for the response section
    """
    st.markdown(f"""
        <div style='
            background-color: #f1f7ff;
            padding: 1rem;
            border-left: 4px solid #3366CC;
            border-radius: 0.5rem;
            color: #111;
            margin-top: 1rem;
        '>
            <h4>{title}</h4>
            <p>{response}</p>
        </div>
    """, unsafe_allow_html=True)

def create_navigation_button(label, view_name, icon=""):
    """Create a navigation button with consistent styling."""
    if st.button(f"{icon} {label}", use_container_width=True):
        st.session_state.current_view = view_name
        st.rerun()

def display_quick_stats(income, expenses):
    """Display quick stats in the sidebar."""
    st.subheader("Quick Stats")
    st.metric("Monthly Income", f"${income:,.2f}")
    st.metric("Monthly Expenses", f"${expenses:,.2f}")
    st.metric("Monthly Savings", f"${income - expenses:,.2f}")

def clean_text(text):
    """Minimal text cleaning function."""
    # Handle TextBlock wrapper
    if isinstance(text, str):
        if "[Text Block" in text:
            # Extract just the text content
            match = re.search(r"text='([^']*)'", text)
            if match:
                text = match.group(1)
        elif "TextBlock" in text:
            text = text.split('text="', 1)[1].rsplit('", type=', 1)[0]
    
    # Convert to string
    text = str(text)
    
    # Join split words and numbers
    text = re.sub(r'(\d)\s*([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])\s*(\d)', r'\1 \2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Clean up basic formatting
    text = text.replace('â¢', '•')
    text = text.replace('â', '-')
    text = text.replace('−', '-')
    text = text.replace('\\n', '\n')
    
    # Add spaces after punctuation
    text = re.sub(r'([.,!?:;])([^\s])', r'\1 \2', text)
    
    # Clean up spaces
    text = ' '.join(text.split())
    
    # Format bullet points
    text = text.replace('• ', '\n• ')
    
    # Format numbered lists
    text = re.sub(r'(\d+\.) ', r'\n\1 ', text)
    
    # Fix italicized text that got joined
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text.strip()

def display_chat_message(role, content):
    """Display a chat message with consistent styling."""
    with st.chat_message(role):
        cleaned_text = clean_text(content)
        st.markdown(f"""
            <div style='white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%;'>
                {cleaned_text}
            </div>
        """, unsafe_allow_html=True)

def format_expert_message(agent_type, message):
    """Format an expert's message."""
    cleaned_message = clean_text(message)
    if agent_type:
        return f"{agent_type.capitalize()} Expert:\n{cleaned_message}"
    return cleaned_message

def format_consensus_message(message):
    """Format a consensus message."""
    return clean_text(message)

def display_agent_comparison(title, content, heading_color="#3366CC", text_color="#333333"):
    """Display an agent comparison response with consistent styling."""
    st.markdown(f"""
        <div class='agent-response' style='white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%;'>
            <h4 style='color:{heading_color};'>{title}</h4>
            <p style='color:{text_color};'>{content}</p>
        </div>
    """, unsafe_allow_html=True)

def display_styled_agent_response(content, background_color="#f1f7ff", border_color="#3366CC"):
    """Display an agent response with custom styling."""
    st.markdown(f"""
        <div class='agent-response' style='background-color: {background_color}; padding: 1rem; border-left: 4px solid {border_color}; border-radius: 8px; margin-top: 1rem; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%;'>
            {content}
        </div>
    """, unsafe_allow_html=True)

def display_chart(fig: go.Figure, key: Optional[str] = None) -> None:
    """Display a Plotly chart with consistent styling.
    
    Args:
        fig: Plotly figure object
        key: Optional unique key for the chart
    """
    st.plotly_chart(fig, use_container_width=True, key=key)

def display_dataframe(df: pd.DataFrame, key: Optional[str] = None) -> None:
    """Display a pandas DataFrame with consistent styling.
    
    Args:
        df: Pandas DataFrame
        key: Optional unique key for the table
    """
    st.dataframe(df, use_container_width=True, key=key)

def display_styled_message(message: str, message_type: str = "info") -> None:
    """Display a styled message box.
    
    Args:
        message: Message text
        message_type: Type of message (info, success, warning, error)
    """
    colors = {
        "info": "#3366CC",
        "success": "#109618",
        "warning": "#FF9900",
        "error": "#DC3545"
    }
    
    color = colors.get(message_type, colors["info"])
    
    st.markdown(f"""
        <div style='
            background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
            padding: 1rem;
            border-left: 4px solid {color};
            border-radius: 0.5rem;
            color: #111;
            margin-top: 1rem;
        '>
            <p>{message}</p>
        </div>
    """, unsafe_allow_html=True)

def create_metric_columns(metrics: List[Tuple[str, str, Optional[str]]]) -> None:
    """Create a row of metric columns.
    
    Args:
        metrics: List of tuples containing (title, value, delta)
    """
    cols = st.columns(len(metrics))
    for col, (title, value, delta) in zip(cols, metrics):
        with col:
            display_metric_card(title, value, delta)

def create_tabs(tab_names: List[str]) -> List[Any]:
    """Create tabs with consistent styling.
    
    Args:
        tab_names: List of tab names
    
    Returns:
        List of tab objects
    """
    return st.tabs(tab_names)

def create_columns(column_widths: List[int]) -> List[Any]:
    """Create columns with specified relative widths.
    
    Args:
        column_widths: List of relative column widths
    
    Returns:
        List of column objects
    """
    return st.columns(column_widths)

def format_currency(value: float) -> str:
    """Format a number as currency.
    
    Args:
        value: Number to format
    
    Returns:
        Formatted currency string
    """
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a number as percentage.
    
    Args:
        value: Number to format
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.1f}%" 