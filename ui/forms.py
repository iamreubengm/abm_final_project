"""
Form input components for the application.
"""
import streamlit as st

def income_input(label, key):
    """Create a consistent income input field."""
    return st.number_input(
        label,
        min_value=0.0,
        step=100.0,
        format="%.2f",
        key=key
    )

def expense_input(label, key):
    """Create a consistent expense input field."""
    return st.number_input(
        label,
        min_value=0.0,
        step=100.0,
        format="%.2f",
        key=key
    )

def date_input(label, key):
    """Create a consistent date input field."""
    return st.date_input(
        label,
        key=key
    )

def text_input(label, key):
    """Create a consistent text input field."""
    return st.text_input(
        label,
        key=key
    )

def selectbox(label, options, key):
    """Create a consistent selectbox input field."""
    return st.selectbox(
        label,
        options,
        key=key
    )

def multiselect(label, options, key):
    """Create a consistent multiselect input field."""
    return st.multiselect(
        label,
        options,
        key=key
    ) 