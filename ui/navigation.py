"""
Navigation components for the application.
"""
import streamlit as st
from .components import create_navigation_button, display_quick_stats

def create_sidebar(components):
    """Create the sidebar with navigation and user info."""
    with st.sidebar:
        st.title("Navigation")
        
        # Navigation buttons
        create_navigation_button("📊 Dashboard", "dashboard")
        create_navigation_button("👤 Financial Profile", "profile")
        create_navigation_button("💰 Budget Manager", "budget")
        create_navigation_button("📈 Investment Planner", "investments")
        create_navigation_button("💳 Debt Manager", "debt")
        create_navigation_button("🎯 Savings Goals", "savings")
        create_navigation_button("📝 Tax Optimizer", "tax")
        create_navigation_button("💬 AI Financial Advisor", "advisor")
        
        st.markdown("---")
        
        # User quick stats
        total_income = sum(st.session_state.user_data["income"].values())
        total_expenses = sum(st.session_state.user_data["expenses"].values())
        display_quick_stats(total_income, total_expenses)
        
        st.markdown("---")
        
        # Data actions
        st.subheader("Actions")
        if st.button("Save Financial Data", use_container_width=True):
            success = components["data_loader"].save_user_data(st.session_state.user_data, user_id="user")
            if success:
                st.success("Financial data saved successfully!")
            else:
                st.error("Failed to save financial data")
        
        # Information
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI-powered financial portal helps you manage your finances, 
        get personalized advice, and make informed financial decisions.
        """) 