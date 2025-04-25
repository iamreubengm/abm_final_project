# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Import UI components
from ui.components import (
    display_header,
    display_metric_card,
    display_card,
    display_agent_response,
    create_navigation_button,
    display_quick_stats,
    display_chat_message,
    format_expert_message,
    format_consensus_message,
    display_agent_comparison,
    display_styled_agent_response,
    display_chart,
    display_dataframe,
    display_styled_message,
    create_metric_columns,
    create_tabs,
    create_columns,
    format_currency,
    format_percentage,
    display_feedback_ui
)

# Import utility functions
from utils.data_processing import (
    calculate_total_income,
    calculate_total_expenses,
    calculate_savings_rate,
    calculate_portfolio_value,
    calculate_asset_allocation,
    format_financial_data,
    calculate_debt_to_income_ratio
)

# Import view components
from ui.views import (
    render_dashboard_metrics,
    render_investment_metrics,
    render_debt_metrics,
    render_savings_metrics,
    render_profile_section,
    render_income_section,
    render_expenses_section,
    render_debt_section,
    render_investment_section,
    render_savings_section
)

# Import project modules
from agents.agent_manager import AgentManager
from utils.data_loader import DataLoader
from utils.visualization import FinancialVisualizer
from utils.rag_utils import FinancialRAG
from utils.llm_utils import LLMUtils
from utils.feedback_manager import FeedbackManager
from config import get_anthropic_client, STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, STREAMLIT_LAYOUT

# Import UI modules
from ui.styles import apply_css
from ui.navigation import create_sidebar
from ui.forms import income_input, expense_input, date_input, text_input, selectbox, multiselect

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the main components of the application."""
    client = get_anthropic_client()
    data_loader = DataLoader()
    visualizer = FinancialVisualizer()
    llm_utils = LLMUtils(client)
    
    # Initialize knowledge base
    knowledge_base = FinancialRAG(client)
    
    # Initialize agent manager with all components
    agent_manager = AgentManager(client, knowledge_base)
    
    return {
        "client": client,
        "data_loader": data_loader,
        "visualizer": visualizer,
        "llm_utils": llm_utils,
        "knowledge_base": knowledge_base,
        "agent_manager": agent_manager
    }

# Set up session state for storing data between interactions
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "user_data" not in st.session_state:
        data_loader = initialize_components()["data_loader"]
        st.session_state.user_data = data_loader.load_user_data("user")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_view" not in st.session_state:
        st.session_state.current_view = "dashboard"
    
    if "agent_outputs" not in st.session_state:
        st.session_state.agent_outputs = {}
    
    if "transaction_data" not in st.session_state:
        st.session_state.transaction_data = []
    
    if "portfolio_data" not in st.session_state:
        st.session_state.portfolio_data = {"holdings": [], "total_value": 0, "asset_allocation": {}}
    
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = ["budget", "investment", "debt", "savings", "tax"]

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)

def format_llm_output(raw_output):
    """
    Format LLM output into clean, readable HTML with proper styling.
    Handles TextBlock, list, dict, or plain string inputs.
    """
    # Extract text from TextBlock if needed
    if hasattr(raw_output, "text"):
        raw_output = raw_output.text
    elif isinstance(raw_output, list):
        raw_output = " ".join(map(str, raw_output))
    elif isinstance(raw_output, dict):
        # Extract relevant fields from dict
        for key in ["analysis", "recommendations", "content", "advice", "plan", "evaluation", "opportunities"]:
            if raw_output.get(key):
                raw_output = raw_output[key]
                break
    
    # Convert to string if not already
    text = str(raw_output)
    
    # Clean up TextBlock wrapper if present
    import re
    match = re.search(r"text=['\"](.*?)['\"](.*?)['\"]\)?$", text, re.DOTALL)
    if match:
        text = match.group(1)
    
    # Replace \n with proper HTML breaks
    text = text.replace("\\n", "<br>")
    
    # Format numbered sections
    text = re.sub(r'(\d+\.)', r'<br><strong>\1</strong>', text)
    
    # Format headers
    text = re.sub(r'([A-Za-z\s]+):(\s*\n|\s+)', r'<br><strong>\1:</strong>', text)
    
    # Add spacing between sections
    text = text.replace("\n\n", "<br><br>")
    
    return f"""
    <div style='
        background-color: #f8f9fa;
        border-left: 4px solid #3366CC;
        border-radius: 4px;
        padding: 15px;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
    '>
        {text}
    </div>
    """

def display_llm_response(content, title=None):
    """
    Display an LLM response with consistent styling and optional title.
    """
    formatted_content = format_llm_output(content)
    
    if title:
        st.markdown(f"<h4 style='color: #3366CC; margin-bottom: 10px;'>{title}</h4>", unsafe_allow_html=True)
    
    st.markdown(formatted_content, unsafe_allow_html=True)

# Main application function
def main():
    """Main application entry point."""
    # Initialize components and session state
    components = initialize_components()
    initialize_session_state()
    
    # Apply CSS styles
    apply_css()
    
    # Create sidebar
    create_sidebar(components)
    
    # Display header
    display_header("AI Personal Financial Portal")
    
    # Display current view based on navigation
    if st.session_state.current_view == "dashboard":
        show_dashboard_view(components)
    elif st.session_state.current_view == "profile":
        show_profile_view(components)
    elif st.session_state.current_view == "budget":
        show_budget_view(components)
    elif st.session_state.current_view == "investments":
        show_investments_view(components)
    elif st.session_state.current_view == "debt":
        show_debt_view(components)
    elif st.session_state.current_view == "savings":
        show_savings_view(components)
    elif st.session_state.current_view == "tax":
        show_tax_view(components)
    elif st.session_state.current_view == "advisor":
        show_advisor_view(components)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 AI Personal Financial Portal | Powered by Anthropic Claude")

def show_dashboard_view(components):
    """Show the main dashboard view with financial overview."""
    st.markdown("<h2 class='sub-header'>Financial Dashboard</h2>", unsafe_allow_html=True)
    
    # Get components
    agent_manager = components["agent_manager"]
    visualizer = components["visualizer"]
    data_loader = components["data_loader"]
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Financial Summary Section
    st.markdown("<h3>Financial Summary</h3>", unsafe_allow_html=True)
    
    # Render dashboard metrics
    render_dashboard_metrics(user_data)
    
    # Charts Section
    st.markdown("<h3>Financial Charts</h3>", unsafe_allow_html=True)
    
    # Create tabs - only three tabs
    tabs = create_tabs(["Budget Overview", "Income & Expenses", "Investments"])
    
    with tabs[0]:
        # Budget Overview tab
        st.markdown("### Budget Breakdown")
        budget_chart = visualizer.create_budget_chart(user_data)
        display_chart(budget_chart, key="dashboard_budget_chart")
        
        # Display quick stats
        col1, col2 = st.columns(2)
        with col1:
            total_income = calculate_total_income(user_data["income"])
            st.metric("Monthly Income", format_currency(total_income))
        with col2:
            total_expenses = calculate_total_expenses(user_data["expenses"])
            st.metric("Monthly Expenses", format_currency(total_expenses))
        
        # Spending Optimization
        st.markdown("### Spending Optimization")
        if st.button("Find Savings Opportunities", key="find_savings_opportunities_btn"):
            with st.spinner("Analyzing expenses for savings opportunities..."):
                budget_agent = agent_manager.get_agent("budget")
                savings_opps = budget_agent.identify_savings_opportunities(user_data["expenses"])
                st.session_state.agent_outputs["savings_opportunities"] = savings_opps
        
        # Display savings opportunities if available
        if "savings_opportunities" in st.session_state.agent_outputs:
            opps = st.session_state.agent_outputs["savings_opportunities"]
            display_llm_response(opps.get("opportunities", ""), "Savings Opportunities")
    
    with tabs[1]:
        # Income & Expenses tab
        col1, col2 = st.columns(2)
        
        with col1:
            # Income Analysis
            st.markdown("### Income Sources")
            income_data = [{"Source": source, "Amount": amount} 
                          for source, amount in user_data["income"].items()]
            income_df = pd.DataFrame(income_data)
            income_df["Percentage"] = income_df["Amount"] / income_df["Amount"].sum() * 100
            
            # Format for display
            income_display = income_df.copy()
            income_display["Amount"] = income_display["Amount"].apply(lambda x: format_currency(x))
            income_display["Percentage"] = income_display["Percentage"].apply(lambda x: format_percentage(x))
            display_dataframe(income_display, key="income_sources_table")
        
        with col2:
            # Expense Analysis
            st.markdown("### Expense Categories")
            expense_data = [{"Category": category, "Amount": amount} 
                           for category, amount in user_data["expenses"].items()]
            expense_df = pd.DataFrame(expense_data)
            expense_df["Percentage"] = expense_df["Amount"] / expense_df["Amount"].sum() * 100
            
            # Format for display
            expense_display = expense_df.copy()
            expense_display["Amount"] = expense_display["Amount"].apply(lambda x: format_currency(x))
            expense_display["Percentage"] = expense_display["Percentage"].apply(lambda x: format_percentage(x))
            display_dataframe(expense_display, key="expense_categories_table")
        
        # Income vs Expenses Trend
        st.markdown("### Income vs Expenses Trend")
        demo_monthly_data = visualizer.generate_demo_monthly_data(12)
        income_expense_chart = visualizer.create_income_expense_trend_chart(demo_monthly_data)
        display_chart(income_expense_chart, key="income_expense_trend_chart")
        
        # Expense Distribution
        st.markdown("### Expense Distribution")
        expense_pie = visualizer.create_expense_pie_chart(user_data["expenses"])
        display_chart(expense_pie, key="expense_distribution_chart")
    
    with tabs[2]:
        # Investments tab
        st.markdown("### Asset Allocation")
        if st.session_state.portfolio_data["holdings"]:
            portfolio = st.session_state.portfolio_data
        else:
            portfolio = {
                "asset_allocation": {
                    "Stocks": 60,
                    "Bonds": 25,
                    "Cash": 10,
                    "Real Estate": 5
                }
            }
        
        allocation_chart = visualizer.create_investment_allocation_chart(portfolio)
        display_chart(allocation_chart, key="asset_allocation_chart")
    
    # AI Insights Section
    st.markdown("<h3>AI Financial Insights</h3>", unsafe_allow_html=True)
    
    # Initialize feedback manager if not already done
    if "feedback_manager" not in st.session_state:
        st.session_state.feedback_manager = FeedbackManager()
    
    # Button to refresh insights
    if st.button("Generate New Insights", key="generate_new_insights_btn"):
        st.session_state.pop("insights", None)
        st.session_state.pop("insight_feedback", None)  # Clear feedback when refreshing
        st.rerun()
    
    # Create insights or load cached ones
    if "insights" not in st.session_state:
        st.session_state.insights = [
            {
                "title": "Spending Patterns",
                "description": "Your highest expense category is Housing at 40% of your income. This is within the recommended 30-40% range.",
                "type": "observation",
                "pattern": "RAG"
            },
            {
                "title": "Savings Opportunity",
                "description": "You could save an additional $120/month by reducing food delivery expenses. Consider meal planning to reduce costs.",
                "type": "recommendation",
                "pattern": "Multi-Path Plan"
            },
            {
                "title": "Investment Allocation",
                "description": "Your current investment allocation is too conservative for your age and goals. Consider increasing equity exposure.",
                "type": "warning",
                "pattern": "Voting"
            }
        ]
    
    # Display insights with feedback
    for i, insight in enumerate(st.session_state.insights):
        if insight["type"] == "observation":
            style = "info"
        elif insight["type"] == "recommendation":
            style = "success"
        elif insight["type"] == "warning":
            style = "warning"
        else:
            style = "info"
        
        # Display the insight
        getattr(st, style)(
            f"### {insight['title']}\n\n"
            f"{insight['description']}\n\n"
            f"*Generated using {insight['pattern']} pattern*"
        )
        
        # Add feedback UI for each insight
        def handle_feedback(feedback):
            if "insight_feedback" not in st.session_state:
                st.session_state.insight_feedback = {}
            st.session_state.insight_feedback[i] = feedback
            
            st.session_state.feedback_manager.record_insight_feedback(
                user_id=user_data.get("personal", {}).get("id", 0),
                insight=insight,
                feedback=feedback
            )
        
        # Only show feedback UI if feedback hasn't been submitted for this insight
        if "insight_feedback" not in st.session_state or i not in st.session_state.insight_feedback:
            display_feedback_ui(insight["description"], handle_feedback)
        
        st.markdown("---")  # Add separator between insights
    
    # Display analysis if available
    if "budget_analysis" in st.session_state.agent_outputs:
        analysis = st.session_state.agent_outputs["budget_analysis"]
        display_llm_response(analysis.get("analysis", ""), "Budget Analysis")
    
    with tabs[1]:
        # Income Analysis
        st.markdown("### Income Sources")
        
        # Create income DataFrame for display
        income_data = [{"Source": source, "Amount": amount} 
                       for source, amount in user_data["income"].items()]
        income_df = pd.DataFrame(income_data)
        
        # Calculate percentages
        income_df["Percentage"] = income_df["Amount"] / income_df["Amount"].sum() * 100
        
        # Format for display
        income_display = income_df.copy()
        income_display["Amount"] = income_display["Amount"].apply(lambda x: format_currency(x))
        income_display["Percentage"] = income_display["Percentage"].apply(lambda x: format_percentage(x))
        
        # Display table
        display_dataframe(income_display, key="income_sources_detail_table")
        
        # Income history chart
        st.markdown("### Income History")
        
        demo_monthly_data = visualizer.generate_demo_monthly_data(12)
        income_trend = go.Figure()
        income_trend.add_trace(go.Scatter(
            x=[data["date"] for data in demo_monthly_data],
            y=[data["income"] for data in demo_monthly_data],
            mode="lines+markers",
            name="Monthly Income",
            line=dict(color="#66BB6A", width=3),
            marker=dict(color="#66BB6A", size=8)
        ))
        
        income_trend.update_layout(
            title="Monthly Income Trend",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template="plotly_white"
        )
        
        display_chart(income_trend, key="income_history_trend_chart")
    
    with tabs[2]:
        # Expense Analysis
        st.markdown("### Expense Categories")
        
        # Create expense DataFrame for display
        expense_data = [{"Category": category, "Amount": amount} 
                        for category, amount in user_data["expenses"].items()]
        expense_df = pd.DataFrame(expense_data)
        
        # Calculate percentages
        expense_df["Percentage"] = expense_df["Amount"] / expense_df["Amount"].sum() * 100
        
        # Format for display
        expense_display = expense_df.copy()
        expense_display["Amount"] = expense_display["Amount"].apply(lambda x: format_currency(x))
        expense_display["Percentage"] = expense_display["Percentage"].apply(lambda x: format_percentage(x))
        
        # Display table
        display_dataframe(expense_display, key="expense_categories_detail_table")
        
        # Expense pie chart
        st.markdown("### Expense Distribution")
        expense_pie = visualizer.create_expense_pie_chart(user_data["expenses"])
        display_chart(expense_pie, key="expense_distribution_detail_chart")
        
        # Spending Optimization
        st.markdown("### Spending Optimization")
        
        if st.button("Find Savings Opportunities", key="find_savings_opportunities_detail_btn"):
            with st.spinner("Analyzing expenses for savings opportunities..."):
                # Get savings opportunities from budget agent
                budget_agent = agent_manager.get_agent("budget")
                savings_opps = budget_agent.identify_savings_opportunities(user_data["expenses"])
                
                # Store in session state
                st.session_state.agent_outputs["savings_opportunities"] = savings_opps
        
        # Display savings opportunities if available
        if "savings_opportunities" in st.session_state.agent_outputs:
            opps = st.session_state.agent_outputs["savings_opportunities"]
            display_llm_response(opps.get("opportunities", ""), "Savings Opportunities")

def show_investments_view(components):
    """Show the investment planner view."""
    st.markdown("<h2 class='sub-header'>Investment Planner</h2>", unsafe_allow_html=True)
    
    # Get components
    agent_manager = components["agent_manager"]
    visualizer = components["visualizer"]
    data_loader = components["data_loader"]
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Create tabs
    tabs = st.tabs(["Portfolio Overview", "Investment Analysis", "Asset Allocation", "Investment Goals"])
    
    with tabs[0]:
        st.markdown("### Investment Portfolio")
        
        # Display current investments
        if user_data["investments"]:
            # Calculate portfolio values
            retirement_value = sum(account.get("balance", 0) for account in user_data["investments"].get("retirement_accounts", []))
            brokerage_value = sum(account.get("balance", 0) for account in user_data["investments"].get("brokerage_accounts", []))
            real_estate_value = sum(property.get("estimated_value", 0) - property.get("mortgage_balance", 0) 
                                   for property in user_data["investments"].get("real_estate", []))
            other_value = sum(investment.get("value", 0) for investment in user_data["investments"].get("other_investments", []))
            
            total_investments = retirement_value + brokerage_value + real_estate_value + other_value
            
            # Display summary metrics
            metrics = [
                ("Retirement Accounts", format_currency(retirement_value), None),
                ("Brokerage Accounts", format_currency(brokerage_value), None),
                ("Real Estate Equity", format_currency(real_estate_value), None),
                ("Total Investments", format_currency(total_investments), None)
            ]
            create_metric_columns(metrics)
            
            # Create investment table
            inv_data = []
            
            # Add retirement accounts
            for account in user_data["investments"].get("retirement_accounts", []):
                inv_data.append({
                    "Type": "Retirement Account",
                    "Account": account.get("name", ""),
                    "Amount": format_currency(account.get("balance", 0)),
                    "Allocation": format_percentage(account.get("balance", 0) / total_investments * 100 if total_investments > 0 else 0)
                })
            
            # Add brokerage accounts
            for account in user_data["investments"].get("brokerage_accounts", []):
                inv_data.append({
                    "Type": "Brokerage Account",
                    "Account": account.get("name", ""),
                    "Amount": format_currency(account.get("balance", 0)),
                    "Allocation": format_percentage(account.get("balance", 0) / total_investments * 100 if total_investments > 0 else 0)
                })
            
            # Add real estate
            for property in user_data["investments"].get("real_estate", []):
                equity = property.get("estimated_value", 0) - property.get("mortgage_balance", 0)
                inv_data.append({
                    "Type": "Real Estate",
                    "Account": property.get("name", ""),
                    "Amount": format_currency(equity),
                    "Allocation": format_percentage(equity / total_investments * 100 if total_investments > 0 else 0)
                })
            
            # Add other investments
            for investment in user_data["investments"].get("other_investments", []):
                inv_data.append({
                    "Type": "Other Investment",
                    "Account": investment.get("name", ""),
                    "Amount": format_currency(investment.get("value", 0)),
                    "Allocation": format_percentage(investment.get("value", 0) / total_investments * 100 if total_investments > 0 else 0)
                })
            
            if inv_data:
                display_dataframe(pd.DataFrame(inv_data))
        else:
            st.info("No investment data available. Add investments in your Financial Profile.")
        
        # Portfolio analysis button
        if st.button("Analyze Portfolio"):
            with st.spinner("Analyzing your investment portfolio..."):
                try:
                    # Get investment agent
                    investment_agent = agent_manager.get_agent("investment")
                    
                    # Prepare portfolio data for analysis
                    portfolio_data = {
                        "investments": user_data["investments"],
                        "risk_tolerance": user_data["profile"].get("risk_tolerance", "moderate"),
                        "time_horizon": user_data["profile"].get("time_horizon", "medium"),
                        "age": user_data["personal"].get("age", 35),
                        "goals": [goal.get("name") for goal in user_data["profile"].get("financial_goals", [])]
                    }
                    
                    # Get portfolio analysis
                    portfolio_analysis = investment_agent.analyze_portfolio(portfolio_data)
                    
                    # Store in session state
                    st.session_state.agent_outputs["portfolio_analysis"] = portfolio_analysis
                    
                    # Display analysis immediately
                    if portfolio_analysis:
                        display_llm_response(portfolio_analysis.get("analysis", ""), "Portfolio Analysis")
                except Exception as e:
                    st.error(f"Error analyzing portfolio: {str(e)}")
                    st.info("Please make sure you have entered your investment data in the Financial Profile section.")
    
    with tabs[1]:
        # Investment Analysis
        st.markdown("### Investment Analysis")
        
        # Display portfolio analysis if available
        if "portfolio_analysis" in st.session_state.agent_outputs:
            analysis = st.session_state.agent_outputs["portfolio_analysis"]
            display_llm_response(analysis.get("analysis", ""), "Portfolio Analysis")
        
        # Investment history chart
        st.markdown("### Investment History")
        
        demo_monthly_data = visualizer.generate_demo_monthly_data(12)
        investment_trend = go.Figure()
        investment_trend.add_trace(go.Scatter(
            x=[data["date"] for data in demo_monthly_data],
            y=[data["savings"] for data in demo_monthly_data],
            mode="lines+markers",
            name="Monthly Investment",
            line=dict(color="#66BB6A", width=3),
            marker=dict(color="#66BB6A", size=8)
        ))
        
        investment_trend.update_layout(
            title="Monthly Investment Trend",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template="plotly_white"
        )
        
        display_chart(investment_trend, key="investment_history_investment_trend")
    
    with tabs[2]:
        # Asset Allocation
        st.markdown("### Current Asset Allocation")
        
        # Use portfolio data if available, otherwise aggregate from user data
        if st.session_state.portfolio_data.get("holdings"):
            portfolio = st.session_state.portfolio_data
            allocation_data = portfolio["asset_allocation"]
        else:
            # Aggregate asset allocation from user data
            allocation_data = calculate_asset_allocation(user_data["investments"])
        
        # Create allocation chart
        if allocation_data:
            # Create pie chart
            portfolio_for_chart = {"asset_allocation": allocation_data}
            allocation_chart = visualizer.create_investment_allocation_chart(portfolio_for_chart)
            display_chart(allocation_chart, key="asset_allocation_chart")
            
            # Display allocation table
            allocation_df = pd.DataFrame([
                {"Asset Class": asset_class, "Percentage": format_percentage(percentage)}
                for asset_class, percentage in allocation_data.items()
            ])
            display_dataframe(allocation_df)
        else:
            st.info("No asset allocation data available. Please enter your investments in the Financial Profile section.")
        
        # Target allocation section
        st.markdown("### Target Asset Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_profile = user_data["profile"].get("risk_tolerance", "moderate")
            st.info(f"Current Risk Profile: {risk_profile.capitalize()}")
        
        with col2:
            time_horizon = user_data["profile"].get("time_horizon", "medium")
            st.info(f"Current Time Horizon: {time_horizon.capitalize()}")
        
        # Get target allocation recommendation
        if st.button("Get Recommended Asset Allocation"):
            with st.spinner("Generating asset allocation recommendation..."):
                # Get recommendation from investment agent
                investment_agent = agent_manager.get_agent("investment")
                
                investment_criteria = {
                    "risk_tolerance": risk_profile,
                    "time_horizon": time_horizon,
                    "age": user_data["personal"].get("age", 35),
                    "goals": [goal.get("name") for goal in user_data["profile"].get("financial_goals", [])]
                }
                
                allocation_recommendation = investment_agent.recommend_investments(user_data, investment_criteria)
                
                # Store in session state
                st.session_state.agent_outputs["allocation_recommendation"] = allocation_recommendation
        
        # Display recommendation if available
        if "allocation_recommendation" in st.session_state.agent_outputs:
            recommendation = st.session_state.agent_outputs["allocation_recommendation"]
            display_llm_response(recommendation[0].get("recommendations", ""), "Recommended Asset Allocation")
    
    with tabs[3]:
        # Investment Goals
        st.markdown("### Investment Goals")
        
        # Investment goals input
        st.markdown("#### Investment Goals")
        investment_goals = st.text_area(
            "Enter your investment goals (one per line)",
            value="Retirement: Save for a comfortable retirement\nEducation: Save for future education\nEmergency Fund: Build a 3-6 months of expenses fund",
            height=100
        )
        
        # Split into list
        goal_list = [goal.strip() for goal in investment_goals.split("\n") if goal.strip()]
        
        if st.button("Create Investment Plan"):
            with st.spinner("Creating personalized investment plan..."):
                # Get investment plan from investment agent
                investment_agent = agent_manager.get_agent("investment")
                investment_plan = investment_agent.create_investment_plan(user_data, goal_list)
                
                # Store in session state
                st.session_state.agent_outputs["investment_plan"] = investment_plan
        
        # Display investment plan if available
        if "investment_plan" in st.session_state.agent_outputs:
            plan = st.session_state.agent_outputs["investment_plan"]
            display_llm_response(plan.get("investment_plan", ""), "Personalized Investment Plan")
        
        # Investment templates
        st.markdown("#### Investment Templates")
        
        template_options = ["Growth-Oriented Portfolio", "Income-Focused Portfolio", "Balanced Portfolio", "Defensive Portfolio"]
        selected_template = st.selectbox("Select an investment template", template_options)
        
        if st.button("Get Template Details"):
            with st.spinner("Loading template details..."):
                # Get template details from knowledge base or LLM
                llm_utils = components["llm_utils"]
                template_details = llm_utils.generate_financial_explanation(
                    f"{selected_template} investment strategy", "intermediate")
                
                # Store in session state
                st.session_state.agent_outputs["investment_template"] = {
                    "name": selected_template,
                    "details": template_details
                }
        
        # Display template details if available
        if "investment_template" in st.session_state.agent_outputs:
            template = st.session_state.agent_outputs["investment_template"]
            
            if template["name"] == selected_template:
                display_llm_response(template["details"], template["name"])

def show_debt_view(components):
    """Show the debt manager view."""
    st.markdown("<h2 class='sub-header'>Debt Manager</h2>", unsafe_allow_html=True)
    
    # Get components
    agent_manager = components["agent_manager"]
    visualizer = components["visualizer"]
    data_loader = components["data_loader"]
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Create tabs
    tabs = st.tabs(["Debt Overview", "Repayment Strategies", "Loan Comparison", "Credit Score"])
    
    with tabs[0]:
        # Debt Overview
        st.markdown("### Debt Summary")
        
        # Get all debts
        all_debts = []
        
        # Credit Cards
        for card in user_data["debts"].get("credit_cards", []):
            all_debts.append({
                "type": "Credit Card",
                "name": card.get("name", "Credit Card"),
                "balance": card.get("balance", 0),
                "interest_rate": card.get("interest_rate", 0),
                "minimum_payment": card.get("minimum_payment", 0)
            })
        
        # Student Loans
        for loan in user_data["debts"].get("student_loans", []):
            all_debts.append({
                "type": "Student Loan",
                "name": loan.get("name", "Student Loan"),
                "balance": loan.get("balance", 0),
                "interest_rate": loan.get("interest_rate", 0),
                "minimum_payment": loan.get("minimum_payment", 0)
            })
        
        # Mortgage
        for mortgage in user_data["debts"].get("mortgage", []):
            all_debts.append({
                "type": "Mortgage",
                "name": mortgage.get("name", "Mortgage"),
                "balance": mortgage.get("balance", 0),
                "interest_rate": mortgage.get("interest_rate", 0),
                "minimum_payment": mortgage.get("minimum_payment", 0)
            })
        
        # Calculate summary metrics
        total_debt = sum(debt["balance"] for debt in all_debts)
        weighted_rate = sum(debt["balance"] * debt["interest_rate"] for debt in all_debts) / total_debt if total_debt > 0 else 0
        monthly_payments = sum(debt["minimum_payment"] for debt in all_debts)
        
        # Display summary metrics
        metrics = [
            ("Total Debt", format_currency(total_debt), None),
            ("Weighted Avg Interest", format_percentage(weighted_rate), None),
            ("Monthly Payments", format_currency(monthly_payments), None)
        ]
        create_metric_columns(metrics)
        
        # Display debt breakdown
        st.markdown("### Debt Breakdown")
        
        if all_debts:
            # Create DataFrame for display
            debt_df = pd.DataFrame(all_debts)
            
            # Add debt-to-income ratio
            monthly_income = calculate_total_income(user_data["income"])
            dti_ratio = calculate_debt_to_income_ratio(monthly_payments, monthly_income)
            
            # Format for display
            display_df = debt_df.copy()
            display_df["balance"] = display_df["balance"].apply(lambda x: format_currency(x))
            display_df["interest_rate"] = display_df["interest_rate"].apply(lambda x: format_percentage(x))
            display_df["minimum_payment"] = display_df["minimum_payment"].apply(lambda x: format_currency(x))
            
            # Display table
            display_dataframe(display_df)
            
            # Debt-to-income info
            st.info(f"Debt-to-Income Ratio: {format_percentage(dti_ratio)} (Monthly Debt Payments / Monthly Income)")
            
            # Display debt analysis button
            if st.button("Analyze Debt Profile"):
                with st.spinner("Analyzing your debt profile..."):
                    # Get debt analysis from debt agent
                    debt_agent = agent_manager.get_agent("debt")
                    debt_analysis = debt_agent.analyze_debt_profile(all_debts)
                    
                    # Store in session state
                    st.session_state.agent_outputs["debt_analysis"] = debt_analysis
            
            # Display analysis if available
            if "debt_analysis" in st.session_state.agent_outputs:
                analysis = st.session_state.agent_outputs["debt_analysis"]
                display_llm_response(analysis.get("analysis", ""), "Debt Profile Analysis")
        else:
            st.info("No debt information available. Please enter your debts in the Financial Profile section.")
    
    with tabs[1]:
        # Repayment Strategies
        st.markdown("### Debt Repayment Strategies")
        
        # Strategy selection
        strategy_options = ["Debt Avalanche (Highest Interest First)", "Debt Snowball (Smallest Balance First)"]
        selected_strategy = st.radio("Select a repayment strategy", strategy_options)
        
        # Map to strategy code
        strategy_code = "avalanche" if "Avalanche" in selected_strategy else "snowball"
        
        # Extra payment amount
        extra_payment = st.slider("Additional monthly payment", min_value=0, max_value=1000, value=200, step=50)
        
        # Create repayment plan
        if st.button("Create Repayment Plan"):
            with st.spinner("Generating debt repayment plan..."):
                # Add extra payment info to user data for context
                user_data_with_extra = user_data.copy()
                user_data_with_extra["extra_debt_payment"] = extra_payment
                
                # Get repayment plan from debt agent
                debt_agent = agent_manager.get_agent("debt")
                repayment_plan = debt_agent.create_debt_repayment_plan(user_data_with_extra, strategy_code)
                
                # Store in session state
                st.session_state.agent_outputs["repayment_plan"] = repayment_plan
                
                # Generate debt projection for chart
                if all_debts:
                    debt_projections = visualizer.generate_demo_debt_projections(
                        months=48,
                        initial_debt=total_debt,
                        monthly_payment=monthly_payments + extra_payment,
                        interest_rate=weighted_rate / 100
                    )
                    
                    # Store projections
                    st.session_state.agent_outputs["debt_projections"] = debt_projections
        
        # Display repayment plan if available
        if "repayment_plan" in st.session_state.agent_outputs:
            plan = st.session_state.agent_outputs["repayment_plan"]
            display_llm_response(plan.get("repayment_plan", ""), f"{plan['strategy'].capitalize()} Repayment Plan")
            
            # Display projection chart if available
            if "debt_projections" in st.session_state.agent_outputs:
                projections = st.session_state.agent_outputs["debt_projections"]
                
                st.markdown("### Debt Payoff Projection")
                projection_chart = visualizer.create_debt_payoff_chart(projections)
                display_chart(projection_chart, key="debt_payoff_projection_chart")
    
    with tabs[2]:
        # Loan Comparison
        st.markdown("### Loan Comparison Tool")
        
        st.markdown("#### Compare Loan Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, value=25000, step=1000)
            loan_term_years = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=5, step=1)
        
        with col2:
            interest_rate_1 = st.number_input("Option 1 Interest Rate (%)", min_value=0.1, max_value=30.0, value=5.99, step=0.1)
            interest_rate_2 = st.number_input("Option 2 Interest Rate (%)", min_value=0.1, max_value=30.0, value=4.49, step=0.1)
        
        # Optional fees
        with st.expander("Additional Fees (Optional)"):
            col1, col2 = st.columns(2)
            
            with col1:
                origination_fee_1 = st.number_input("Option 1 Origination Fee", min_value=0, max_value=5000, value=0, step=100)
                closing_costs_1 = st.number_input("Option 1 Closing Costs", min_value=0, max_value=10000, value=0, step=100)
            
            with col2:
                origination_fee_2 = st.number_input("Option 2 Origination Fee", min_value=0, max_value=5000, value=500, step=100)
                closing_costs_2 = st.number_input("Option 2 Closing Costs", min_value=0, max_value=10000, value=0, step=100)
        
        # Compare loans
        if st.button("Compare Loans"):
            with st.spinner("Comparing loan options..."):
                # Create loan details
                loan_1 = {
                    "type": "Loan Option 1",
                    "amount": loan_amount,
                    "interest_rate": interest_rate_1,
                    "term_years": loan_term_years,
                    "origination_fee": origination_fee_1,
                    "closing_costs": closing_costs_1
                }
                
                loan_2 = {
                    "type": "Loan Option 2",
                    "amount": loan_amount,
                    "interest_rate": interest_rate_2,
                    "term_years": loan_term_years,
                    "origination_fee": origination_fee_2,
                    "closing_costs": closing_costs_2
                }
                
                # Calculate basic metrics
                monthly_payment_1 = loan_amount * (interest_rate_1/100/12) * (1 + (interest_rate_1/100/12))**(loan_term_years*12) / ((1 + (interest_rate_1/100/12))**(loan_term_years*12) - 1)
                monthly_payment_2 = loan_amount * (interest_rate_2/100/12) * (1 + (interest_rate_2/100/12))**(loan_term_years*12) / ((1 + (interest_rate_2/100/12))**(loan_term_years*12) - 1)
                
                total_interest_1 = monthly_payment_1 * loan_term_years * 12 - loan_amount
                total_interest_2 = monthly_payment_2 * loan_term_years * 12 - loan_amount
                
                total_cost_1 = total_interest_1 + loan_amount + origination_fee_1 + closing_costs_1
                total_cost_2 = total_interest_2 + loan_amount + origination_fee_2 + closing_costs_2
                
                # Store comparison results
                comparison_results = {
                    "loan_1": {
                        "monthly_payment": monthly_payment_1,
                        "total_interest": total_interest_1,
                        "total_cost": total_cost_1,
                        "details": loan_1
                    },
                    "loan_2": {
                        "monthly_payment": monthly_payment_2,
                        "total_interest": total_interest_2,
                        "total_cost": total_cost_2,
                        "details": loan_2
                    }
                }
                
                st.session_state.agent_outputs["loan_comparison"] = comparison_results
                
                # Get loan evaluation from debt agent
                debt_agent = agent_manager.get_agent("debt")
                loan_evaluation = debt_agent.evaluate_loan_option(loan_1 if total_cost_1 <= total_cost_2 else loan_2, user_data)
                
                # Store evaluation
                st.session_state.agent_outputs["loan_evaluation"] = loan_evaluation
        
        # Display comparison if available
        if "loan_comparison" in st.session_state.agent_outputs:
            comparison = st.session_state.agent_outputs["loan_comparison"]
            
            st.markdown("### Loan Comparison Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Option 1 ({comparison['loan_1']['details']['interest_rate']}%)")
                st.metric("Monthly Payment", format_currency(comparison['loan_1']['monthly_payment']))
                st.metric("Total Interest", format_currency(comparison['loan_1']['total_interest']))
                st.metric("Total Cost", format_currency(comparison['loan_1']['total_cost']))
                
                if comparison['loan_1']['details']['origination_fee'] > 0:
                    st.text(f"Origination Fee: {format_currency(comparison['loan_1']['details']['origination_fee'])}")
                
                if comparison['loan_1']['details']['closing_costs'] > 0:
                    st.text(f"Closing Costs: ${comparison['loan_1']['details']['closing_costs']:,.2f}")
            
            with col2:
                st.markdown(f"#### Option 2 ({comparison['loan_2']['details']['interest_rate']}%)")
                st.metric("Monthly Payment", f"${comparison['loan_2']['monthly_payment']:,.2f}")
                st.metric("Total Interest", f"${comparison['loan_2']['total_interest']:,.2f}")
                st.metric("Total Cost", f"${comparison['loan_2']['total_cost']:,.2f}")
                
                if comparison['loan_2']['details']['origination_fee'] > 0:
                    st.text(f"Origination Fee: ${comparison['loan_2']['details']['origination_fee']:,.2f}")
                
                if comparison['loan_2']['details']['closing_costs'] > 0:
                    st.text(f"Closing Costs: ${comparison['loan_2']['details']['closing_costs']:,.2f}")
            
            # Savings comparison
            savings = abs(comparison['loan_1']['total_cost'] - comparison['loan_2']['total_cost'])
            better_option = "Option 1" if comparison['loan_1']['total_cost'] < comparison['loan_2']['total_cost'] else "Option 2"
            
            st.success(f"{better_option} saves you ${savings:,.2f} over the life of the loan.")
            
            # Display loan evaluation if available
            if "loan_evaluation" in st.session_state.agent_outputs:
                evaluation = st.session_state.agent_outputs["loan_evaluation"]
                display_llm_response(evaluation, "Loan Evaluation")

    with tabs[3]:
        # Credit Score
        st.markdown("### Credit Score Management")
        
        # Display current credit score
        credit_score = user_data.get("credit_score", 0)
        
        # Credit score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Score"},
            gauge={
                'axis': {'range': [300, 850], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [300, 580], 'color': 'firebrick'},
                    {'range': [580, 670], 'color': 'darkorange'},
                    {'range': [670, 740], 'color': 'gold'},
                    {'range': [740, 800], 'color': 'yellowgreen'},
                    {'range': [800, 850], 'color': 'forestgreen'}
                ],
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True, key="credit_score_fig")
        
        # Credit score rating
        rating = "Poor"
        if credit_score >= 800:
            rating = "Excellent"
        elif credit_score >= 740:
            rating = "Very Good"
        elif credit_score >= 670:
            rating = "Good"
        elif credit_score >= 580:
            rating = "Fair"
        
        st.info(f"Your credit score of {credit_score} is considered {rating}")
        
        # Credit score history
        st.markdown("### Credit Score History")
        
        # Generate demo credit history
        credit_history = visualizer.generate_demo_credit_history(24, credit_score - 40)
        
        # Create chart
        credit_chart = visualizer.create_credit_score_chart(credit_history)
        st.plotly_chart(credit_chart, use_container_width=True, key="credit_history_chart")
        
        # Credit score improvement
        st.markdown("### Credit Score Improvement")
        
        if st.button("Get Credit Improvement Tips"):
            with st.spinner("Analyzing your credit profile..."):
                # Create simple credit report
                credit_report = {
                    "score": credit_score,
                    "rating": rating,
                    "factors": {
                        "payment_history": "Good",
                        "credit_utilization": "Fair",
                        "credit_age": "Good",
                        "account_mix": "Good",
                        "recent_inquiries": "Fair"
                    },
                    "accounts": {
                        "credit_cards": len(user_data["debts"].get("credit_cards", [])),
                        "installment_loans": len(user_data["debts"].get("student_loans", [])) + 
                                            len(user_data["debts"].get("auto_loans", [])) +
                                            len(user_data["debts"].get("personal_loans", [])),
                        "mortgage": len(user_data["debts"].get("mortgage", []))
                    }
                }
                
                # Get credit analysis from debt agent
                debt_agent = agent_manager.get_agent("debt")
                credit_analysis = debt_agent.analyze_credit_score(credit_report)
                
                # Store in session state
                st.session_state.agent_outputs["credit_analysis"] = credit_analysis
        
        # Display credit analysis if available
        if "credit_analysis" in st.session_state.agent_outputs:
            analysis = st.session_state.agent_outputs["credit_analysis"]
            display_llm_response(analysis.get("analysis", ""), "Credit Score Improvement Recommendations")


def show_savings_view(components):
    """Show the savings goals view."""
    st.markdown("<h2 class='sub-header'>Savings Goals</h2>", unsafe_allow_html=True)
    
    # Get components
    agent_manager = components["agent_manager"]
    visualizer = components["visualizer"]
    data_loader = components["data_loader"]
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Create tabs
    tabs = st.tabs(["Goals Overview", "Emergency Fund", "Goal Planning", "Savings Strategies"])
    
    with tabs[0]:
        # Goals Overview
        st.markdown("### Savings Goals Progress")
        
        # Get savings goals
        savings_goals = user_data["savings"].get("savings_goals", [])
        
        if savings_goals:
            # Create progress chart
            goals_chart = visualizer.create_savings_goal_progress_chart(savings_goals)
            st.plotly_chart(goals_chart, use_container_width=True, key="goals_chart")
            
            # Display goals in table
            goals_df = pd.DataFrame(savings_goals)
            
            # Add progress percentage
            if "target" in goals_df.columns and "current" in goals_df.columns:
                goals_df["progress"] = (goals_df["current"] / goals_df["target"] * 100).apply(lambda x: f"{x:.1f}%")
            
            # Add remaining amount
            if "target" in goals_df.columns and "current" in goals_df.columns:
                goals_df["remaining"] = goals_df["target"] - goals_df["current"]
            
            # Format for display
            display_df = goals_df.copy()
            if "target" in display_df.columns:
                display_df["target"] = display_df["target"].apply(lambda x: f"${x:,.2f}")
            if "current" in display_df.columns:
                display_df["current"] = display_df["current"].apply(lambda x: f"${x:,.2f}")
            if "remaining" in display_df.columns:
                display_df["remaining"] = display_df["remaining"].apply(lambda x: f"${x:,.2f}")
            
            # Display table
            st.dataframe(display_df, use_container_width=True)
            
            # Goal prioritization
            if len(savings_goals) > 1 and st.button("Prioritize Savings Goals"):
                with st.spinner("Analyzing and prioritizing your savings goals..."):
                    # Get prioritization from savings agent
                    savings_agent = agent_manager.get_agent("savings")
                    prioritization = savings_agent.prioritize_savings_goals(savings_goals, user_data)
                    
                    # Store in session state
                    st.session_state.agent_outputs["goal_prioritization"] = prioritization
            
            # Display prioritization if available
            if "goal_prioritization" in st.session_state.agent_outputs:
                prioritization = st.session_state.agent_outputs["goal_prioritization"]
                display_llm_response(prioritization.get("prioritized_plan", ""), "Goal Prioritization Recommendation")

        else:
            st.info("No savings goals found. Add goals in the Financial Profile section or create a new goal below.")
        
        # Quick goal creation
        st.markdown("### Add New Goal")
        
        with st.form("add_goal_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_goal_name = st.text_input("Goal Name")
                new_goal_target = st.number_input("Target Amount", min_value=100.0, step=100.0)
            
            with col2:
                new_goal_current = st.number_input("Current Amount", min_value=0.0, step=100.0)
                new_goal_deadline = st.date_input("Target Date")
            
            submit_button = st.form_submit_button("Add Goal")
            
            if submit_button and new_goal_name:
                if "savings_goals" not in user_data["savings"]:
                    user_data["savings"]["savings_goals"] = []
                
                user_data["savings"]["savings_goals"].append({
                    "name": new_goal_name,
                    "target": new_goal_target,
                    "current": new_goal_current,
                    "deadline": new_goal_deadline.strftime("%Y-%m-%d")
                })
                
                st.success(f"Added new goal: {new_goal_name}")
                st.rerun()
    
    with tabs[1]:
        # Emergency Fund
        st.markdown("### Emergency Fund")
        
        # Get emergency fund data
        emergency_fund = user_data["savings"].get("emergency_fund", {"balance": 0, "target": 0})
        
        # Calculate metrics
        monthly_expenses = sum(user_data["expenses"].values())
        current_months_covered = emergency_fund["balance"] / monthly_expenses if monthly_expenses > 0 else 0
        target_months = emergency_fund["target"] / monthly_expenses if monthly_expenses > 0 else 0
        progress_percentage = emergency_fund["balance"] / emergency_fund["target"] * 100 if emergency_fund["target"] > 0 else 0
        
        # Display emergency fund metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Balance", f"${emergency_fund['balance']:,.2f}")
        
        with col2:
            st.metric("Target Amount", f"${emergency_fund['target']:,.2f}")
        
        with col3:
            st.metric("Progress", f"{progress_percentage:.1f}%")
        
        # Progress bar
        st.progress(min(progress_percentage / 100, 1.0))
        
        # Months of expenses covered
        st.info(f"Your emergency fund covers {current_months_covered:.1f} months of expenses (target: {target_months:.1f} months)")
        
        # Emergency fund optimization
        if st.button("Optimize Emergency Fund"):
            with st.spinner("Analyzing your emergency fund..."):
                # Get optimization from savings agent
                savings_agent = agent_manager.get_agent("savings")
                ef_optimization = savings_agent.optimize_emergency_fund(user_data)
                
                # Store in session state
                st.session_state.agent_outputs["ef_optimization"] = ef_optimization
        
        # Display optimization if available
        if "ef_optimization" in st.session_state.agent_outputs:
            optimization = st.session_state.agent_outputs["ef_optimization"]
            display_llm_response(optimization.get("recommendations", ""), "Emergency Fund Recommendations")

    
    with tabs[2]:
        # Goal Planning
        st.markdown("### Savings Goal Planner")
        
        # Goal planning form
        with st.form("goal_planner_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                goal_name = st.text_input("Goal Name", "Down Payment")
                goal_amount = st.number_input("Target Amount", min_value=1000.0, value=20000.0, step=1000.0)
            
            with col2:
                current_amount = st.number_input("Current Amount", min_value=0.0, value=5000.0, step=500.0)
                time_frame = st.selectbox("Time Frame", ["6 months", "1 year", "2 years", "3 years", "5 years", "10 years"])
            
            monthly_contribution = st.slider("Monthly Contribution", min_value=100, max_value=2000, value=500, step=50)
            
            plan_button = st.form_submit_button("Create Savings Plan")
            
            if plan_button:
                with st.spinner("Creating your savings plan..."):
                    # Map time frame to months
                    time_map = {
                        "6 months": 6, 
                        "1 year": 12, 
                        "2 years": 24, 
                        "3 years": 36,
                        "5 years": 60,
                        "10 years": 120
                    }
                    months = time_map.get(time_frame, 12)
                    
                    # Create goal dict
                    goal_dict = {
                        "name": goal_name,
                        "target": goal_amount,
                        "current": current_amount,
                        "deadline": (datetime.now() + timedelta(days=30*months)).strftime("%Y-%m-%d"),
                        "monthly_contribution": monthly_contribution
                    }
                    
                    # Create savings plan
                    savings_agent = agent_manager.get_agent("savings")
                    savings_plan = savings_agent.create_savings_plan(user_data, goal_dict)
                    
                    # Store in session state
                    st.session_state.agent_outputs["savings_plan"] = savings_plan
        
        # Display savings plan if available
        if "savings_plan" in st.session_state.agent_outputs:
            plan = st.session_state.agent_outputs["savings_plan"]
            display_llm_response(plan.get("savings_plan", ""), "Savings Plan")
        
        # Simple savings calculator
        st.markdown("### Savings Calculator")
        
        with st.expander("Simple Savings Calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                calc_initial = st.number_input("Initial Amount", min_value=0.0, value=1000.0, step=500.0, key="calc_initial")
                calc_monthly = st.number_input("Monthly Contribution", min_value=0.0, value=200.0, step=50.0, key="calc_monthly")
            
            with col2:
                calc_rate = st.number_input("Annual Interest Rate (%)", min_value=0.1, max_value=10.0, value=3.0, step=0.25, key="calc_rate")
                calc_years = st.number_input("Years", min_value=1, max_value=40, value=5, step=1, key="calc_years")
            
            if st.button("Calculate Future Value"):
                # Simple compound interest calculation
                rate = calc_rate / 100
                months = calc_years * 12
                
                # Future value with monthly contributions
                future_value = calc_initial * (1 + rate/12) ** months
                future_contributions = calc_monthly * ((1 + rate/12) ** months - 1) / (rate/12)
                total_future_value = future_value + future_contributions
                
                st.success(f"Future Value: ${total_future_value:,.2f}")
                st.info(f"Initial investment: ${future_value:,.2f}, Contributions: ${future_contributions:,.2f}")
    
    with tabs[3]:
        # Savings Strategies
        st.markdown("### Savings Strategies")
        
        # Analyze savings potential
        if st.button("Analyze Savings Potential"):
            with st.spinner("Analyzing your savings potential..."):
                # Get savings potential from savings agent
                savings_agent = agent_manager.get_agent("savings")
                savings_potential = savings_agent.analyze_savings_potential(user_data["income"], user_data["expenses"])
                
                # Store in session state
                st.session_state.agent_outputs["savings_potential"] = savings_potential
        
        # Display savings potential if available
        if "savings_potential" in st.session_state.agent_outputs:
            potential = st.session_state.agent_outputs["savings_potential"]
            display_llm_response(potential.get("analysis", ""), "Savings Potential Analysis")

            
            # Display current savings metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Monthly Savings", f"${potential['current_monthly_savings']:,.2f}")
            
            with col2:
                savings_rate = (potential['current_monthly_savings'] / potential['total_income']) * 100 if potential['total_income'] > 0 else 0
                st.metric("Current Savings Rate", f"{savings_rate:.1f}%")
            
            with col3:
                st.metric("Annual Savings Potential", f"${potential['annual_savings_potential']:,.2f}")
        
        # Savings strategies explorer
        st.markdown("### Savings Strategy Explorer")
        
        savings_goals = ["Emergency Fund", "Home Down Payment", "Retirement", "Education", "Car Purchase", "Wedding", "Vacation"]
        selected_goal = st.selectbox("Select a savings goal", savings_goals)
        
        if st.button("Generate Strategies"):
            with st.spinner(f"Generating strategies for {selected_goal}..."):
                # Get strategies from savings agent
                savings_agent = agent_manager.get_agent("savings")
                
                # Generate multiple strategies (3 options)
                strategies = savings_agent.generate_strategies(user_data, selected_goal, 3)
                
                # Store in session state
                st.session_state.agent_outputs["savings_strategies"] = {
                    "goal": selected_goal,
                    "strategies": strategies
                }
        
        # Display strategies if available
        if "savings_strategies" in st.session_state.agent_outputs:
            strategy_data = st.session_state.agent_outputs["savings_strategies"]
            if strategy_data["goal"] == selected_goal:
                st.markdown(f"### Strategies for {strategy_data['goal']}")
                for i, strategy in enumerate(strategy_data["strategies"]):
                    with st.expander(f"Strategy {i+1}: {strategy.get('name', f'Option {i+1}')}"):
                        display_llm_response(strategy, f"Strategy {i+1}")

def show_tax_view(components):
    """Show the tax optimizer view."""
    st.markdown("<h2 class='sub-header'>Tax Optimizer</h2>", unsafe_allow_html=True)
    
    # Get components
    agent_manager = components["agent_manager"]
    visualizer = components["visualizer"]
    data_loader = components["data_loader"]
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Create tabs
    tabs = st.tabs(["Tax Summary", "Tax Planning", "Tax-Advantaged Accounts", "Tax Implications"])
    
    with tabs[0]:
        # Tax Summary
        st.markdown("### Tax Situation Summary")
        
        # Basic tax information
        filing_status = user_data["personal"].get("filing_status", "single")
        dependents = user_data["personal"].get("dependents", 0)
        
        # Income information
        income = user_data["income"]
        total_income = sum(income.values())
        
        # Display basic tax info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"Filing Status: {filing_status.replace('_', ' ').title()}")
        
        with col2:
            st.info(f"Dependents: {dependents}")
        
        with col3:
            st.info(f"Annual Income: ${total_income * 12:,.2f}")
        
        # Tax analysis button
        if st.button("Analyze Tax Situation"):
            with st.spinner("Analyzing your tax situation..."):
                # Get tax analysis from tax agent
                tax_agent = agent_manager.get_agent("tax")
                tax_analysis = tax_agent.analyze_tax_situation(user_data)
                
                # Store in session state
                st.session_state.agent_outputs["tax_analysis"] = tax_analysis
        
        # Display tax analysis if available
        if "tax_analysis" in st.session_state.agent_outputs:
            analysis = st.session_state.agent_outputs["tax_analysis"]
            display_llm_response(analysis.get("analysis", ""), "Tax Situation Analysis")
    
    with tabs[1]:
        # Tax Planning
        st.markdown("### Tax Planning Strategies")
        
        # Year selection
        tax_year = st.selectbox("Tax Year", [2024, 2025])
        
        # Tax planning area selection
        tax_areas = [
            "Income Tax Reduction", 
            "Investment Tax Optimization", 
            "Retirement Tax Planning",
            "Business Tax Strategies",
            "Real Estate Tax Considerations",
            "Charitable Giving",
            "Education Tax Benefits"
        ]
        
        selected_area = st.selectbox("Select Tax Planning Area", tax_areas)
        
        if st.button("Get Tax Planning Strategies"):
            with st.spinner(f"Generating {selected_area} strategies for {tax_year}..."):
                # Get strategies from tax agent
                tax_agent = agent_manager.get_agent("tax")
                
                # Create planning query
                planning_query = f"Provide {selected_area} strategies for tax year {tax_year} for someone with my financial profile."
                
                # Get advice
                tax_planning = tax_agent.get_advice(user_data, planning_query)
                
                # Store in session state
                st.session_state.agent_outputs["tax_planning"] = {
                    "area": selected_area,
                    "year": tax_year,
                    "strategies": tax_planning
                }
        
        # Display tax planning if available
        if "tax_planning" in st.session_state.agent_outputs:
            planning = st.session_state.agent_outputs["tax_planning"]
            if planning["area"] == selected_area and planning["year"] == tax_year:
                display_llm_response(planning["strategies"], f"{planning['area']} Strategies for {planning['year']}")
    
    with tabs[2]:
        # Tax-Advantaged Accounts
        st.markdown("### Tax-Advantaged Accounts")
        
        if st.button("Get Account Recommendations"):
            with st.spinner("Analyzing your situation for tax-advantaged accounts..."):
                # Get recommendations from tax agent
                tax_agent = agent_manager.get_agent("tax")
                account_recommendations = tax_agent.recommend_tax_advantaged_accounts(user_data)
                
                # Store in session state
                st.session_state.agent_outputs["account_recommendations"] = account_recommendations
        
        # Display account recommendations if available
        if "account_recommendations" in st.session_state.agent_outputs:
            recommendations = st.session_state.agent_outputs["account_recommendations"]
            display_llm_response(recommendations.get("recommendations", ""), "Tax-Advantaged Account Recommendations")

        # Account comparison
        st.markdown("### Account Type Comparison")
        
        account_types = [
            "Traditional 401(k) vs. Roth 401(k)",
            "Traditional IRA vs. Roth IRA",
            "HSA vs. FSA",
            "529 Plan vs. Coverdell ESA"
        ]
        
        selected_comparison = st.selectbox("Compare Account Types", account_types)
        
        if st.button("Compare"):
            with st.spinner(f"Comparing {selected_comparison}..."):
                # Get explanation from tax agent
                tax_agent = agent_manager.get_agent("tax")
                
                # Create comparison query
                comparison_query = f"Compare {selected_comparison} in detail, including tax benefits, contribution limits, withdrawal rules, and which is better for different scenarios."
                
                # Get advice
                comparison = tax_agent.get_advice(user_data, comparison_query)
                
                # Store in session state
                st.session_state.agent_outputs["account_comparison"] = {
                    "comparison": selected_comparison,
                    "explanation": comparison
                }
        
        # Display comparison if available
        if "account_comparison" in st.session_state.agent_outputs:
            comparison = st.session_state.agent_outputs["account_comparison"]
            display_llm_response(comparison.get("explanation", ""), "Account Type Comparison")

    
    with tabs[3]:
        # Tax Implications
        st.markdown("### Financial Decision Tax Implications")
        
        # Decision type selection
        decision_types = [
            "Selling Investments",
            "Home Purchase",
            "Retirement Withdrawal",
            "Starting a Business",
            "Relocation",
            "Inheritance",
            "Education Funding"
        ]
        
        selected_decision = st.selectbox("Financial Decision Type", decision_types)
        
        # Decision details
        decision_details = st.text_area("Decision Details", 
                                        placeholder="Describe your planned financial decision in detail...")
        
        if st.button("Analyze Tax Implications") and decision_details:
            with st.spinner(f"Analyzing tax implications of {selected_decision}..."):
                # Create decision dict
                financial_decision = {
                    "type": selected_decision,
                    "details": decision_details
                }
                
                # Get tax implications from tax agent
                tax_agent = agent_manager.get_agent("tax")
                tax_implications = tax_agent.analyze_tax_implications(financial_decision, user_data)
                
                # Store in session state
                st.session_state.agent_outputs["tax_implications"] = tax_implications
        
        # Display tax implications if available
        if "tax_implications" in st.session_state.agent_outputs:
            implications = st.session_state.agent_outputs["tax_implications"]
            display_llm_response(implications.get("tax_analysis", ""), "Tax Implications Analysis")

        
        # Tax calendar
        st.markdown("### Tax Calendar")
        
        # Create simple tax calendar
        tax_events = [
            {"date": "January 15", "event": "Q4 Estimated Tax Payment Due"},
            {"date": "April 15", "event": "Tax Filing Deadline & Q1 Estimated Tax Payment Due"},
            {"date": "June 15", "event": "Q2 Estimated Tax Payment Due"},
            {"date": "September 15", "event": "Q3 Estimated Tax Payment Due"},
            {"date": "October 15", "event": "Extended Tax Filing Deadline"},
            {"date": "December 31", "event": "Last Day for Tax-Year Transactions"}
        ]
        
        # Display calendar
        tax_calendar_df = pd.DataFrame(tax_events)
        st.table(tax_calendar_df)

def format_chat_message(content: str, is_user: bool = False) -> str:
    """Format a chat message with proper styling."""
    if is_user:
        return f"""
        <div style='
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            border-bottom-right-radius: 5px;
            max-width: 80%;
            margin-left: auto;
        '>
            <p style='color: #444; margin: 0;'>{content}</p>
        </div>
        """
    else:
        # For assistant messages, use a different style
        return f"""
        <div style='
            background-color: #f8f9fa;
            border-left: 4px solid #3366CC;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            max-width: 90%;
        '>
            <p style='color: #333; margin: 0; line-height: 1.6;'>{format_llm_text(content)}</p>
        </div>
        """

def format_llm_text(content: str) -> str:
    """Format LLM text content with proper styling."""
    # Clean up TextBlock wrapper if present
    import re
    if hasattr(content, "text"):
        content = content.text
    elif isinstance(content, list):
        content = " ".join(map(str, content))
    
    text = str(content)
    
    # Extract text from TextBlock if present
    match = re.search(r"text=['\"](.*?)['\"]\)?$", text, re.DOTALL)
    if match:
        text = match.group(1)
    elif "TextBlock" in text:
        # Alternative pattern
        match = re.search(r'text\s*=\s*["\'](.+?)["\']', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Remove all asterisks completely
    text = text.replace('*', '')
    text = text.replace('**', '')
    
    # Also remove any ', type='text' trailing part
    text = re.sub(r', type=\'text\'', '', text)
    text = re.sub(r', type="text"', '', text)
    
    # Fix run-together words by adding spaces
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split camelCase
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Add space between numbers and text
    
    # Clean up escape characters
    text = text.replace('\\n', '\n')
    text = text.replace('â¢', '•')
    text = text.replace('â', '-')
    text = text.replace('−', '-')
    
    # Format numbered sections and headers - using different formatting since we removed asterisks
    text = re.sub(r'(\d+\.)', r'\n\1', text)
    text = re.sub(r'([A-Za-z\s]+):(\s*\n|\s+)', r'\n\1:', text)
    
    # Add spaces after punctuation
    text = re.sub(r'([.,!?:;])([^\s])', r'\1 \2', text)
    
    # Clean up newlines and spaces
    text = ' '.join(text.split())
    text = text.replace("\n\n", "\n")
    
    # Format bullet points and lists
    text = text.replace('• ', '\n• ')
    text = re.sub(r'(\d+\.) ', r'\n\1 ', text)
    
    return text.strip()

def display_chat_message(role: str, content: str) -> None:
    """Display a chat message with emoji style."""
    if role == "user":
        st.write('👤 **You**: ' + content)
    else:
        # Format the LLM text content but keep chat style
        formatted_content = format_llm_text(content)
        st.write('🤖 **AI**: ' + formatted_content)

def format_expert_message(agent_type: str, response: str) -> str:
    """Format an expert agent message for chat display."""
    formatted_text = format_llm_text(response)
    return f"**{agent_type.title()} Expert** 👨‍💼\n{formatted_text}"

def format_consensus_message(response: str) -> str:
    """Format a consensus message for chat display."""
    formatted_text = format_llm_text(response)
    return f"**Expert Consensus** 🤝\n{formatted_text}"

def show_advisor_view(components):
    """Show the AI financial advisor chat view."""
    st.markdown("## 💬 AI Financial Advisor")
    
    # Get components
    agent_manager = components["agent_manager"]
    
    # Initialize chat history if needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Display multi-agent selection
    with st.expander("⚙️ Configure Advisor Options"):
        st.markdown("#### Select Financial Experts")
        
        # Agent selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget_selected = st.checkbox("💰 Budget Expert", value="budget" in st.session_state.selected_agents)
            investment_selected = st.checkbox("📈 Investment Expert", value="investment" in st.session_state.selected_agents)
        
        with col2:
            debt_selected = st.checkbox("💳 Debt Expert", value="debt" in st.session_state.selected_agents)
            savings_selected = st.checkbox("🏦 Savings Expert", value="savings" in st.session_state.selected_agents)
        
        with col3:
            tax_selected = st.checkbox("📊 Tax Expert", value="tax" in st.session_state.selected_agents)
        
        # Update selected agents
        selected_agents = []
        if budget_selected: selected_agents.append("budget")
        if investment_selected: selected_agents.append("investment")
        if debt_selected: selected_agents.append("debt")
        if savings_selected: selected_agents.append("savings")
        if tax_selected: selected_agents.append("tax")
        
        st.session_state.selected_agents = selected_agents if selected_agents else ["budget", "investment", "debt", "savings", "tax"]
        
        # Option for debate or consensus
        advisor_mode = st.radio("Advisor Mode", ["Consensus", "Debate"], index=0)
        
        if advisor_mode == "Debate":
            st.info("💭 In debate mode, each expert will provide their own perspective on your question.")
        else:
            st.info("🤝 In consensus mode, the experts will collaborate to provide a unified response.")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    # Chat input
    user_query = st.chat_input("Ask your financial question here...")
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])
    
    if user_query:
        # Display user message
        display_chat_message("user", user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        if advisor_mode == "Consensus":
            # Get holistic advice
            with st.spinner("🤔 Getting expert financial advice..."):
                response = agent_manager.get_holistic_advice(user_data, user_query)
                formatted_response = format_consensus_message(response.get("consensus", ""))
                
                # Display and store response
                display_chat_message("assistant", formatted_response)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": formatted_response
                })
        else:  # Debate mode
            responses = []
            
            # Get responses from each expert
            for agent_type in st.session_state.selected_agents:
                with st.spinner(f"💭 Getting advice from {agent_type} expert..."):
                    agent = agent_manager.get_agent(agent_type)
                    response = agent.chat_response(user_query, user_data, st.session_state.chat_history)
                    formatted_response = format_expert_message(agent_type, response)
                    responses.append(formatted_response)
            
            # Display combined response
            combined_response = "\n\n".join(responses)
            display_chat_message("assistant", combined_response)
            
            # Store in chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": combined_response
            })

def show_profile_view(components):
    """Show the profile view with all user information."""
    st.title("Profile")
    
    # Create tabs for different sections of the profile
    tabs = create_tabs([
        "Personal Info",
        "Income",
        "Expenses",
        "Debt",
        "Investments",
        "Savings"
    ])
    
    with tabs[0]:
        render_profile_section(st.session_state.user_data)
    
    with tabs[1]:
        render_income_section(st.session_state.user_data)
    
    with tabs[2]:
        render_expenses_section(st.session_state.user_data)
    
    with tabs[3]:
        render_debt_section(st.session_state.user_data)
    
    with tabs[4]:
        render_investment_section(st.session_state.user_data)
    
    with tabs[5]:
        render_savings_section(st.session_state.user_data)

def show_budget_view(components):
    """Show the budget planner view."""
    st.markdown("<h2 class='sub-header'>Budget Planner</h2>", unsafe_allow_html=True)
    
    # Get components
    agent_manager = components["agent_manager"]
    visualizer = components["visualizer"]
    data_loader = components["data_loader"]
    
    # Get user data
    user_data = st.session_state.user_data
    
    # Create tabs
    tabs = st.tabs(["Budget Overview", "Income & Expenses", "Budget Analysis"])
    
    with tabs[0]:
        # Budget Overview
        st.markdown("### Budget Overview")
        
        # Calculate totals
        total_income = calculate_total_income(user_data["income"])
        total_expenses = calculate_total_expenses(user_data["expenses"])
        savings_rate = calculate_savings_rate(total_income, total_expenses)
        
        # Display metrics
        metrics = [
            ("Total Income", format_currency(total_income), None),
            ("Total Expenses", format_currency(total_expenses), None),
            ("Savings Rate", format_percentage(savings_rate), None)
        ]
        create_metric_columns(metrics)
        
        # Budget breakdown chart
        st.markdown("### Budget Breakdown")
        budget_chart = visualizer.create_budget_chart(user_data)
        display_chart(budget_chart, key="budget_breakdown_chart")
    
    with tabs[1]:
        # Income & Expenses
        st.markdown("### Income & Expenses")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Income Analysis
            st.markdown("#### Income Sources")
            income_data = [{"Source": source, "Amount": amount} 
                          for source, amount in user_data["income"].items()]
            income_df = pd.DataFrame(income_data)
            income_df["Percentage"] = income_df["Amount"] / income_df["Amount"].sum() * 100
            
            # Format for display
            income_display = income_df.copy()
            income_display["Amount"] = income_display["Amount"].apply(lambda x: format_currency(x))
            income_display["Percentage"] = income_display["Percentage"].apply(lambda x: format_percentage(x))
            display_dataframe(income_display, key="income_sources_table")
        
        with col2:
            # Expense Analysis
            st.markdown("#### Expense Categories")
            expense_data = [{"Category": category, "Amount": amount} 
                           for category, amount in user_data["expenses"].items()]
            expense_df = pd.DataFrame(expense_data)
            expense_df["Percentage"] = expense_df["Amount"] / expense_df["Amount"].sum() * 100
            
            # Format for display
            expense_display = expense_df.copy()
            expense_display["Amount"] = expense_display["Amount"].apply(lambda x: format_currency(x))
            expense_display["Percentage"] = expense_display["Percentage"].apply(lambda x: format_percentage(x))
            display_dataframe(expense_display, key="expense_categories_table")
    
    with tabs[2]:
        # Budget Analysis
        st.markdown("### Budget Analysis")
        
        if st.button("Analyze Budget"):
            with st.spinner("Analyzing your budget..."):
                # Get budget analysis from budget agent
                budget_agent = agent_manager.get_agent("budget")
                budget_analysis = budget_agent.analyze_budget(user_data)
                
                # Store in session state
                st.session_state.agent_outputs["budget_analysis"] = budget_analysis
        
        # Display budget analysis if available
        if "budget_analysis" in st.session_state.agent_outputs:
            analysis = st.session_state.agent_outputs["budget_analysis"]
            display_llm_response(analysis.get("analysis", ""), "Budget Analysis")
        
        # Spending Optimization
        st.markdown("### Spending Optimization")
        
        if st.button("Find Savings Opportunities"):
            with st.spinner("Analyzing expenses for savings opportunities..."):
                budget_agent = agent_manager.get_agent("budget")
                savings_opps = budget_agent.identify_savings_opportunities(user_data["expenses"])
                st.session_state.agent_outputs["savings_opportunities"] = savings_opps
        
        # Display savings opportunities if available
        if "savings_opportunities" in st.session_state.agent_outputs:
            opps = st.session_state.agent_outputs["savings_opportunities"]
            display_llm_response(opps.get("opportunities", ""), "Savings Opportunities")

# Run the application
if __name__ == "__main__":
    main()