import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import random
from pathlib import Path
import os

from agents.agent_factory import create_agent
from patterns.voting_pattern import VotingPattern
from patterns.debate_pattern import DebatePattern
from patterns.multi_path_plan import MultiPathPlanPattern
from utils.data_loader import load_user_data

def render_dashboard():
    """
    Render the main dashboard page of the financial portal.
    """
    st.title("AI-Powered Financial Dashboard")
    
    # Show welcome message or financial summary based on data
    if not st.session_state.user_data["personal"].get("name"):
        render_welcome_dashboard()
    else:
        render_financial_summary_dashboard()

def render_welcome_dashboard():
    """
    Render welcome dashboard for new users.
    """
    st.markdown("""
    ## Welcome to Your AI Financial Portal!
    
    This portal uses multiple AI agents working together to help you manage your finances.
    
    ### Get Started:
    
    1. **Complete your profile** in the sidebar
    2. **Add your financial information** in each section
    3. **Get personalized recommendations** from our AI agents
    
    Our AI agents use collaborative patterns to provide better advice:
    - **Voting-Based Cooperation**: Agents vote on the best financial strategies
    - **Debate-Based Cooperation**: Agents debate pros and cons of financial decisions
    - **RAG (Retrieval Augmented Generation)**: Enhances responses with financial knowledge
    - **Human Reflection**: Incorporates your feedback to improve recommendations
    - **Multi-Path Planning**: Generates alternative financial scenarios
    """)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### Budget Management\nTrack expenses and optimize your spending habits.")
    
    with col2:
        st.info("### Investment Planning\nGet personalized investment strategies for your goals.")
    
    with col3:
        st.info("### Credit Card Optimization\nFind the best credit card strategies for your needs.")
    
    # Sample AI agent conversation
    st.markdown("### Sample AI Advisor Interaction")
    
    if "welcome_conversation" not in st.session_state:
        st.session_state.welcome_conversation = [
            {"role": "assistant", "content": "Hello! I'm your AI Financial Advisor. What financial goal would you like to work on today?"},
            {"role": "user", "content": "I want to save for a house down payment in 3 years."},
            {"role": "assistant", "content": "That's a great goal! To save for a house down payment in 3 years, I'll need to know a few things:\n\n1. How much do you need for the down payment?\n2. What's your current monthly income?\n3. What are your current monthly expenses?\n\nWith this information, we can create a savings plan to reach your goal."}
        ]
    
    for message in st.session_state.welcome_conversation:
        if message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        else:
            st.chat_message("user").write(message["content"])
    
    # Allow demo interaction
    if len(st.session_state.welcome_conversation) < 6:
        demo_input = st.chat_input("Try asking a question (demo)")
        if demo_input:
            st.session_state.welcome_conversation.append({"role": "user", "content": demo_input})
            
            # Generate a demo response
            demo_responses = [
                "Based on saving for a house down payment in 3 years, I recommend setting aside 20% of your monthly income in a high-yield savings account. This will help you reach your goal while maintaining financial flexibility. Would you like me to create a detailed savings plan?",
                "I understand you want to save for a house down payment. A typical down payment is 20% of the home's value. For a $300,000 home, you'd need $60,000. To save this in 3 years, you'd need to save about $1,667 per month. Is this amount feasible for your current financial situation?",
                "To help with your house down payment goal, I recommend setting up an automatic transfer to a high-yield savings account each payday. This 'pay yourself first' approach ensures consistent progress toward your goal. Would you like advice on specific high-yield savings accounts?"
            ]
            
            response = random.choice(demo_responses)
            st.session_state.welcome_conversation.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Get started button
    st.button("Get Started", type="primary", on_click=lambda: st.sidebar.write("👈 Complete your profile here"))

def render_financial_summary_dashboard():
    """
    Render financial summary dashboard for existing users.
    """
    user_data = st.session_state.user_data
    name = user_data["personal"].get("name", "User")
    
    st.markdown(f"## Hello, {name}!")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Financial Overview", "AI Insights", "Recent Activity"])
    
    with tab1:
        render_financial_overview(user_data)
    
    with tab2:
        render_ai_insights(user_data)
    
    with tab3:
        render_recent_activity(user_data)
    
    # Agent interaction section
    st.markdown("### Ask Your Financial AI Assistant")
    
    if "dashboard_conversation" not in st.session_state:
        st.session_state.dashboard_conversation = []
        
    # Display existing conversation
    for message in st.session_state.dashboard_conversation:
        if message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        else:
            st.chat_message("user").write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your finances...")
    if user_input:
        st.session_state.dashboard_conversation.append({"role": "user", "content": user_input})
        
        # Process with agent
        try:
            # Create a general financial advisor agent
            advisor = create_agent("general")
            response = advisor.process_user_input(user_input, user_data)
            
            st.session_state.dashboard_conversation.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")

def render_financial_overview(user_data):
    """
    Render financial overview section.
    """
    # Extract data
    personal = user_data.get("personal", {})
    budget = user_data.get("budget", {})
    investments = user_data.get("investments", {})
    
    monthly_income = personal.get("income", 0)
    expenses = personal.get("expenses", {})
    total_expenses = sum(expenses.values()) if expenses else 0
    savings = monthly_income - total_expenses if monthly_income > 0 else 0
    
    portfolio = investments.get("portfolio", {})
    total_invested = sum(asset_info.get("value", 0) for asset_info in portfolio.values())
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Income", f"${monthly_income:,.2f}")
    
    with col2:
        st.metric("Monthly Expenses", f"${total_expenses:,.2f}")
    
    with col3:
        st.metric("Monthly Savings", f"${savings:,.2f}")
        
    with col4:
        st.metric("Investments", f"${total_invested:,.2f}")
    
    # Create charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Income vs. Expenses")
        
        # Generate sample data for demonstration if no real data exists
        if not expenses:
            expenses = {
                "Housing": 1200,
                "Food": 500,
                "Transportation": 300,
                "Utilities": 200,
                "Entertainment": 150,
                "Other": 250
            }
            
        # Create income vs expenses figure
        fig = go.Figure()
        
        # Add income bar
        fig.add_trace(go.Bar(
            x=["Income"],
            y=[monthly_income],
            name="Income",
            marker_color="green"
        ))
        
        # Add expense categories as a stacked bar
        for category, amount in expenses.items():
            fig.add_trace(go.Bar(
                x=["Expenses"],
                y=[amount],
                name=category
            ))
        
        fig.update_layout(
            barmode="stack",
            height=400,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("Expense Breakdown")
        
        # Create pie chart for expense breakdown
        if expenses:
            fig = px.pie(
                values=list(expenses.values()),
                names=list(expenses.keys()),
                height=400
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data available. Add your expenses in the Budget section.")
    
    # Budget progress
    st.subheader("Budget Progress")
    
    categories = budget.get("categories", {})
    if categories and expenses:
        # Create a DataFrame for budget vs actual
        budget_data = []
        for category, budgeted in categories.items():
            actual = expenses.get(category, 0)
            percent = (actual / budgeted * 100) if budgeted > 0 else 0
            status = "Under Budget" if actual <= budgeted else "Over Budget"
            
            budget_data.append({
                "Category": category,
                "Budgeted": budgeted,
                "Actual": actual,
                "Percentage": percent,
                "Status": status
            })
        
        if budget_data:
            df = pd.DataFrame(budget_data)
            
            # Create a horizontal bar chart showing budget vs actual
            fig = go.Figure()
            
            for i, row in df.iterrows():
                # Add budget bar
                fig.add_trace(go.Bar(
                    y=[row["Category"]],
                    x=[row["Budgeted"]],
                    name="Budget",
                    orientation="h",
                    marker_color="rgba(58, 71, 80, 0.6)",
                    showlegend=i==0  # Only show legend for first item
                ))
                
                # Add actual spending bar
                fig.add_trace(go.Bar(
                    y=[row["Category"]],
                    x=[row["Actual"]],
                    name="Actual",
                    orientation="h",
                    marker_color="rgba(246, 78, 139, 0.6)" if row["Actual"] > row["Budgeted"] else "rgba(58, 184, 120, 0.6)",
                    showlegend=i==0  # Only show legend for first item
                ))
            
            fig.update_layout(
                barmode="group",
                height=50 * len(df) + 100,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No budget categories defined. Set up your budget in the Budget section.")
    else:
        st.info("No budget data available. Set up your budget in the Budget section.")

def render_ai_insights(user_data):
    """
    Render AI insights section.
    """
    st.subheader("AI-Generated Financial Insights")
    
    # Check if we have sufficient data to generate insights
    personal = user_data.get("personal", {})
    if not personal.get("income"):
        st.info("Add your income and expense information to get AI-generated insights.")
        return
    
    # Create insights or load cached ones
    if "insights" not in st.session_state:
        # Generate some sample insights (in a real app, these would come from agents)
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
    
    # Display insights
    for insight in st.session_state.insights:
        if insight["type"] == "observation":
            style = "info"
        elif insight["type"] == "recommendation":
            style = "success"
        elif insight["type"] == "warning":
            style = "warning"
        else:
            style = "info"
        
        getattr(st, style)(
            f"### {insight['title']}\n\n"
            f"{insight['description']}\n\n"
            f"*Generated using {insight['pattern']} pattern*"
        )
    
    # Button to refresh insights
    if st.button("Generate New Insights"):
        # In a real app, this would call the agent system to generate new insights
        st.session_state.pop("insights", None)
        st.rerun()

def render_recent_activity(user_data):
    """
    Render recent financial activity.
    """
    st.subheader("Recent Financial Activity")
    
    # In a real app, this would show actual transaction history
    # Here we'll create some sample data for demonstration
    
    if "transactions" not in st.session_state:
        # Generate sample transactions
        today = datetime.now()
        st.session_state.transactions = [
            {
                "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "description": "Grocery Store",
                "category": "Food",
                "amount": -78.52,
                "balance": 1243.67
            },
            {
                "date": (today - timedelta(days=3)).strftime("%Y-%m-%d"),
                "description": "Monthly Salary",
                "category": "Income",
                "amount": 3200.00,
                "balance": 1322.19
            },
            {
                "date": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
                "description": "Electric Bill",
                "category": "Utilities",
                "amount": -94.27,
                "balance": -1877.81
            },
            {
                "date": (today - timedelta(days=8)).strftime("%Y-%m-%d"),
                "description": "Restaurant Dinner",
                "category": "Food",
                "amount": -62.47,
                "balance": -1783.54
            },
            {
                "date": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
                "description": "Gas Station",
                "category": "Transportation",
                "amount": -38.72,
                "balance": -1721.07
            }
        ]
    
    # Create a DataFrame for the transactions
    df = pd.DataFrame(st.session_state.transactions)
    
    # Format the amount column
    df["amount_str"] = df["amount"].apply(
        lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
    )
    
    # Format the balance column
    df["balance_str"] = df["balance"].apply(
        lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
    )
    
    # Add color to amounts
    def color_amount(val):
        if val >= 0:
            return "color: green"
        else:
            return "color: red"
    
    # Display the transactions table
    st.dataframe(
        df[["date", "description", "category", "amount_str", "balance_str"]].rename(
            columns={
                "date": "Date",
                "description": "Description",
                "category": "Category",
                "amount_str": "Amount",
                "balance_str": "Balance"
            }
        ),
        height=300,
        use_container_width=True
    )