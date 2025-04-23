import streamlit as st
from .components import (
    display_metric_card, display_chart, display_dataframe,
    display_styled_message, create_metric_columns, create_tabs,
    create_columns, display_agent_response, format_currency,
    format_percentage
)
from utils.data_processing import (
    calculate_total_income, calculate_total_expenses,
    calculate_savings_rate, calculate_portfolio_value,
    calculate_asset_allocation, format_financial_data
)

def render_dashboard_metrics(user_data):
    """Render dashboard metrics."""
    total_income = calculate_total_income(user_data["income"])
    total_expenses = calculate_total_expenses(user_data["expenses"])
    savings_rate = calculate_savings_rate(total_income, total_expenses)
    
    metrics = [
        ("Total Income", format_financial_data(total_income), None),
        ("Total Expenses", format_financial_data(total_expenses), None),
        ("Savings Rate", format_financial_data(savings_rate, "percentage"), None)
    ]
    
    create_metric_columns(metrics)

def render_investment_metrics(user_data):
    """Render investment metrics."""
    portfolio_value = calculate_portfolio_value(user_data["investments"])
    allocation_data = calculate_asset_allocation(user_data["investments"])
    
    metrics = [
        ("Portfolio Value", format_financial_data(portfolio_value), None),
        ("Stocks", format_financial_data(allocation_data.get("stocks", 0), "percentage"), None),
        ("Bonds", format_financial_data(allocation_data.get("bonds", 0), "percentage"), None)
    ]
    
    create_metric_columns(metrics)

def render_debt_metrics(user_data):
    """Render debt metrics."""
    total_debt = sum(
        card.get("balance", 0) for card in user_data["debts"].get("credit_cards", [])
    ) + sum(
        loan.get("balance", 0) for loan in user_data["debts"].get("student_loans", [])
    ) + sum(
        mortgage.get("balance", 0) for mortgage in user_data["debts"].get("mortgage", [])
    )
    
    monthly_payments = sum(
        card.get("minimum_payment", 0) for card in user_data["debts"].get("credit_cards", [])
    ) + sum(
        loan.get("minimum_payment", 0) for loan in user_data["debts"].get("student_loans", [])
    ) + sum(
        mortgage.get("minimum_payment", 0) for mortgage in user_data["debts"].get("mortgage", [])
    )
    
    monthly_income = calculate_total_income(user_data["income"])
    dti_ratio = calculate_debt_to_income_ratio(monthly_payments, monthly_income)
    
    metrics = [
        ("Total Debt", format_financial_data(total_debt), None),
        ("Monthly Payments", format_financial_data(monthly_payments), None),
        ("Debt-to-Income Ratio", format_financial_data(dti_ratio, "percentage"), None)
    ]
    
    create_metric_columns(metrics)

def render_savings_metrics(user_data):
    """Render savings metrics."""
    emergency_fund = user_data["savings"].get("emergency_fund", {})
    current_balance = emergency_fund.get("balance", 0)
    target_amount = emergency_fund.get("target", 0)
    progress = (current_balance / target_amount * 100) if target_amount > 0 else 0
    
    metrics = [
        ("Emergency Fund", format_financial_data(current_balance), None),
        ("Target Amount", format_financial_data(target_amount), None),
        ("Progress", format_financial_data(progress, "percentage"), None)
    ]
    
    create_metric_columns(metrics)

def render_profile_section(user_data):
    """Render the profile section."""
    st.subheader("Personal Information")
    
    col1, col2 = create_columns(2)
    
    with col1:
        user_data["personal"]["name"] = st.text_input("Name", value=user_data["personal"].get("name", ""))
        user_data["personal"]["age"] = st.number_input("Age", min_value=18, max_value=100, value=user_data["personal"].get("age", 30))
    
    with col2:
        user_data["personal"]["filing_status"] = st.selectbox(
            "Filing Status", 
            ["single", "married", "married_filing_separately", "head_of_household"],
            index=["single", "married", "married_filing_separately", "head_of_household"].index(user_data["personal"].get("filing_status", "single"))
        )
        user_data["personal"]["dependents"] = st.number_input("Number of Dependents", min_value=0, max_value=10, value=user_data["personal"].get("dependents", 0))
    
    st.subheader("Location")
    col1, col2 = create_columns(2)
    
    with col1:
        user_data["personal"]["location"]["country"] = st.selectbox(
            "Country", 
            ["US", "Canada", "UK", "Australia", "Other"],
            index=["US", "Canada", "UK", "Australia", "Other"].index(user_data["personal"]["location"].get("country", "US"))
        )
    
    with col2:
        user_data["personal"]["location"]["state"] = st.text_input("State/Province", value=user_data["personal"]["location"].get("state", ""))

def render_income_section(user_data):
    """Render the income section."""
    st.subheader("Income Sources")
    
    col1, col2 = create_columns(2)
    
    with col1:
        user_data["income"]["salary"] = st.number_input(
            "Salary (Monthly)", 
            min_value=0.0, 
            value=float(user_data["income"].get("salary", 0)),
            step=100.0,
            format="%0.2f"
        )
        user_data["income"]["self_employment"] = st.number_input(
            "Self-Employment (Monthly)", 
            min_value=0.0, 
            value=float(user_data["income"].get("self_employment", 0)),
            step=100.0,
            format="%0.2f"
        )
    
    with col2:
        user_data["income"]["investments"] = st.number_input(
            "Investment Income (Monthly)", 
            min_value=0.0, 
            value=float(user_data["income"].get("investments", 0)),
            step=100.0,
            format="%0.2f"
        )
        user_data["income"]["other"] = st.number_input(
            "Other Income (Monthly)", 
            min_value=0.0, 
            value=float(user_data["income"].get("other", 0)),
            step=100.0,
            format="%0.2f"
        )
    
    total_income = calculate_total_income(user_data["income"])
    st.metric("Total Monthly Income", format_currency(total_income))

def render_expenses_section(user_data):
    """Render the expenses section."""
    st.subheader("Monthly Expenses")
    
    col1, col2 = create_columns(2)
    
    with col1:
        user_data["expenses"]["housing"] = st.number_input(
            "Housing", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("housing", 0)),
            step=10.0,
            format="%0.2f"
        )
        user_data["expenses"]["transportation"] = st.number_input(
            "Transportation", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("transportation", 0)),
            step=10.0,
            format="%0.2f"
        )
        user_data["expenses"]["food"] = st.number_input(
            "Food", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("food", 0)),
            step=10.0,
            format="%0.2f"
        )
        user_data["expenses"]["utilities"] = st.number_input(
            "Utilities", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("utilities", 0)),
            step=10.0,
            format="%0.2f"
        )
    
    with col2:
        user_data["expenses"]["insurance"] = st.number_input(
            "Insurance", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("insurance", 0)),
            step=10.0,
            format="%0.2f"
        )
        user_data["expenses"]["healthcare"] = st.number_input(
            "Healthcare", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("healthcare", 0)),
            step=10.0,
            format="%0.2f"
        )
        user_data["expenses"]["personal"] = st.number_input(
            "Personal", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("personal", 0)),
            step=10.0,
            format="%0.2f"
        )
        user_data["expenses"]["entertainment"] = st.number_input(
            "Entertainment", 
            min_value=0.0, 
            value=float(user_data["expenses"].get("entertainment", 0)),
            step=10.0,
            format="%0.2f"
        )
    
    user_data["expenses"]["other"] = st.number_input(
        "Other Expenses", 
        min_value=0.0, 
        value=float(user_data["expenses"].get("other", 0)),
        step=10.0,
        format="%0.2f"
    )
    
    total_expenses = calculate_total_expenses(user_data["expenses"])
    st.metric("Total Monthly Expenses", format_currency(total_expenses))
    
    # Update cashflow in user data
    user_data["monthly_cashflow"]["total_income"] = total_income
    user_data["monthly_cashflow"]["total_expenses"] = total_expenses
    user_data["monthly_cashflow"]["surplus_deficit"] = total_income - total_expenses

def render_debt_section(user_data):
    """Render the debt section."""
    st.subheader("Debts")
    
    # Credit Cards
    st.markdown("#### Credit Cards")
    for i, card in enumerate(user_data["debts"].get("credit_cards", [])):
        cols = create_columns([3, 2, 2, 2, 1])
        with cols[0]:
            user_data["debts"]["credit_cards"][i]["name"] = st.text_input(f"Card Name {i+1}", value=card.get("name", ""), key=f"cc_name_{i}")
        with cols[1]:
            user_data["debts"]["credit_cards"][i]["balance"] = st.number_input(f"Balance {i+1}", min_value=0.0, value=float(card.get("balance", 0)), step=100.0, format="%0.2f", key=f"cc_bal_{i}")
        with cols[2]:
            user_data["debts"]["credit_cards"][i]["interest_rate"] = st.number_input(f"Interest Rate {i+1} (%)", min_value=0.0, max_value=30.0, value=float(card.get("interest_rate", 0)), step=0.5, format="%0.2f", key=f"cc_rate_{i}")
        with cols[3]:
            user_data["debts"]["credit_cards"][i]["minimum_payment"] = st.number_input(f"Min Payment {i+1}", min_value=0.0, value=float(card.get("minimum_payment", 0)), step=10.0, format="%0.2f", key=f"cc_min_{i}")
        with cols[4]:
            if st.button("🗑️", key=f"del_cc_{i}"):
                user_data["debts"]["credit_cards"].pop(i)
                st.rerun()
    
    if st.button("Add Credit Card"):
        if "credit_cards" not in user_data["debts"]:
            user_data["debts"]["credit_cards"] = []
        user_data["debts"]["credit_cards"].append({"name": "", "balance": 0, "interest_rate": 0, "minimum_payment": 0})
        st.rerun()
    
    # Student Loans
    st.markdown("#### Student Loans")
    for i, loan in enumerate(user_data["debts"].get("student_loans", [])):
        cols = create_columns([3, 2, 2, 2, 1])
        with cols[0]:
            user_data["debts"]["student_loans"][i]["name"] = st.text_input(f"Loan Name {i+1}", value=loan.get("name", ""), key=f"sl_name_{i}")
        with cols[1]:
            user_data["debts"]["student_loans"][i]["balance"] = st.number_input(f"Balance {i+1}", min_value=0.0, value=float(loan.get("balance", 0)), step=100.0, format="%0.2f", key=f"sl_bal_{i}")
        with cols[2]:
            user_data["debts"]["student_loans"][i]["interest_rate"] = st.number_input(f"Interest Rate {i+1} (%)", min_value=0.0, max_value=15.0, value=float(loan.get("interest_rate", 0)), step=0.25, format="%0.2f", key=f"sl_rate_{i}")
        with cols[3]:
            user_data["debts"]["student_loans"][i]["minimum_payment"] = st.number_input(f"Min Payment {i+1}", min_value=0.0, value=float(loan.get("minimum_payment", 0)), step=10.0, format="%0.2f", key=f"sl_min_{i}")
        with cols[4]:
            if st.button("🗑️", key=f"del_sl_{i}"):
                user_data["debts"]["student_loans"].pop(i)
                st.rerun()
    
    if st.button("Add Student Loan"):
        if "student_loans" not in user_data["debts"]:
            user_data["debts"]["student_loans"] = []
        user_data["debts"]["student_loans"].append({"name": "", "balance": 0, "interest_rate": 0, "minimum_payment": 0})
        st.rerun()

def render_investment_section(user_data):
    """Render the investment section."""
    st.subheader("Investments")
    
    # Retirement Accounts
    st.markdown("#### Retirement Accounts")
    for i, account in enumerate(user_data["investments"].get("retirement_accounts", [])):
        cols = create_columns([3, 2, 2, 2, 1])
        with cols[0]:
            user_data["investments"]["retirement_accounts"][i]["name"] = st.text_input(f"Account {i+1}", value=account.get("name", ""), key=f"ra_name_{i}")
        with cols[1]:
            user_data["investments"]["retirement_accounts"][i]["balance"] = st.number_input(f"Balance {i+1}", min_value=0.0, value=float(account.get("balance", 0)), step=1000.0, format="%0.2f", key=f"ra_bal_{i}")
        with cols[2]:
            user_data["investments"]["retirement_accounts"][i]["contribution_rate"] = st.number_input(f"Contribution % {i+1}", min_value=0.0, max_value=100.0, value=float(account.get("contribution_rate", 0)), step=1.0, format="%0.2f", key=f"ra_cont_{i}")
        with cols[3]:
            stock_percent = account.get("asset_allocation", {}).get("stocks", 0)
            stock_percent = st.number_input(f"Stock % {i+1}", min_value=0.0, max_value=100.0, value=float(stock_percent), step=5.0, format="%0.2f", key=f"ra_stock_{i}")
            user_data["investments"]["retirement_accounts"][i]["asset_allocation"] = {"stocks": stock_percent, "bonds": 100 - stock_percent}
        with cols[4]:
            if st.button("🗑️", key=f"del_ra_{i}"):
                user_data["investments"]["retirement_accounts"].pop(i)
                st.rerun()
    
    if st.button("Add Retirement Account"):
        if "retirement_accounts" not in user_data["investments"]:
            user_data["investments"]["retirement_accounts"] = []
        user_data["investments"]["retirement_accounts"].append({"name": "", "balance": 0, "contribution_rate": 0, "asset_allocation": {"stocks": 0, "bonds": 0}})
        st.rerun()
    
    # Brokerage Accounts
    st.markdown("#### Brokerage Accounts")
    for i, account in enumerate(user_data["investments"].get("brokerage_accounts", [])):
        cols = create_columns([4, 3, 2, 1])
        with cols[0]:
            user_data["investments"]["brokerage_accounts"][i]["name"] = st.text_input(f"Account {i+1}", value=account.get("name", ""), key=f"ba_name_{i}")
        with cols[1]:
            user_data["investments"]["brokerage_accounts"][i]["balance"] = st.number_input(f"Balance {i+1}", min_value=0.0, value=float(account.get("balance", 0)), step=1000.0, format="%0.2f", key=f"ba_bal_{i}")
        with cols[2]:
            allocation = account.get("asset_allocation", {})
            options = ["stocks", "bonds", "cash", "other"]
            allocation_type = st.selectbox(f"Main Asset {i+1}", options, index=0, key=f"ba_type_{i}")
            
            if "asset_allocation" not in user_data["investments"]["brokerage_accounts"][i]:
                user_data["investments"]["brokerage_accounts"][i]["asset_allocation"] = {}
            
            for opt in options:
                user_data["investments"]["brokerage_accounts"][i]["asset_allocation"][opt] = 100 if opt == allocation_type else 0
        
        with cols[3]:
            if st.button("🗑️", key=f"del_ba_{i}"):
                user_data["investments"]["brokerage_accounts"].pop(i)
                st.rerun()
    
    if st.button("Add Brokerage Account"):
        if "brokerage_accounts" not in user_data["investments"]:
            user_data["investments"]["brokerage_accounts"] = []
        user_data["investments"]["brokerage_accounts"].append({"name": "", "balance": 0, "asset_allocation": {"stocks": 100, "bonds": 0, "cash": 0, "other": 0}})
        st.rerun()

def render_savings_section(user_data):
    """Render the savings section."""
    st.subheader("Savings")
    
    # Emergency Fund
    st.markdown("#### Emergency Fund")
    col1, col2 = create_columns(2)
    with col1:
        user_data["savings"]["emergency_fund"]["balance"] = st.number_input(
            "Current Balance", 
            min_value=0.0, 
            value=float(user_data["savings"]["emergency_fund"].get("balance", 0)),
            step=500.0,
            format="%0.2f"
        )
    with col2:
        user_data["savings"]["emergency_fund"]["target"] = st.number_input(
            "Target Amount", 
            min_value=0.0, 
            value=float(user_data["savings"]["emergency_fund"].get("target", 0)),
            step=500.0,
            format="%0.2f"
        )
    
    # Savings Accounts
    st.markdown("#### Savings Accounts")
    for i, account in enumerate(user_data["savings"].get("savings_accounts", [])):
        cols = create_columns([3, 2, 2, 3, 1])
        with cols[0]:
            user_data["savings"]["savings_accounts"][i]["name"] = st.text_input(f"Account {i+1}", value=account.get("name", ""), key=f"sa_name_{i}")
        with cols[1]:
            user_data["savings"]["savings_accounts"][i]["balance"] = st.number_input(f"Balance {i+1}", min_value=0.0, value=float(account.get("balance", 0)), step=500.0, format="%0.2f", key=f"sa_bal_{i}")
        with cols[2]:
            user_data["savings"]["savings_accounts"][i]["interest_rate"] = st.number_input(f"Interest Rate {i+1} (%)", min_value=0.0, max_value=10.0, value=float(account.get("interest_rate", 0)), step=0.1, format="%0.2f", key=f"sa_int_{i}")
        with cols[3]:
            user_data["savings"]["savings_accounts"][i]["purpose"] = st.text_input(f"Purpose {i+1}", value=account.get("purpose", ""), key=f"sa_purp_{i}")
        with cols[4]:
            if st.button("🗑️", key=f"del_sa_{i}"):
                user_data["savings"]["savings_accounts"].pop(i)
                st.rerun()
    
    if st.button("Add Savings Account"):
        if "savings_accounts" not in user_data["savings"]:
            user_data["savings"]["savings_accounts"] = []
        user_data["savings"]["savings_accounts"].append({"name": "", "balance": 0, "interest_rate": 0, "purpose": ""})
        st.rerun() 