def calculate_total_income(income_data):
    """Calculate total income from income data."""
    return sum(income_data.values())

def calculate_total_expenses(expenses_data):
    """Calculate total expenses from expenses data."""
    return sum(expenses_data.values())

def calculate_savings_rate(income, expenses):
    """Calculate savings rate as a percentage."""
    return ((income - expenses) / income * 100) if income > 0 else 0

def calculate_debt_to_income_ratio(monthly_debt_payments: float, monthly_income: float) -> float:
    """Calculate the debt-to-income ratio.
    
    Args:
        monthly_debt_payments: Total monthly debt payments
        monthly_income: Total monthly income
    
    Returns:
        Debt-to-income ratio as a percentage
    """
    if monthly_income == 0:
        return 0.0
    return (monthly_debt_payments / monthly_income) * 100

def calculate_portfolio_value(investments_data):
    """Calculate total portfolio value from investments data."""
    total = 0
    
    # Add retirement accounts
    for account in investments_data.get("retirement_accounts", []):
        total += account.get("balance", 0)
    
    # Add brokerage accounts
    for account in investments_data.get("brokerage_accounts", []):
        total += account.get("balance", 0)
    
    # Add real estate equity
    for property in investments_data.get("real_estate", []):
        equity = property.get("estimated_value", 0) - property.get("mortgage_balance", 0)
        total += equity
    
    # Add other investments
    for investment in investments_data.get("other_investments", []):
        total += investment.get("value", 0)
    
    return total

def calculate_asset_allocation(investments_data):
    """Calculate asset allocation percentages."""
    allocation_data = {}
    total_value = calculate_portfolio_value(investments_data)
    
    if total_value > 0:
        # Add retirement account allocations
        for account in investments_data.get("retirement_accounts", []):
            balance = account.get("balance", 0)
            for asset_class, percentage in account.get("asset_allocation", {}).items():
                allocation_data[asset_class] = allocation_data.get(asset_class, 0) + (balance * percentage / 100)
        
        # Add brokerage account allocations
        for account in investments_data.get("brokerage_accounts", []):
            balance = account.get("balance", 0)
            for asset_class, percentage in account.get("asset_allocation", {}).items():
                allocation_data[asset_class] = allocation_data.get(asset_class, 0) + (balance * percentage / 100)
        
        # Add real estate
        real_estate_equity = sum(property.get("estimated_value", 0) - property.get("mortgage_balance", 0) 
                               for property in investments_data.get("real_estate", []))
        if real_estate_equity > 0:
            allocation_data["real_estate"] = allocation_data.get("real_estate", 0) + real_estate_equity
        
        # Convert to percentages
        allocation_data = {k: (v / total_value) * 100 for k, v in allocation_data.items()}
    
    return allocation_data

def format_financial_data(data, format_type="currency"):
    """Format financial data based on type."""
    if format_type == "currency":
        return f"${data:,.2f}"
    elif format_type == "percentage":
        return f"{data:.1f}%"
    return str(data) 