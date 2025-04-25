from typing import Dict, List, Union
import numpy as np

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

def calculate_portfolio_value(investments: Dict) -> float:
    """
    Calculate the total value of an investment portfolio.
    
    Args:
        investments: Dictionary containing investment data
        
    Returns:
        Total portfolio value
    """
    total_value = 0
    
    # Sum up values from different investment types
    for investment_type, holdings in investments.items():
        if isinstance(holdings, list):
            for holding in holdings:
                total_value += holding.get("current_value", 0)
        elif isinstance(holdings, dict):
            total_value += holdings.get("current_value", 0)
    
    return total_value

def calculate_asset_allocation(investments: Dict) -> Dict[str, float]:
    """
    Calculate the percentage allocation of different asset classes in the portfolio.
    
    Args:
        investments: Dictionary containing investment data
        
    Returns:
        Dictionary with asset class allocations as percentages
    """
    allocation = {
        "Stocks": 0.0,
        "Bonds": 0.0,
        "Cash": 0.0,
        "Real Estate": 0.0,
        "Other": 0.0
    }
    
    total_value = 0
    
    # Calculate total value first
    if "retirement_accounts" in investments:
        for account in investments["retirement_accounts"]:
            total_value += account.get("balance", 0)
            
    if "brokerage_accounts" in investments:
        for account in investments["brokerage_accounts"]:
            total_value += account.get("balance", 0)
            
    if "real_estate" in investments:
        for property in investments["real_estate"]:
            total_value += property.get("estimated_value", 0) - property.get("mortgage_balance", 0)
            
    if "other_investments" in investments:
        for investment in investments["other_investments"]:
            total_value += investment.get("value", 0)
    
    if total_value == 0:
        return allocation
    
    # Calculate allocations
    if "retirement_accounts" in investments:
        for account in investments["retirement_accounts"]:
            balance = account.get("balance", 0)
            asset_alloc = account.get("asset_allocation", {})
            for asset_type, percentage in asset_alloc.items():
                asset_value = balance * (percentage / 100)
                if asset_type.lower() == "stocks":
                    allocation["Stocks"] += (asset_value / total_value) * 100
                elif asset_type.lower() == "bonds":
                    allocation["Bonds"] += (asset_value / total_value) * 100
                else:
                    allocation["Other"] += (asset_value / total_value) * 100
    
    if "brokerage_accounts" in investments:
        for account in investments["brokerage_accounts"]:
            balance = account.get("balance", 0)
            asset_alloc = account.get("asset_allocation", {})
            for asset_type, percentage in asset_alloc.items():
                if percentage > 0:  # Only count if percentage is greater than 0
                    if asset_type.lower() == "stocks":
                        allocation["Stocks"] += (balance / total_value) * 100
                    elif asset_type.lower() == "bonds":
                        allocation["Bonds"] += (balance / total_value) * 100
                    elif asset_type.lower() == "cash":
                        allocation["Cash"] += (balance / total_value) * 100
                    else:
                        allocation["Other"] += (balance / total_value) * 100
    
    if "real_estate" in investments:
        for property in investments["real_estate"]:
            equity = property.get("estimated_value", 0) - property.get("mortgage_balance", 0)
            if equity > 0:
                allocation["Real Estate"] += (equity / total_value) * 100
    
    if "other_investments" in investments:
        for investment in investments["other_investments"]:
            value = investment.get("value", 0)
            allocation["Other"] += (value / total_value) * 100
    
    return allocation

def calculate_monthly_savings_rate(income: float, expenses: float) -> float:
    """
    Calculate the monthly savings rate as a percentage of income.
    
    Args:
        income: Total monthly income
        expenses: Total monthly expenses
        
    Returns:
        Savings rate as a percentage
    """
    if income == 0:
        return 0.0
    
    savings = income - expenses
    return (savings / income) * 100

def calculate_debt_metrics(debts: Dict) -> Dict[str, Union[float, Dict]]:
    """
    Calculate various debt-related metrics.
    
    Args:
        debts: Dictionary containing debt information
        
    Returns:
        Dictionary containing debt metrics
    """
    metrics = {
        "total_debt": 0.0,
        "weighted_avg_interest": 0.0,
        "debt_by_type": {},
        "high_interest_debt": 0.0
    }
    
    total_weighted_interest = 0.0
    
    for debt_type, debt_list in debts.items():
        type_total = 0.0
        
        for debt in debt_list:
            balance = debt.get("balance", 0)
            interest_rate = debt.get("interest_rate", 0)
            
            metrics["total_debt"] += balance
            type_total += balance
            total_weighted_interest += balance * interest_rate
            
            if interest_rate > 15:
                metrics["high_interest_debt"] += balance
        
        metrics["debt_by_type"][debt_type] = type_total
    
    # Calculate weighted average interest rate
    if metrics["total_debt"] > 0:
        metrics["weighted_avg_interest"] = total_weighted_interest / metrics["total_debt"]
    
    return metrics

def analyze_expense_trends(expenses: List[Dict], months: int = 6) -> Dict[str, float]:
    """
    Analyze expense trends over the specified number of months.
    
    Args:
        expenses: List of monthly expense records
        months: Number of months to analyze
        
    Returns:
        Dictionary containing trend analysis
    """
    if not expenses or months <= 0:
        return {}
    
    # Get the most recent months' data
    recent_expenses = expenses[-months:]
    
    # Calculate month-over-month changes
    changes = []
    for i in range(1, len(recent_expenses)):
        prev_total = sum(recent_expenses[i-1].values())
        curr_total = sum(recent_expenses[i].values())
        if prev_total > 0:
            change = ((curr_total - prev_total) / prev_total) * 100
            changes.append(change)
    
    return {
        "avg_monthly_change": np.mean(changes) if changes else 0.0,
        "volatility": np.std(changes) if changes else 0.0,
        "max_increase": max(changes) if changes else 0.0,
        "max_decrease": min(changes) if changes else 0.0
    }

def format_financial_data(data, format_type="currency"):
    """Format financial data based on type."""
    if format_type == "currency":
        return f"${data:,.2f}"
    elif format_type == "percentage":
        return f"{data:.1f}%"
    return str(data) 