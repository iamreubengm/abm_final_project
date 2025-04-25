from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
from .data_processing import (
    calculate_portfolio_value,
    calculate_asset_allocation,
    calculate_monthly_savings_rate,
    calculate_debt_metrics,
    analyze_expense_trends
)

class InsightsGenerator:
    """
    Generates dynamic insights from user financial data.
    Analyzes spending patterns, savings goals, investment allocation,
    and overall financial health to provide actionable recommendations.
    """
    
    def __init__(self):
        self.insight_categories = {
            "spending": [],
            "savings": [],
            "investments": [],
            "debt": [],
            "cashflow": [],
            "goals": []
        }
        
    def generate_insights(self, user_data: Dict) -> Dict[str, List[str]]:
        """
        Generate financial insights based on user data.
        
        Args:
            user_data: Dictionary containing user's financial data
            
        Returns:
            Dictionary of insights categorized by financial aspect
        """
        # Reset insights for new generation
        for category in self.insight_categories:
            self.insight_categories[category] = []

        # Generate insights for each category
        self._analyze_spending(user_data)
        self._analyze_savings(user_data)
        self._analyze_investments(user_data)
        self._analyze_debt(user_data)
        self._analyze_cashflow(user_data)
        self._analyze_goals(user_data)

        return self.insight_categories
    
    def _analyze_spending(self, user_data: Dict):
        """Analyze spending patterns and generate insights."""
        expenses = user_data.get("expenses", [])
        income = user_data.get("monthly_income", 0)
        
        if not expenses:
            return
        
        # Analyze expense trends
        trends = analyze_expense_trends(expenses)
        
        # Check for significant expense increases
        if trends["avg_monthly_change"] > 5:
            self.insight_categories["spending"].append(
                f"Your expenses have increased by an average of {trends['avg_monthly_change']:.1f}% "
                "month-over-month. Consider reviewing your spending habits."
            )
        
        # Check expense volatility
        if trends["volatility"] > 15:
            self.insight_categories["spending"].append(
                "Your monthly expenses show high volatility. Creating a budget could help stabilize spending."
            )
        
        # Analyze expense categories
        latest_expenses = expenses[-1] if expenses else {}
        total_expenses = sum(latest_expenses.values())
        
        for category, amount in latest_expenses.items():
            ratio = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            if ratio > 30:
                self.insight_categories["spending"].append(
                    f"Your {category} expenses represent {ratio:.1f}% of total spending. "
                    "This may be an area to optimize."
                )

    def _analyze_savings(self, user_data: Dict):
        """Analyze savings and emergency fund status."""
        monthly_income = user_data.get("monthly_income", 0)
        monthly_expenses = sum(user_data.get("expenses", [{}])[-1].values())
        emergency_fund = user_data.get("emergency_fund", 0)
        
        # Calculate savings rate
        savings_rate = calculate_monthly_savings_rate(monthly_income, monthly_expenses)
        
        if savings_rate < 20:
            self.insight_categories["savings"].append(
                f"Your current savings rate is {savings_rate:.1f}%. Consider aiming for at least 20%."
            )
        else:
            self.insight_categories["savings"].append(
                f"Great job! Your savings rate of {savings_rate:.1f}% is above the recommended minimum."
            )
        
        # Analyze emergency fund
        months_covered = emergency_fund / monthly_expenses if monthly_expenses > 0 else 0
        
        if months_covered < 3:
            self.insight_categories["savings"].append(
                f"Your emergency fund covers {months_covered:.1f} months of expenses. "
                "Aim for 3-6 months of coverage."
            )
        elif months_covered < 6:
            self.insight_categories["savings"].append(
                f"Your emergency fund covers {months_covered:.1f} months of expenses. "
                "You're on the right track!"
            )

    def _analyze_investments(self, user_data: Dict):
        """Analyze investment allocation and portfolio health."""
        investments = user_data.get("investments", {})
        age = user_data.get("age", 30)
        
        if not investments:
            return
        
        portfolio_value = calculate_portfolio_value(investments)
        allocation = calculate_asset_allocation(investments)
        
        # Check stock allocation based on age
        recommended_stock_allocation = 100 - age
        current_stock_allocation = allocation.get("stocks", 0)
        
        if abs(current_stock_allocation - recommended_stock_allocation) > 15:
            self.insight_categories["investments"].append(
                f"Your stock allocation ({current_stock_allocation:.1f}%) differs significantly from the "
                f"age-based recommendation of {recommended_stock_allocation}%."
            )
        
        # Check diversification
        for asset_class, percentage in allocation.items():
            if percentage > 50 and asset_class != "stocks":
                self.insight_categories["investments"].append(
                    f"Your portfolio has {percentage:.1f}% in {asset_class}. "
                    "Consider diversifying to reduce risk."
                )

    def _analyze_debt(self, user_data: Dict):
        """Analyze debt levels and suggest optimization strategies."""
        debts = user_data.get("debts", {})
        monthly_income = user_data.get("monthly_income", 0)
        
        if not debts:
            return
        
        debt_metrics = calculate_debt_metrics(debts)
        
        # Check high-interest debt
        if debt_metrics["high_interest_debt"] > 0:
            self.insight_categories["debt"].append(
                f"You have ${debt_metrics['high_interest_debt']:,.2f} in high-interest debt. "
                "Prioritize paying this off to save on interest."
            )
        
        # Analyze debt-to-income ratio
        if monthly_income > 0:
            dti_ratio = (debt_metrics["total_debt"] / (monthly_income * 12)) * 100
            if dti_ratio > 40:
                self.insight_categories["debt"].append(
                    f"Your debt-to-income ratio is {dti_ratio:.1f}%. "
                    "Consider debt consolidation or accelerated repayment strategies."
                )

    def _analyze_cashflow(self, user_data: Dict):
        """Analyze cash flow patterns and sustainability."""
        monthly_income = user_data.get("monthly_income", 0)
        expenses = user_data.get("expenses", [])
        
        if not expenses or monthly_income == 0:
            return
        
        recent_expenses = expenses[-3:]  # Last 3 months
        avg_monthly_expenses = sum(sum(month.values()) for month in recent_expenses) / len(recent_expenses)
        
        # Calculate monthly surplus/deficit
        monthly_surplus = monthly_income - avg_monthly_expenses
        
        if monthly_surplus < 0:
            self.insight_categories["cashflow"].append(
                f"You're averaging a monthly deficit of ${-monthly_surplus:,.2f}. "
                "Review expenses to identify areas for reduction."
            )
        elif monthly_surplus < (monthly_income * 0.2):
            self.insight_categories["cashflow"].append(
                "Your monthly surplus is less than 20% of income. "
                "Consider ways to increase your savings buffer."
            )

    def _analyze_goals(self, user_data: Dict):
        """Analyze progress towards financial goals."""
        goals = user_data.get("financial_goals", [])
        monthly_surplus = user_data.get("monthly_income", 0) - sum(user_data.get("expenses", [{}])[-1].values())
        
        for goal in goals:
            target_amount = goal.get("target_amount", 0)
            current_amount = goal.get("current_amount", 0)
            target_date = goal.get("target_date", "")
            
            if target_amount > 0:
                progress = (current_amount / target_amount) * 100
                self.insight_categories["goals"].append(
                    f"You're {progress:.1f}% of the way to your {goal.get('name', '')} goal. "
                    f"${target_amount - current_amount:,.2f} remaining."
                )
                
                # Check if monthly surplus is sufficient for goal
                if monthly_surplus > 0 and target_date:
                    # Simple calculation assuming target_date is in "YYYY-MM" format
                    try:
                        import datetime
                        target = datetime.datetime.strptime(target_date, "%Y-%m")
                        months_remaining = (target.year - datetime.datetime.now().year) * 12 + \
                                        (target.month - datetime.datetime.now().month)
                        
                        required_monthly = (target_amount - current_amount) / months_remaining
                        
                        if required_monthly > monthly_surplus:
                            self.insight_categories["goals"].append(
                                f"You need to save ${required_monthly:,.2f} monthly to reach your "
                                f"{goal.get('name', '')} goal. Consider adjusting the timeline or "
                                "increasing savings."
                            )
                    except ValueError:
                        pass 