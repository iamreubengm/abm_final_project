# utils/visualization.py
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

class FinancialVisualizer:
    """
    Utility class for creating financial data visualizations.
    
    This class handles creating interactive charts and graphs for various
    financial data types, such as budgets, investments, debt, and more.
    """
    
    def __init__(self, color_scheme: Optional[Dict[str, str]] = None):
        """
        Initialize the FinancialVisualizer.
        
        Args:
            color_scheme: Optional custom color scheme for visualizations
        """
        # Default color scheme
        self.color_scheme = color_scheme or {
            "primary": "#3366CC",
            "secondary": "#DC3912",
            "tertiary": "#FF9900",
            "quaternary": "#109618",
            "quinary": "#990099",
            "background": "#F8F9FA",
            "text": "#333333",
            "income": "#66BB6A",
            "expense": "#EF5350",
            "savings": "#42A5F5",
            "debt": "#FF7043",
            "investment": "#9575CD",
            "cash": "#4DD0E1",
            "stocks": "#4CAF50",
            "bonds": "#FFC107",
            "real_estate": "#9C27B0"
        }
        
        # Set Plotly template
        self.template = "plotly_white"
        
    def create_budget_chart(self, user_data: Dict) -> go.Figure:
        """
        Create a budget breakdown chart showing income and expenses.
        
        Args:
            user_data: Dictionary containing user's financial information
            
        Returns:
            Plotly figure with budget visualization
        """
        # Extract income and expenses
        income = user_data.get("income", {})
        expenses = user_data.get("expenses", {})
        
        # Prepare data for chart
        income_items = [{"category": k, "amount": v, "type": "Income"} for k, v in income.items() if v > 0]
        expense_items = [{"category": k, "amount": v, "type": "Expense"} for k, v in expenses.items() if v > 0]
        
        # Combine data
        data = income_items + expense_items
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create figure
        fig = px.bar(
            df,
            x="category",
            y="amount",
            color="type",
            barmode="group",
            color_discrete_map={"Income": self.color_scheme["income"], "Expense": self.color_scheme["expense"]},
            template=self.template,
            title="Monthly Budget Breakdown"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            legend_title="Type",
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_expense_pie_chart(self, expenses: Dict) -> go.Figure:
        """
        Create a pie chart showing expense distribution.
        
        Args:
            expenses: Dictionary with expense categories and amounts
            
        Returns:
            Plotly figure with expense distribution pie chart
        """
        # Filter out zero or negative values
        expenses = {k: v for k, v in expenses.items() if v > 0}
        
        # Prepare data for chart
        labels = list(expenses.keys())
        values = list(expenses.values())
        
        # Create figure
        fig = px.pie(
            names=labels,
            values=values,
            title="Monthly Expense Distribution",
            template=self.template,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Update layout
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            legend_title="Category"
        )
        
        # Update traces
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color=self.color_scheme["background"], width=2))
        )
        
        return fig
    
    def create_net_worth_chart(self, historical_net_worth: List[Dict]) -> go.Figure:
        """
        Create a line chart showing net worth over time.
        
        Args:
            historical_net_worth: List of dictionaries with date and net worth values
            
        Returns:
            Plotly figure with net worth trend
        """
        # Convert to DataFrame
        df = pd.DataFrame(historical_net_worth)
        
        # Ensure date is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        # Create figure
        fig = px.line(
            df,
            x="date",
            y="net_worth",
            markers=True,
            line_shape="spline",
            template=self.template,
            title="Net Worth Over Time"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Net Worth ($)",
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        # Update traces
        fig.update_traces(
            line=dict(color=self.color_scheme["primary"], width=3),
            marker=dict(color=self.color_scheme["primary"], size=8)
        )
        
        return fig
    
    def create_debt_payoff_chart(self, debt_projections: Dict) -> go.Figure:
        """
        Create a chart showing debt payoff projections.
        
        Args:
            debt_projections: Dictionary with dates and projected balances
            
        Returns:
            Plotly figure with debt payoff projection
        """
        # Extract data
        dates = list(debt_projections.keys())
        balances = list(debt_projections.values())
        
        # Convert dates to datetime
        dates = [datetime.strptime(date, "%Y-%m") for date in dates]
        
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=balances,
                mode="lines+markers",
                name="Debt Balance",
                line=dict(color=self.color_scheme["debt"], width=3),
                marker=dict(color=self.color_scheme["debt"], size=8)
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Debt Payoff Projection",
            xaxis_title="Date",
            yaxis_title="Remaining Balance ($)",
            template=self.template,
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_investment_allocation_chart(self, portfolio: Dict) -> go.Figure:
        """
        Create a chart showing investment portfolio allocation.
        
        Args:
            portfolio: Dictionary with investment portfolio data
            
        Returns:
            Plotly figure with investment allocation
        """
        # Extract asset allocation
        asset_allocation = portfolio.get("asset_allocation", {})
        
        # Prepare data for chart
        labels = list(asset_allocation.keys())
        values = list(asset_allocation.values())
        
        # Create color map
        colors = {
            "Stocks": self.color_scheme["stocks"],
            "Bonds": self.color_scheme["bonds"],
            "Cash": self.color_scheme["cash"],
            "Real Estate": self.color_scheme["real_estate"],
        }
        
        # Default color for other asset classes
        default_colors = [self.color_scheme["primary"], self.color_scheme["secondary"], 
                          self.color_scheme["tertiary"], self.color_scheme["quaternary"],
                          self.color_scheme["quinary"]]
        
        # Create color list
        color_list = []
        for label in labels:
            if label in colors:
                color_list.append(colors[label])
            else:
                # Assign default color if not in predefined colors
                color_list.append(default_colors[len(color_list) % len(default_colors)])
        
        # Create figure
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,
                    marker=dict(colors=color_list)
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title="Investment Portfolio Allocation",
            template=self.template,
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            legend_title="Asset Class"
        )
        
        # Update traces
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color=self.color_scheme["background"], width=2))
        )
        
        return fig
    
    def create_savings_goal_progress_chart(self, savings_goals: List[Dict]) -> go.Figure:
        """
        Create a chart showing progress toward savings goals.
        
        Args:
            savings_goals: List of dictionaries with savings goal data
            
        Returns:
            Plotly figure with savings goal progress
        """
        # Prepare data
        goals = []
        for goal in savings_goals:
            name = goal.get("name", "Unknown")
            target = goal.get("target", 0)
            current = goal.get("current", 0)
            deadline = goal.get("deadline", "")
            
            if target > 0:
                percent_complete = (current / target) * 100
            else:
                percent_complete = 0
            
            goals.append({
                "name": name,
                "percent_complete": percent_complete,
                "current": current,
                "target": target,
                "deadline": deadline
            })
        
        # Sort goals by completion percentage
        goals = sorted(goals, key=lambda x: x["percent_complete"])
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each goal
        for goal in goals:
            fig.add_trace(
                go.Bar(
                    x=[goal["percent_complete"]],
                    y=[goal["name"]],
                    orientation="h",
                    name=goal["name"],
                    text=f"{goal['percent_complete']:.1f}%",
                    textposition="auto",
                    marker=dict(color=self.color_scheme["savings"]),
                    hovertemplate=(
                        f"<b>{goal['name']}</b><br>" +
                        f"Progress: {goal['current']} / {goal['target']} " +
                        f"({goal['percent_complete']:.1f}%)<br>" +
                        f"Deadline: {goal['deadline']}"
                    )
                )
            )
        
        # Add 100% reference line
        fig.add_shape(
            type="line",
            x0=100,
            x1=100,
            y0=-0.5,
            y1=len(goals) - 0.5,
            line=dict(color="gray", width=2, dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            title="Savings Goal Progress",
            xaxis_title="Completion Percentage",
            xaxis=dict(range=[0, 110]),
            template=self.template,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_income_expense_trend_chart(self, monthly_data: List[Dict]) -> go.Figure:
        """
        Create a chart showing income and expense trends over time.
        
        Args:
            monthly_data: List of dictionaries with monthly financial data
            
        Returns:
            Plotly figure with income and expense trends
        """
        # Create DataFrame
        df = pd.DataFrame(monthly_data)
        
        # Ensure date is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        # Create figure
        fig = go.Figure()
        
        # Add income trace
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["income"],
                mode="lines+markers",
                name="Income",
                line=dict(color=self.color_scheme["income"], width=3),
                marker=dict(color=self.color_scheme["income"], size=8)
            )
        )
        
        # Add expense trace
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["expenses"],
                mode="lines+markers",
                name="Expenses",
                line=dict(color=self.color_scheme["expense"], width=3),
                marker=dict(color=self.color_scheme["expense"], size=8)
            )
        )
        
        # Add surplus/deficit area
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["income"] - df["expenses"],
                mode="lines",
                name="Surplus/Deficit",
                line=dict(color=self.color_scheme["primary"], width=0),
                fill="tozeroy",
                fillcolor=self.color_scheme["primary"] + "50"  # Add transparency
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Monthly Income and Expenses",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template=self.template,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_investment_performance_chart(self, performance_data: List[Dict]) -> go.Figure:
        """
        Create a chart showing investment performance over time.
        
        Args:
            performance_data: List of dictionaries with date and performance values
            
        Returns:
            Plotly figure with investment performance
        """
        # Create DataFrame
        df = pd.DataFrame(performance_data)
        
        # Ensure date is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        # Create figure
        fig = go.Figure()
        
        # Add portfolio return trace
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["portfolio_return"],
                mode="lines",
                name="Portfolio Return",
                line=dict(color=self.color_scheme["investment"], width=3),
            )
        )
        
        # Add benchmark return trace if available
        if "benchmark_return" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["benchmark_return"],
                    mode="lines",
                    name="Benchmark",
                    line=dict(color=self.color_scheme["secondary"], width=3, dash="dash"),
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Investment Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template=self.template,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_retirement_projection_chart(self, projection_data: Dict) -> go.Figure:
        """
        Create a chart showing retirement savings projections.
        
        Args:
            projection_data: Dictionary with projection scenarios
            
        Returns:
            Plotly figure with retirement projections
        """
        # Prepare data
        dates = projection_data.get("dates", [])
        baseline = projection_data.get("baseline", [])
        optimistic = projection_data.get("optimistic", [])
        conservative = projection_data.get("conservative", [])
        
        # Convert dates to datetime
        dates = [datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date for date in dates]
        
        # Create figure
        fig = go.Figure()
        
        # Add conservative scenario
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=conservative,
                name="Conservative",
                line=dict(color=self.color_scheme["tertiary"], width=2),
            )
        )
        
        # Add baseline scenario
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=baseline,
                name="Baseline",
                line=dict(color=self.color_scheme["primary"], width=3),
            )
        )
        
        # Add optimistic scenario
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=optimistic,
                name="Optimistic",
                line=dict(color=self.color_scheme["quaternary"], width=2),
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Retirement Savings Projection",
            xaxis_title="Year",
            yaxis_title="Projected Balance ($)",
            template=self.template,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_cash_flow_sankey(self, income: Dict, expenses: Dict) -> go.Figure:
        """
        Create a Sankey diagram showing cash flow from income to expenses.
        
        Args:
            income: Dictionary with income sources and amounts
            expenses: Dictionary with expense categories and amounts
            
        Returns:
            Plotly figure with cash flow Sankey diagram
        """
        # Prepare node labels
        income_labels = list(income.keys())
        expense_labels = list(expenses.keys())
        
        all_labels = income_labels + ["Total Income"] + expense_labels
        
        # Prepare source indices for links
        sources = []
        # Income sources to Total Income
        for i in range(len(income_labels)):
            sources.append(i)
        
        # Total Income to Expense categories
        for i in range(len(expense_labels)):
            sources.append(len(income_labels))
        
        # Prepare target indices for links
        targets = []
        # Income sources to Total Income
        for i in range(len(income_labels)):
            targets.append(len(income_labels))
        
        # Total Income to Expense categories
        for i in range(len(expense_labels)):
            targets.append(len(income_labels) + 1 + i)
        
        # Prepare values for links
        values = []
        # Income sources to Total Income
        for source, amount in income.items():
            values.append(amount)
        
        # Total Income to Expense categories
        for category, amount in expenses.items():
            values.append(amount)
        
        # Create color list
        node_colors = []
        # Income source colors
        for _ in range(len(income_labels)):
            node_colors.append(self.color_scheme["income"])
        
        # Total Income color
        node_colors.append(self.color_scheme["primary"])
        
        # Expense category colors
        for _ in range(len(expense_labels)):
            node_colors.append(self.color_scheme["expense"])
        
        # Create figure
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_labels,
                        color=node_colors
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values
                    )
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title="Monthly Cash Flow",
            template=self.template,
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"])
        )
        
        return fig
    
    def create_debt_comparison_chart(self, strategies: List[Dict]) -> go.Figure:
        """
        Create a chart comparing different debt repayment strategies.
        
        Args:
            strategies: List of dictionaries with debt repayment strategies
            
        Returns:
            Plotly figure with debt repayment strategy comparison
        """
        # Prepare data
        strategy_names = []
        payoff_times = []
        total_interests = []
        
        for strategy in strategies:
            strategy_names.append(strategy.get("name", "Unknown"))
            payoff_times.append(strategy.get("months_to_payoff", 0))
            total_interests.append(strategy.get("total_interest", 0))
        
        # Create figure
        fig = go.Figure()
        
        # Add payoff time trace
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=payoff_times,
                name="Months to Payoff",
                marker=dict(color=self.color_scheme["primary"]),
                yaxis="y"
            )
        )
        
        # Add total interest trace
        fig.add_trace(
            go.Bar(
                x=strategy_names,
                y=total_interests,
                name="Total Interest Paid",
                marker=dict(color=self.color_scheme["debt"]),
                yaxis="y2"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Debt Repayment Strategy Comparison",
            template=self.template,
            yaxis=dict(
                title="Months to Payoff",
                side="left"
            ),
            yaxis2=dict(
                title="Total Interest Paid ($)",
                side="right",
                overlaying="y",
                showgrid=False
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_spending_trend_chart(self, transactions: List[Dict]) -> go.Figure:
        """
        Create a chart showing spending trends by category over time.
        
        Args:
            transactions: List of transaction dictionaries with date, amount, category
            
        Returns:
            Plotly figure with spending trends
        """
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Ensure date is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Create month column
        df["month"] = df["date"].dt.to_period("M").astype(str)
        
        # Group by month and category
        monthly_spend = df.groupby(["month", "category"])["amount"].sum().reset_index()
        
        # Pivot table
        pivot_df = monthly_spend.pivot(index="month", columns="category", values="amount").fillna(0)
        
        # Sort by month
        pivot_df = pivot_df.sort_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each category
        for category in pivot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[category],
                    mode="lines+markers",
                    name=category,
                    stackgroup="one"
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Monthly Spending by Category",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            template=self.template,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        return fig
    
    def create_credit_score_chart(self, credit_history: List[Dict]) -> go.Figure:
        """
        Create a chart showing credit score history.
        
        Args:
            credit_history: List of dictionaries with date and score values
            
        Returns:
            Plotly figure with credit score history
        """
        # Create DataFrame
        df = pd.DataFrame(credit_history)
        
        # Ensure date is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values("date")
        
        # Create figure
        fig = go.Figure()
        
        # Add credit score trace
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["score"],
                mode="lines+markers",
                name="Credit Score",
                line=dict(color=self.color_scheme["primary"], width=3),
                marker=dict(color=self.color_scheme["primary"], size=8)
            )
        )
        
        # Add credit score ranges
        fig.add_shape(
            type="rect",
            x0=df["date"].min(),
            x1=df["date"].max(),
            y0=800,
            y1=850,
            fillcolor="green",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=df["date"].min(),
            x1=df["date"].max(),
            y0=740,
            y1=800,
            fillcolor="lightgreen",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=df["date"].min(),
            x1=df["date"].max(),
            y0=670,
            y1=740,
            fillcolor="yellow",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=df["date"].min(),
            x1=df["date"].max(),
            y0=580,
            y1=670,
            fillcolor="orange",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        fig.add_shape(
            type="rect",
            x0=df["date"].min(),
            x1=df["date"].max(),
            y0=300,
            y1=580,
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0
        )
        
        # Update layout
        fig.update_layout(
            title="Credit Score History",
            xaxis_title="Date",
            yaxis_title="Credit Score",
            yaxis=dict(range=[300, 850]),
            template=self.template,
            font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme["text"]),
            plot_bgcolor=self.color_scheme["background"]
        )
        
        # Add annotations for score ranges
        fig.add_annotation(
            x=df["date"].max(),
            y=825,
            text="Excellent (800-850)",
            showarrow=False,
            xshift=10,
            align="left"
        )
        
        fig.add_annotation(
            x=df["date"].max(),
            y=770,
            text="Very Good (740-799)",
            showarrow=False,
            xshift=10,
            align="left"
        )
        
        fig.add_annotation(
            x=df["date"].max(),
            y=705,
            text="Good (670-739)",
            showarrow=False,
            xshift=10,
            align="left"
        )
        
        fig.add_annotation(
            x=df["date"].max(),
            y=625,
            text="Fair (580-669)",
            showarrow=False,
            xshift=10,
            align="left"
        )
        
        fig.add_annotation(
            x=df["date"].max(),
            y=440,
            text="Poor (300-579)",
            showarrow=False,
            xshift=10,
            align="left"
        )
        
        return fig
    
    def generate_demo_monthly_data(self, months: int = 12) -> List[Dict]:
        """
        Generate demo data for monthly income and expenses.
        
        Args:
            months: Number of months of data to generate
            
        Returns:
            List of dictionaries with monthly financial data
        """
        # Start date (months ago)
        start_date = datetime.now() - timedelta(days=30 * months)
        
        # Base income and expense values
        base_income = 8000
        base_expenses = 6000
        
        # Create data
        monthly_data = []
        
        for i in range(months):
            # Calculate date
            date = start_date + timedelta(days=30 * i)
            
            # Add some randomness
            income_variation = np.random.normal(0, 500)
            expense_variation = np.random.normal(0, 300)
            
            # Add seasonal patterns
            month = date.month
            if month in [11, 12]:  # Holiday season
                expense_variation += 800
                income_variation += 1000 if month == 12 else 0  # Year-end bonus
            elif month in [6, 7, 8]:  # Summer
                expense_variation += 400
            
            # Calculate values
            income = max(0, base_income + income_variation)
            expenses = max(0, base_expenses + expense_variation)
            
            # Add data point
            monthly_data.append({
                "date": date,
                "income": income,
                "expenses": expenses,
                "savings": income - expenses
            })
        
        return monthly_data
    
    def generate_demo_portfolio_performance(self, years: int = 5) -> List[Dict]:
        """
        Generate demo data for investment portfolio performance.
        
        Args:
            years: Number of years of data to generate
            
        Returns:
            List of dictionaries with portfolio performance data
        """
        # Start date (years ago)
        start_date = datetime.now() - timedelta(days=365 * years)
        
        # Base return values
        portfolio_annual_return = 0.08  # 8% expected return
        benchmark_annual_return = 0.07  # 7% expected return
        
        # Create data
        performance_data = []
        
        # Initial values (100%)
        portfolio_value = 100
        benchmark_value = 100
        
        for i in range(years * 12):
            # Calculate date (monthly)
            date = start_date + timedelta(days=30 * i)
            
            # Add some randomness
            portfolio_monthly_return = (portfolio_annual_return / 12) + np.random.normal(0, 0.015)
            benchmark_monthly_return = (benchmark_annual_return / 12) + np.random.normal(0, 0.012)
            
            # Add some correlation between portfolio and benchmark
            correlation_factor = 0.7
            common_factor = np.random.normal(0, 0.01)
            portfolio_monthly_return += correlation_factor * common_factor
            benchmark_monthly_return += correlation_factor * common_factor
            
            # Add some cyclical patterns
            month = date.month
            if month in [1, 2]:  # January effect
                portfolio_monthly_return += 0.01
                benchmark_monthly_return += 0.01
            elif month in [5, 9, 10]:  # Historically weaker months
                portfolio_monthly_return -= 0.005
                benchmark_monthly_return -= 0.005
            
            # Update values
            portfolio_value *= (1 + portfolio_monthly_return)
            benchmark_value *= (1 + benchmark_monthly_return)
            
            # Add data point
            performance_data.append({
                "date": date,
                "portfolio_return": portfolio_value - 100,  # Cumulative return as percentage
                "benchmark_return": benchmark_value - 100
            })
        
        return performance_data
    
    def generate_demo_debt_projections(self, months: int = 36, initial_debt: float = 25000, 
                                       monthly_payment: float = 800, interest_rate: float = 0.15) -> Dict:
        """
        Generate demo data for debt payoff projections.
        
        Args:
            months: Number of months to project
            initial_debt: Initial debt balance
            monthly_payment: Monthly payment amount
            interest_rate: Annual interest rate (decimal)
            
        Returns:
            Dictionary with dates and projected balances
        """
        # Start date (current month)
        start_date = datetime.now().replace(day=1)
        
        # Monthly interest rate
        monthly_rate = interest_rate / 12
        
        # Create projections
        projections = {}
        
        # Initial balance
        balance = initial_debt
        
        for i in range(months):
            # Calculate date
            date = start_date + timedelta(days=30 * i)
            date_key = date.strftime("%Y-%m")
            
            # Calculate interest
            interest = balance * monthly_rate
            
            # Apply payment
            balance = balance + interest - monthly_payment
            
            # Ensure balance doesn't go below zero
            balance = max(0, balance)
            
            # Add to projections
            projections[date_key] = balance
            
            # Stop if debt is paid off
            if balance == 0:
                break
        
        return projections
    
    def generate_demo_credit_history(self, months: int = 24, initial_score: int = 680) -> List[Dict]:
        """
        Generate demo data for credit score history.
        
        Args:
            months: Number of months of history to generate
            initial_score: Initial credit score
            
        Returns:
            List of dictionaries with date and score values
        """
        # Start date (months ago)
        start_date = datetime.now() - timedelta(days=30 * months)
        
        # Create history
        credit_history = []
        
        # Current score
        score = initial_score
        
        for i in range(months):
            # Calculate date
            date = start_date + timedelta(days=30 * i)
            
            # Add some randomness with upward trend
            score_change = np.random.normal(1.5, 3)
            
            # Add some patterns
            month_number = i + 1
            
            # Simulate credit events
            if month_number == 3:
                # Simulate missed payment
                score_change = -35
            elif month_number == 7:
                # Simulate credit limit increase
                score_change = 15
            elif month_number == 12:
                # Simulate paying off a loan
                score_change = 25
            elif month_number == 18:
                # Simulate new credit account
                score_change = -10
            
            # Update score
            score += score_change
            
            # Ensure score stays in valid range
            score = max(300, min(850, score))
            
            # Add data point
            credit_history.append({
                "date": date,
                "score": int(score)
            })
        
        return credit_history
    
    def generate_demo_retirement_projection(self, years: int = 30) -> Dict:
        """
        Generate demo data for retirement savings projection.
        
        Args:
            years: Number of years to project
            
        Returns:
            Dictionary with projection scenarios
        """
        # Start date (current year)
        start_year = datetime.now().year
        
        # Create date list
        dates = [datetime(year=start_year + i, month=1, day=1) for i in range(years + 1)]
        
        # Initial balance
        initial_balance = 150000
        
        # Monthly contribution
        monthly_contribution = 1000
        
        # Annual returns for different scenarios
        baseline_return = 0.07
        optimistic_return = 0.09
        conservative_return = 0.05
        
        # Create projection data
        baseline = [initial_balance]
        optimistic = [initial_balance]
        conservative = [initial_balance]
        
        for i in range(1, years + 1):
            # Calculate balances for each scenario
            annual_contribution = monthly_contribution * 12
            
            # Baseline scenario
            baseline_balance = baseline[i-1] * (1 + baseline_return) + annual_contribution
            baseline.append(baseline_balance)
            
            # Optimistic scenario
            optimistic_balance = optimistic[i-1] * (1 + optimistic_return) + annual_contribution
            optimistic.append(optimistic_balance)
            
            # Conservative scenario
            conservative_balance = conservative[i-1] * (1 + conservative_return) + annual_contribution
            conservative.append(conservative_balance)
        
        return {
            "dates": dates,
            "baseline": baseline,
            "optimistic": optimistic,
            "conservative": conservative
        }

    
