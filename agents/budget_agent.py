# agents/budget_agent.py
from typing import Dict, List, Any, Optional
import json

from anthropic import Anthropic
from config import DEFAULT_MODEL, SYSTEM_PROMPTS_PATH

class BudgetAgent:
    """
    Specialized agent for budget planning, expense tracking, and cashflow management.
    
    This agent helps users optimize their spending, create effective budgets,
    identify areas for savings, and improve overall financial health.
    """
    
    def __init__(self, client: Anthropic, knowledge_base=None):
        """
        Initialize the BudgetAgent with an Anthropic client and knowledge base.
        
        Args:
            client: Anthropic API client
            knowledge_base: RAG knowledge base for financial information
        """
        self.client = client
        self.knowledge_base = knowledge_base
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt for this agent from file."""
        try:
            with open(f"{SYSTEM_PROMPTS_PATH}/budget_agent.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback system prompt if file doesn't exist
            return """You are a specialized AI financial advisor focusing on budget planning and expense management.
            Your goal is to help users optimize their spending, create effective budgets, and improve their financial health.
            Provide practical, actionable advice based on the user's financial data and goals.
            Always be specific and personalized in your recommendations."""
    
    def get_advice(self, user_financial_data: Dict, user_query: str) -> str:
        """
        Generate budget advice based on user financial data and query.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            user_query: The user's specific question or request
            
        Returns:
            Personalized budget advice
        """
        # Get relevant knowledge base information if available
        context = ""
        if self.knowledge_base:
            context = self.knowledge_base.query("budget planning " + user_query)
        
        # Format the user's financial data for the prompt
        formatted_data = self._format_financial_data(user_financial_data)
        
        # Create the full prompt
        prompt = f"""
        USER QUERY: {user_query}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        {context}
        
        Based on this information, provide personalized budget advice to help the user.
        Focus specifically on budgeting, expense management, and cashflow optimization.
        Be concrete and specific with your recommendations.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def _format_financial_data(self, user_financial_data: Dict) -> str:
        """Format financial data for inclusion in prompts."""
        # Extract relevant budget information
        budget_data = {
            "income": user_financial_data.get("income", {}),
            "expenses": user_financial_data.get("expenses", {}),
            "savings": user_financial_data.get("savings", {}),
            "monthly_cashflow": user_financial_data.get("monthly_cashflow", {})
        }
        
        # Format as a readable string
        # In a real implementation, you would do more sophisticated formatting
        return json.dumps(budget_data, indent=2)
    
    def analyze_spending(self, transactions: List[Dict]) -> Dict:
        """
        Analyze spending patterns and identify optimization opportunities.
        
        Args:
            transactions: List of transaction dictionaries with date, amount, category, description
            
        Returns:
            Analysis of spending patterns with recommendations
        """
        # Format transactions for the prompt
        formatted_transactions = json.dumps(transactions, indent=2)
        
        prompt = f"""
        Please analyze the following transactions:
        
        {formatted_transactions}
        
        Provide the following analysis:
        1. Spending breakdown by category (percentage of total)
        2. Month-over-month spending trends
        3. Identification of potential areas for savings
        4. Unusual or discretionary spending that could be reduced
        5. Specific recommendations for optimizing the budget
        
        Focus on actionable insights that can help improve financial health.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1536,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        # This simplified version just returns the raw text
        return {
            "analysis": response.content,
            "transaction_count": len(transactions)
        }
    
    def create_budget_plan(self, user_financial_data: Dict, budget_goals: List[str]) -> Dict:
        """
        Create a personalized budget plan based on income, expenses, and goals.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            budget_goals: List of budget goals (e.g., "save 20% of income", "reduce food spending")
            
        Returns:
            Detailed budget plan with category allocations
        """
        # Format financial data and goals
        formatted_data = self._format_financial_data(user_financial_data)
        formatted_goals = "\n".join([f"- {goal}" for goal in budget_goals])
        
        prompt = f"""
        Please create a personalized budget plan based on the following information:
        
        FINANCIAL DATA:
        {formatted_data}
        
        BUDGET GOALS:
        {formatted_goals}
        
        Create a detailed, realistic budget plan that:
        1. Allocates income to different spending categories
        2. Incorporates savings goals
        3. Provides specific dollar amounts for each category
        4. Balances needs, wants, and financial goals
        5. Accounts for irregular expenses
        
        The budget should be practical and sustainable while working toward the stated goals.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1536,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        # This simplified version just returns the raw text
        return {
            "budget_plan": response.content,
            "goals": budget_goals
        }
    
    def identify_savings_opportunities(self, expenses: List[Dict]) -> List[Dict]:
        """
        Identify specific opportunities to reduce expenses.
        
        Args:
            expenses: List of expense categories with amounts
            
        Returns:
            List of savings opportunities with potential impact
        """
        # Format expenses
        formatted_expenses = json.dumps(expenses, indent=2)
        
        prompt = f"""
        Please analyze these expense categories and identify specific savings opportunities:
        
        {formatted_expenses}
        
        For each opportunity, provide:
        1. The expense category
        2. Specific action to take
        3. Estimated monthly savings
        4. Difficulty level (easy, medium, hard)
        5. Impact on lifestyle (minimal, moderate, significant)
        
        Focus on practical, high-impact opportunities that would be realistic to implement.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        # This simplified version just returns the raw text as a list item
        return [{
            "opportunities": response.content,
            "expense_count": len(expenses)
        }]
    
    def get_perspective(self, user_financial_data: Dict, topic: str) -> str:
        """
        Get this agent's perspective on a financial topic for debate.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            topic: The financial topic to provide perspective on
            
        Returns:
            Budget agent's perspective on the topic
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        As a budget and cash flow management expert, provide your professional perspective on this financial topic:
        
        TOPIC: {topic}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a thoughtful, nuanced perspective that:
        1. Emphasizes cash flow management and budgeting considerations
        2. Highlights how this topic impacts day-to-day finances
        3. Considers short-term and long-term budget implications
        4. Offers practical recommendations from a budgeting perspective
        
        Your perspective should be balanced but focus on your area of expertise.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def respond_to_debate(self, debate_context: str, topic: str, round_num: int) -> str:
        """
        Respond to other agents in a debate.
        
        Args:
            debate_context: Context from previous debate rounds
            topic: The financial topic being debated
            round_num: Current round number
            
        Returns:
            Budget agent's response for this debate round
        """
        prompt = f"""
        As a budget and cash flow management expert, respond to the other financial experts in this debate:
        
        TOPIC: {topic}
        
        DEBATE CONTEXT (WHAT OTHER EXPERTS HAVE SAID):
        {debate_context}
        
        This is round {round_num} of the debate. Please:
        1. Address key points raised by other experts
        2. Clarify or strengthen your position where needed
        3. Find areas of agreement while maintaining your budgeting expertise perspective
        4. Contribute new insights from a cash flow and budgeting perspective
        
        Focus on how this topic specifically relates to budgeting, spending optimization, and day-to-day financial management.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def generate_strategies(self, user_financial_data: Dict, goal: str, num_options: int) -> List[Dict]:
        """
        Generate multiple strategies to achieve a financial goal from a budgeting perspective.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            goal: The financial goal to generate strategies for
            num_options: Number of strategy options to generate
            
        Returns:
            List of strategy dictionaries
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        As a budget and cash flow management expert, generate {num_options} different strategies to achieve this financial goal:
        
        GOAL: {goal}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        For each strategy, provide:
        1. A clear name/title for the strategy
        2. A brief description (1-2 sentences)
        3. Specific action steps focused on budgeting and cash flow management
        4. Estimated timeline
        5. Potential impact on the user's budget and finances
        
        Generate diverse strategies with different approaches, timeframes, or intensity levels.
        Focus on budgeting aspects but consider the whole financial picture.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1536,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        # This simplified version creates mock strategy objects
        strategies = []
        for i in range(num_options):
            strategies.append({
                "id": f"budget_strategy_{i+1}",
                "name": f"Budget Strategy Option {i+1}",
                "description": "Strategy description would be extracted from Claude response",
                "source": "budget_agent",
                "content": response.content,
                "goal": goal
            })
        
        return strategies
    
    def evaluate_strategy(self, strategy: Dict, user_financial_data: Dict, goal: str) -> Dict:
        """
        Evaluate a strategy from a budgeting perspective.
        
        Args:
            strategy: Strategy dictionary to evaluate
            user_financial_data: Dictionary containing user's financial information
            goal: The financial goal the strategy aims to achieve
            
        Returns:
            Evaluation dictionary with pros, cons, and rating
        """
        # Format inputs
        formatted_data = self._format_financial_data(user_financial_data)
        strategy_content = json.dumps(strategy, indent=2)
        
        prompt = f"""
        As a budget and cash flow management expert, evaluate this financial strategy:
        
        GOAL: {goal}
        
        STRATEGY:
        {strategy_content}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide an evaluation that includes:
        1. How this strategy impacts the user's budget and cash flow
        2. Strengths from a budgeting perspective
        3. Weaknesses or risks from a budgeting perspective
        4. A rating from 1-10 on how well this serves the user's budgeting needs
        5. Suggestions to improve the strategy from a budgeting standpoint
        
        Focus on practicality, sustainability, and alignment with healthy financial practices.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        return {
            "evaluation": response.content,
            "source": "budget_agent",
            "strategy_id": strategy.get("id", "unknown")
        }
    
    def chat_response(self, user_query: str, user_financial_data: Dict, chat_history: List[Dict]) -> str:
        """
        Generate a conversational response to a user query about budgeting.
        
        Args:
            user_query: User's question or request
            user_financial_data: User's financial data
            chat_history: List of previous chat messages
            
        Returns:
            Conversational response to the user query
        """
        # Format chat history
        formatted_history = ""
        for message in chat_history[-5:]:  # Include only the last 5 messages for context
            role = "User" if message.get("role") == "user" else "Assistant"
            formatted_history += f"{role}: {message.get('content', '')}\n\n"
        
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        CHAT HISTORY:
        {formatted_history}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        USER QUERY:
        {user_query}
        
        Please respond to the user's query about budgeting and financial management.
        Be conversational but informative, and provide specific advice based on their financial data.
        Focus on practical, actionable recommendations related to budgeting, spending, and cash flow.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content