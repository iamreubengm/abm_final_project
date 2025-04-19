# agents/savings_agent.py
from typing import Dict, List, Any, Optional
import json

from anthropic import Anthropic
from config import DEFAULT_MODEL, SYSTEM_PROMPTS_PATH

class SavingsAgent:
    """
    Specialized agent for savings strategies, goal planning, and emergency funds.
    
    This agent helps users set appropriate savings goals, create effective saving plans,
    optimize emergency funds, and develop strategies for specific savings objectives.
    """
    
    def __init__(self, client: Anthropic, knowledge_base=None):
        """
        Initialize the SavingsAgent with an Anthropic client and knowledge base.
        
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
            with open(f"{SYSTEM_PROMPTS_PATH}/savings_agent.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback system prompt if file doesn't exist
            return """You are a specialized AI financial advisor focusing on savings strategies and goal planning.
            Your goal is to help users set appropriate savings targets, create effective saving plans,
            optimize emergency funds, and develop strategies for specific savings objectives.
            Provide practical, actionable advice based on the user's financial situation and goals.
            Be specific and concrete in your recommendations."""
    
    def get_advice(self, user_financial_data: Dict, user_query: str) -> str:
        """
        Generate savings advice based on user financial data and query.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            user_query: The user's specific question or request
            
        Returns:
            Personalized savings advice
        """
        # Get relevant knowledge base information if available
        context = ""
        if self.knowledge_base:
            context = self.knowledge_base.query("savings strategies " + user_query)
        
        # Format the user's financial data for the prompt
        formatted_data = self._format_financial_data(user_financial_data)
        
        # Create the full prompt
        prompt = f"""
        USER QUERY: {user_query}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        {context}
        
        Based on this information, provide personalized savings advice to help the user.
        Focus specifically on savings strategies, goal planning, and emergency fund optimization.
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
        # Extract relevant savings information
        savings_data = {
            "income": user_financial_data.get("income", {}),
            "expenses": user_financial_data.get("expenses", {}),
            "monthly_cashflow": user_financial_data.get("monthly_cashflow", {}),
            "savings": user_financial_data.get("savings", {}),
            "savings_goals": user_financial_data.get("savings_goals", []),
            "emergency_fund": user_financial_data.get("emergency_fund", {})
        }
        
        # Format as a readable string
        return json.dumps(savings_data, indent=2)
    
    def analyze_savings_potential(self, income: Dict, expenses: Dict) -> Dict:
        """
        Analyze savings potential based on income and expenses.
        
        Args:
            income: Dictionary with income sources and amounts
            expenses: Dictionary with expense categories and amounts
            
        Returns:
            Analysis of savings potential with recommendations
        """
        # Format income and expenses for the prompt
        formatted_income = json.dumps(income, indent=2)
        formatted_expenses = json.dumps(expenses, indent=2)
        
        prompt = f"""
        Please analyze the following income and expenses to determine savings potential:
        
        INCOME:
        {formatted_income}
        
        EXPENSES:
        {formatted_expenses}
        
        Provide the following analysis:
        1. Current savings rate (percentage of income saved)
        2. Recommended savings rate based on financial best practices
        3. Specific expense categories that could be reduced to increase savings
        4. Potential strategies to increase income
        5. Automated savings recommendations
        6. Priority order for implementing changes
        
        Focus on practical, actionable recommendations that can help increase savings rate.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1536,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Calculate total income and expenses
        total_income = sum(amount for source, amount in income.items())
        total_expenses = sum(amount for category, amount in expenses.items())
        current_savings = total_income - total_expenses
        
        # In a real implementation, you would parse the response into structured data
        return {
            "analysis": response.content,
            "total_income": total_income,
            "total_expenses": total_expenses,
            "current_monthly_savings": current_savings,
            "annual_savings_potential": current_savings * 12
        }
    
    def create_savings_plan(self, user_financial_data: Dict, savings_goal: Dict) -> Dict:
        """
        Create a savings plan for a specific financial goal.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            savings_goal: Dictionary with goal details (target amount, deadline, purpose)
            
        Returns:
            Detailed savings plan with timeline and strategies
        """
        # Format financial data and goal
        formatted_data = self._format_financial_data(user_financial_data)
        formatted_goal = json.dumps(savings_goal, indent=2)
        
        prompt = f"""
        Please create a detailed savings plan for the following goal:
        
        SAVINGS GOAL:
        {formatted_goal}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Create a detailed, realistic savings plan that includes:
        1. Monthly savings target required to reach the goal
        2. Specific strategies to free up money for this goal
        3. Recommended savings vehicles or accounts
        4. Timeline with milestones
        5. Potential obstacles and how to overcome them
        6. Accountability mechanisms
        
        The plan should be practical and sustainable while working toward the stated goal.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1536,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        return {
            "savings_plan": response.content,
            "goal": savings_goal
        }
    
    def optimize_emergency_fund(self, user_financial_data: Dict) -> Dict:
        """
        Provide recommendations for optimizing emergency fund.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            
        Returns:
            Emergency fund recommendations
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Please analyze the user's financial situation and provide recommendations for optimizing their emergency fund:
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide specific recommendations that include:
        1. Ideal emergency fund target amount based on their situation
        2. Current emergency fund adequacy assessment
        3. Suggested timeline for building/maintaining the emergency fund
        4. Recommended savings vehicles for the emergency fund
        5. Strategies to balance emergency fund with other financial priorities
        6. When and how to use the emergency fund appropriately
        
        Focus on practical, actionable advice tailored to this user's specific situation.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Calculate monthly expenses for emergency fund context
        monthly_expenses = sum(amount for category, amount in user_financial_data.get("expenses", {}).items())
        
        # In a real implementation, you would parse the response into structured data
        return {
            "recommendations": response.content,
            "monthly_expenses": monthly_expenses,
            "current_emergency_fund": user_financial_data.get("emergency_fund", {}).get("balance", 0)
        }
    
    def get_perspective(self, user_financial_data: Dict, topic: str) -> str:
        """
        Get this agent's perspective on a financial topic for debate.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            topic: The financial topic to provide perspective on
            
        Returns:
            Savings agent's perspective on the topic
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        As a savings strategy and goal planning expert, provide your professional perspective on this financial topic:
        
        TOPIC: {topic}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a thoughtful, nuanced perspective that:
        1. Emphasizes savings and financial security considerations
        2. Highlights how this topic impacts long-term financial goals
        3. Considers emergency preparedness and financial stability
        4. Offers practical recommendations from a savings perspective
        
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
            Savings agent's response for this debate round
        """
        prompt = f"""
        As a savings strategy and goal planning expert, respond to the other financial experts in this debate:
        
        TOPIC: {topic}
        
        DEBATE CONTEXT (WHAT OTHER EXPERTS HAVE SAID):
        {debate_context}
        
        This is round {round_num} of the debate. Please:
        1. Address key points raised by other experts
        2. Clarify or strengthen your position where needed
        3. Find areas of agreement while maintaining your savings expertise perspective
        4. Contribute new insights from a savings and financial security perspective
        
        Focus on how this topic specifically relates to savings strategies, emergency preparedness, and goal planning.
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
        Generate multiple strategies to achieve a financial goal from a savings perspective.
        
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
        As a savings strategy and goal planning expert, generate {num_options} different strategies to achieve this financial goal:
        
        GOAL: {goal}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        For each strategy, provide:
        1. A clear name/title for the strategy
        2. A brief description (1-2 sentences)
        3. Specific action steps focused on savings and resource allocation
        4. Estimated timeline for implementation and results
        5. Level of lifestyle adjustment required
        
        Generate diverse strategies with different approaches, timeframes, or intensity levels.
        Focus on savings aspects but consider the whole financial picture.
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
                "id": f"savings_strategy_{i+1}",
                "name": f"Savings Strategy Option {i+1}",
                "description": "Strategy description would be extracted from Claude response",
                "source": "savings_agent",
                "content": response.content,
                "goal": goal
            })
        
        return strategies
    
    def evaluate_strategy(self, strategy: Dict, user_financial_data: Dict, goal: str) -> Dict:
        """
        Evaluate a strategy from a savings perspective.
        
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
        As a savings strategy and goal planning expert, evaluate this financial strategy:
        
        GOAL: {goal}
        
        STRATEGY:
        {strategy_content}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide an evaluation that includes:
        1. How this strategy impacts the user's savings rate and financial security
        2. Strengths from a savings perspective
        3. Weaknesses or risks from a savings perspective
        4. A rating from 1-10 on how well this serves the user's savings needs
        5. Suggestions to improve the strategy from a savings standpoint
        
        Focus on savings rate, goal achievement probability, and financial security.
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
            "source": "savings_agent",
            "strategy_id": strategy.get("id", "unknown")
        }
    
    def prioritize_savings_goals(self, goals: List[Dict], user_financial_data: Dict) -> Dict:
        """
        Prioritize multiple savings goals based on importance, timeline, and feasibility.
        
        Args:
            goals: List of savings goals with amounts, deadlines, and purposes
            user_financial_data: User's financial information
            
        Returns:
            Prioritized list with recommendations
        """
        # Format goals and financial data
        formatted_goals = json.dumps(goals, indent=2)
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Please help prioritize these savings goals based on importance, timeline, and feasibility:
        
        SAVINGS GOALS:
        {formatted_goals}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a prioritized plan that includes:
        1. Recommended priority order for these goals
        2. Rationale for the prioritization
        3. Suggested allocation of available savings for each goal
        4. Recommendations for adjusting goal amounts or timelines if needed
        5. Strategies for pursuing multiple goals simultaneously if possible
        
        Focus on creating a balanced approach that addresses critical needs first while making progress on all goals.
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
            "prioritized_plan": response.content,
            "goal_count": len(goals)
        }
    
    def chat_response(self, user_query: str, user_financial_data: Dict, chat_history: List[Dict]) -> str:
        """
        Generate a conversational response to a user query about savings.
        
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
        
        Please respond to the user's query about savings strategies and goal planning.
        Be conversational but informative, and provide specific advice based on their financial data.
        Focus on practical, actionable recommendations related to savings rates, emergency funds, and goal achievement.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content