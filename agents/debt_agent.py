# agents/debt_agent.py
from typing import Dict, List, Any, Optional
import json

from anthropic import Anthropic
from config import DEFAULT_MODEL, SYSTEM_PROMPTS_PATH

class DebtAgent:
    """
    Specialized agent for debt management, repayment strategies, and credit optimization.
    
    This agent helps users understand debt options, create effective repayment plans,
    improve credit profiles, and make informed borrowing decisions.
    """
    
    def __init__(self, client: Anthropic, knowledge_base=None):
        """
        Initialize the DebtAgent with an Anthropic client and knowledge base.
        
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
            with open(f"{SYSTEM_PROMPTS_PATH}/debt_agent.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback system prompt if file doesn't exist
            return """You are a specialized AI financial advisor focusing on debt management and credit optimization.
            Your goal is to help users understand their debt, create effective repayment strategies,
            improve their credit profiles, and make informed borrowing decisions.
            Provide practical, actionable advice based on mathematical optimization and the 
            psychological aspects of debt management. Be specific and educational in your recommendations."""
    
    def get_advice(self, user_financial_data: Dict, user_query: str) -> str:
        """
        Generate debt management advice based on user financial data and query.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            user_query: The user's specific question or request
            
        Returns:
            Personalized debt management advice
        """
        # Get relevant knowledge base information if available
        context = ""
        if self.knowledge_base:
            context = self.knowledge_base.query("debt management " + user_query)
        
        # Format the user's financial data for the prompt
        formatted_data = self._format_financial_data(user_financial_data)
        
        # Create the full prompt
        prompt = f"""
        USER QUERY: {user_query}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        {context}
        
        Based on this information, provide personalized debt management advice to help the user.
        Focus specifically on debt repayment strategies, credit optimization, and borrowing decisions.
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
        # Extract relevant debt information
        debt_data = {
            "debts": user_financial_data.get("debts", {}),
            "credit_score": user_financial_data.get("credit_score", "Unknown"),
            "income": user_financial_data.get("income", {}),
            "expenses": user_financial_data.get("expenses", {}),
            "monthly_cashflow": user_financial_data.get("monthly_cashflow", {})
        }
        
        # Format as a readable string
        return json.dumps(debt_data, indent=2)
    
    def analyze_debt_profile(self, debts: List[Dict]) -> Dict:
        """
        Analyze debt profile and identify optimization opportunities.
        
        Args:
            debts: List of debt dictionaries with balance, interest rate, payment, etc.
            
        Returns:
            Analysis of debt situation with recommendations
        """
        # Format debts for the prompt
        formatted_debts = json.dumps(debts, indent=2)
        
        prompt = f"""
        Please analyze the following debt profile:
        
        {formatted_debts}
        
        Provide the following analysis:
        1. Total debt burden and weighted average interest rate
        2. Debt-to-income ratio (assuming the user's income information is included)
        3. Monthly debt service amount and percentage of take-home pay
        4. Prioritization of debts for repayment (mathematical analysis)
        5. Potential refinancing or consolidation opportunities
        6. Credit utilization impact and considerations
        7. Specific recommendations for debt optimization
        
        Focus on actionable insights that can help improve the user's financial health.
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
            "total_debt": sum(debt.get("balance", 0) for debt in debts),
            "debt_count": len(debts)
        }
    
    def create_debt_repayment_plan(self, user_financial_data: Dict, strategy: str = "avalanche") -> Dict:
        """
        Create a personalized debt repayment plan based on financial situation.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            strategy: Repayment strategy - "avalanche" (highest interest first) or "snowball" (smallest balance first)
            
        Returns:
            Detailed debt repayment plan with timeline
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Please create a personalized debt repayment plan using the {strategy} method based on the following information:
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Create a detailed, realistic debt repayment plan that:
        1. Prioritizes debts according to the {strategy} method
        2. Specifies monthly payment amounts for each debt
        3. Provides a timeline for when each debt will be paid off
        4. Calculates total interest saved compared to minimum payments
        5. Considers the user's monthly cash flow and expenses
        
        The plan should be practical and sustainable while maximizing debt reduction efficiency.
        If the {strategy} method isn't optimal for this situation, explain why and suggest alternatives.
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
            "repayment_plan": response.content,
            "strategy": strategy
        }
    
    def evaluate_loan_option(self, loan_details: Dict, user_financial_data: Dict) -> Dict:
        """
        Evaluate a potential loan or refinancing option.
        
        Args:
            loan_details: Details of the loan being considered
            user_financial_data: User's financial information
            
        Returns:
            Evaluation of the loan option with recommendation
        """
        # Format loan details and financial data
        formatted_loan = json.dumps(loan_details, indent=2)
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Please evaluate this potential loan or refinancing option:
        
        LOAN DETAILS:
        {formatted_loan}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a comprehensive evaluation that includes:
        1. Total cost of the loan (principal + interest + fees)
        2. Impact on monthly cash flow
        3. Comparison to current debt situation (if refinancing)
        4. Affordability analysis
        5. Risk assessment
        6. Alternative options to consider
        7. Clear recommendation (proceed, negotiate better terms, or avoid)
        
        Focus on helping the user make an informed decision based on both math and practical considerations.
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
            "loan_amount": loan_details.get("amount", 0),
            "loan_type": loan_details.get("type", "Unknown")
        }
    
    def get_perspective(self, user_financial_data: Dict, topic: str) -> str:
        """
        Get this agent's perspective on a financial topic for debate.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            topic: The financial topic to provide perspective on
            
        Returns:
            Debt agent's perspective on the topic
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        As a debt management and credit optimization expert, provide your professional perspective on this financial topic:
        
        TOPIC: {topic}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a thoughtful, nuanced perspective that:
        1. Emphasizes debt management and credit considerations
        2. Highlights how this topic impacts borrowing capacity and costs
        3. Considers debt-to-income impacts and credit profile effects
        4. Offers practical recommendations from a debt management perspective
        
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
            Debt agent's response for this debate round
        """
        prompt = f"""
        As a debt management and credit optimization expert, respond to the other financial experts in this debate:
        
        TOPIC: {topic}
        
        DEBATE CONTEXT (WHAT OTHER EXPERTS HAVE SAID):
        {debate_context}
        
        This is round {round_num} of the debate. Please:
        1. Address key points raised by other experts
        2. Clarify or strengthen your position where needed
        3. Find areas of agreement while maintaining your debt expertise perspective
        4. Contribute new insights from a debt management and credit optimization perspective
        
        Focus on how this topic specifically relates to debt management, credit optimization, and borrowing decisions.
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
        Generate multiple strategies to achieve a financial goal from a debt management perspective.
        
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
        As a debt management and credit optimization expert, generate {num_options} different strategies to achieve this financial goal:
        
        GOAL: {goal}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        For each strategy, provide:
        1. A clear name/title for the strategy
        2. A brief description (1-2 sentences)
        3. Specific action steps related to debt management and credit optimization
        4. Estimated timeline for implementation and results
        5. Impact on the user's debt profile and credit score
        
        Generate diverse strategies with different approaches, timeframes, or intensity levels.
        Focus on debt management aspects but consider the whole financial picture.
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
                "id": f"debt_strategy_{i+1}",
                "name": f"Debt Strategy Option {i+1}",
                "description": "Strategy description would be extracted from Claude response",
                "source": "debt_agent",
                "content": response.content,
                "goal": goal
            })
        
        return strategies
    
    def evaluate_strategy(self, strategy: Dict, user_financial_data: Dict, goal: str) -> Dict:
        """
        Evaluate a strategy from a debt management perspective.
        
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
        As a debt management and credit optimization expert, evaluate this financial strategy:
        
        GOAL: {goal}
        
        STRATEGY:
        {strategy_content}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide an evaluation that includes:
        1. How this strategy impacts the user's debt profile and credit score
        2. Strengths from a debt management perspective
        3. Weaknesses or risks from a debt management perspective
        4. A rating from 1-10 on how well this serves the user's debt reduction needs
        5. Suggestions to improve the strategy from a debt management standpoint
        
        Focus on interest minimization, debt reduction efficiency, and credit score impact.
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
            "source": "debt_agent",
            "strategy_id": strategy.get("id", "unknown")
        }
    
    def analyze_credit_score(self, credit_report: Dict) -> Dict:
        """
        Analyze credit report and provide improvement recommendations.
        
        Args:
            credit_report: Dictionary with credit score and report details
            
        Returns:
            Analysis with improvement recommendations
        """
        # Format credit report
        formatted_report = json.dumps(credit_report, indent=2)
        
        prompt = f"""
        Please analyze this credit report and provide recommendations for improvement:
        
        {formatted_report}
        
        Provide a comprehensive analysis that includes:
        1. Key factors affecting the current credit score
        2. Specific actions that would improve the score
        3. Problematic items and how to address them
        4. Timeline for potential improvement
        5. Prioritized steps to take immediately
        
        Focus on practical, actionable steps the user can take to improve their credit profile.
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
            "analysis": response.content,
            "current_score": credit_report.get("score", "Unknown")
        }
    
    def chat_response(self, user_query: str, user_financial_data: Dict, chat_history: List[Dict]) -> str:
        """
        Generate a conversational response to a user query about debt management.
        
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
        
        Please respond to the user's query about debt management and credit optimization.
        Be conversational but informative, and provide specific advice based on their financial data.
        Focus on practical, actionable recommendations related to debt repayment, credit improvement, and borrowing decisions.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content