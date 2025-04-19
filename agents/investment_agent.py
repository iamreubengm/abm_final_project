# agents/investment_agent.py
from typing import Dict, List, Any, Optional
import json

from anthropic import Anthropic
from config import DEFAULT_MODEL, SYSTEM_PROMPTS_PATH

class InvestmentAgent:
    """
    Specialized agent for investment advice, portfolio analysis, and wealth growth strategies.
    
    This agent helps users optimize their investments, understand risk/reward trade-offs,
    develop long-term growth strategies, and make informed investment decisions.
    """
    
    def __init__(self, client: Anthropic, knowledge_base=None):
        """
        Initialize the InvestmentAgent with an Anthropic client and knowledge base.
        
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
            with open(f"{SYSTEM_PROMPTS_PATH}/investment_agent.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback system prompt if file doesn't exist
            return """You are a specialized AI financial advisor focusing on investments and portfolio management.
            Your goal is to help users optimize their investments, understand investment options, 
            and develop effective long-term growth strategies.
            Provide balanced, thoughtful advice based on modern portfolio theory, considering risk tolerance,
            time horizon, and financial goals. Always acknowledge investment risks and
            avoid promising specific returns. Be specific and educational in your recommendations."""
    
    def get_advice(self, user_financial_data: Dict, user_query: str) -> str:
        """
        Generate investment advice based on user financial data and query.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            user_query: The user's specific question or request
            
        Returns:
            Personalized investment advice
        """
        # Get relevant knowledge base information if available
        context = ""
        if self.knowledge_base:
            context = self.knowledge_base.query("investment advice " + user_query)
        
        # Format the user's financial data for the prompt
        formatted_data = self._format_financial_data(user_financial_data)
        
        # Create the full prompt
        prompt = f"""
        USER QUERY: {user_query}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        {context}
        
        Based on this information, provide personalized investment advice to help the user.
        Focus specifically on investment strategies, portfolio composition, and long-term wealth growth.
        Be balanced in discussing risks and potential rewards, and avoid making specific return predictions.
        Provide educational context where helpful.
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
        # Extract relevant investment information
        investment_data = {
            "investments": user_financial_data.get("investments", {}),
            "retirement_accounts": user_financial_data.get("retirement_accounts", {}),
            "risk_tolerance": user_financial_data.get("risk_tolerance", "Unknown"),
            "time_horizon": user_financial_data.get("time_horizon", "Unknown"),
            "investment_goals": user_financial_data.get("investment_goals", []),
            "income": user_financial_data.get("income", {}),
            "age": user_financial_data.get("age", "Unknown")
        }
        
        # Format as a readable string
        return json.dumps(investment_data, indent=2)
    
    def analyze_portfolio(self, portfolio: Dict) -> Dict:
        """
        Analyze an investment portfolio and provide insights.
        
        Args:
            portfolio: Dictionary with investment holdings, allocations, and performance
            
        Returns:
            Analysis of portfolio composition, performance, and recommendations
        """
        # Format portfolio for the prompt
        formatted_portfolio = json.dumps(portfolio, indent=2)
        
        prompt = f"""
        Please analyze the following investment portfolio:
        
        {formatted_portfolio}
        
        Provide the following analysis:
        1. Asset allocation breakdown (percentage in each asset class)
        2. Sector exposure and concentration
        3. Geographic diversification
        4. Risk assessment (volatility, drawdown risk, etc.)
        5. Fee analysis
        6. Observations on portfolio construction and alignment with modern portfolio theory
        7. Specific recommendations for potential optimization
        
        Focus on educational insights that help the user understand their investments better.
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
            "portfolio_size": len(portfolio.get("holdings", [])),
            "total_value": sum(holding.get("value", 0) for holding in portfolio.get("holdings", []))
        }
    
    def recommend_investments(self, user_financial_data: Dict, investment_criteria: Dict) -> List[Dict]:
        """
        Recommend specific investments based on user criteria and financial situation.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            investment_criteria: Dictionary with criteria like risk level, time horizon, etc.
            
        Returns:
            List of investment recommendations with rationales
        """
        # Format financial data and criteria
        formatted_data = self._format_financial_data(user_financial_data)
        formatted_criteria = json.dumps(investment_criteria, indent=2)
        
        prompt = f"""
        Please recommend investments based on the following information:
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        INVESTMENT CRITERIA:
        {formatted_criteria}
        
        Provide 3-5 specific investment recommendations that:
        1. Match the user's risk tolerance and time horizon
        2. Align with their financial goals
        3. Consider their existing portfolio and diversification needs
        4. Represent different approaches or asset classes where appropriate
        
        For each recommendation, include:
        - Investment type/name (be general rather than recommending specific securities)
        - Allocation suggestion (percentage of investable assets)
        - Rationale
        - Potential risks and considerations
        - Time horizon appropriateness
        
        Focus on educational value and helping the user understand investment options.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1536,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        # This simplified version uses a single recommendation object with the full response
        return [{
            "recommendations": response.content,
            "criteria_matched": list(investment_criteria.keys())
        }]
    
    def explain_investment_concept(self, concept: str, complexity_level: str = "intermediate") -> str:
        """
        Provide an educational explanation of an investment concept.
        
        Args:
            concept: The investment concept to explain
            complexity_level: Desired complexity level (basic, intermediate, advanced)
            
        Returns:
            Educational explanation of the concept
        """
        # Get relevant knowledge base information if available
        context = ""
        if self.knowledge_base:
            context = self.knowledge_base.query(f"explain {concept} investment")
        
        prompt = f"""
        Please explain this investment concept: {concept}
        
        Complexity level: {complexity_level}
        
        {context}
        
        Provide an educational explanation that:
        1. Defines the concept clearly
        2. Explains why it matters to investors
        3. Provides relevant examples or applications
        4. Mentions any key considerations or limitations
        
        The explanation should be tailored to a {complexity_level} level of understanding.
        Focus on being educational and helpful to someone learning about investments.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def get_perspective(self, user_financial_data: Dict, topic: str) -> str:
        """
        Get this agent's perspective on a financial topic for debate.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            topic: The financial topic to provide perspective on
            
        Returns:
            Investment agent's perspective on the topic
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        As an investment and portfolio management expert, provide your professional perspective on this financial topic:
        
        TOPIC: {topic}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a thoughtful, nuanced perspective that:
        1. Emphasizes long-term investment considerations
        2. Considers risk/reward tradeoffs
        3. Addresses portfolio construction implications
        4. Offers practical recommendations from an investment perspective
        
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
            Investment agent's response for this debate round
        """
        prompt = f"""
        As an investment and portfolio management expert, respond to the other financial experts in this debate:
        
        TOPIC: {topic}
        
        DEBATE CONTEXT (WHAT OTHER EXPERTS HAVE SAID):
        {debate_context}
        
        This is round {round_num} of the debate. Please:
        1. Address key points raised by other experts
        2. Clarify or strengthen your position where needed
        3. Find areas of agreement while maintaining your investment expertise perspective
        4. Contribute new insights from an investment and long-term wealth building perspective
        
        Focus on how this topic specifically relates to investment strategy, portfolio construction, and long-term wealth growth.
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
        Generate multiple strategies to achieve a financial goal from an investment perspective.
        
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
        As an investment and portfolio management expert, generate {num_options} different strategies to achieve this financial goal:
        
        GOAL: {goal}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        For each strategy, provide:
        1. A clear name/title for the strategy
        2. A brief description (1-2 sentences)
        3. Specific investment approaches and asset allocations
        4. Estimated timeline and milestones
        5. Risk level and potential return characteristics
        
        Generate diverse strategies with different risk profiles, timeframes, and investment approaches.
        Focus on investment aspects but consider the whole financial picture.
        Be educational in explaining the rationale behind each strategy.
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
                "id": f"investment_strategy_{i+1}",
                "name": f"Investment Strategy Option {i+1}",
                "description": "Strategy description would be extracted from Claude response",
                "source": "investment_agent",
                "content": response.content,
                "goal": goal
            })
        
        return strategies
    
    def evaluate_strategy(self, strategy: Dict, user_financial_data: Dict, goal: str) -> Dict:
        """
        Evaluate a strategy from an investment perspective.
        
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
        As an investment and portfolio management expert, evaluate this financial strategy:
        
        GOAL: {goal}
        
        STRATEGY:
        {strategy_content}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide an evaluation that includes:
        1. How this strategy aligns with modern portfolio theory
        2. Strengths from an investment perspective
        3. Weaknesses or risks from an investment perspective
        4. A rating from 1-10 on how well this serves the user's investment needs
        5. Suggestions to improve the strategy from an investment standpoint
        
        Focus on long-term wealth building, risk-adjusted returns, and alignment with financial science.
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
            "source": "investment_agent",
            "strategy_id": strategy.get("id", "unknown")
        }
    
    def suggest_portfolio_rebalancing(self, current_portfolio: Dict, target_allocation: Dict) -> Dict:
        """
        Suggest portfolio rebalancing actions to align with target allocation.
        
        Args:
            current_portfolio: Current investment portfolio
            target_allocation: Target asset allocation percentages
            
        Returns:
            Rebalancing recommendations
        """
        # Format portfolio and target allocation
        formatted_portfolio = json.dumps(current_portfolio, indent=2)
        formatted_target = json.dumps(target_allocation, indent=2)
        
        prompt = f"""
        Please suggest portfolio rebalancing actions to align with the target allocation:
        
        CURRENT PORTFOLIO:
        {formatted_portfolio}
        
        TARGET ALLOCATION:
        {formatted_target}
        
        Provide specific recommendations for:
        1. Assets to reduce (specific holdings and approximate amounts)
        2. Assets to increase (specific asset classes and approximate amounts)
        3. Tax-efficient ways to implement these changes
        4. Priority order for making these adjustments
        5. Any considerations for minimizing transaction costs or tax impacts
        
        Focus on practical, actionable steps to move toward the target allocation.
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
            "rebalancing_recommendations": response.content,
            "current_total": sum(holding.get("value", 0) for holding in current_portfolio.get("holdings", [])),
            "target_allocation": target_allocation
        }
    
    def chat_response(self, user_query: str, user_financial_data: Dict, chat_history: List[Dict]) -> str:
        """
        Generate a conversational response to a user query about investments.
        
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
        
        Please respond to the user's query about investments and portfolio management.
        Be conversational but informative, and provide specific advice based on their financial data.
        Focus on educational content related to investments, risk management, and long-term wealth building.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content