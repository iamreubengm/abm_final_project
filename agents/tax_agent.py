# agents/tax_agent.py
from typing import Dict, List, Any, Optional
import json

from anthropic import Anthropic
from config import DEFAULT_MODEL, SYSTEM_PROMPTS_PATH

class TaxAgent:
    """
    Specialized agent for tax optimization, planning, and compliance strategies.
    
    This agent helps users understand tax implications of financial decisions,
    optimize tax strategies, plan for tax events, and navigate tax-advantaged accounts.
    """
    
    def __init__(self, client: Anthropic, knowledge_base=None):
        """
        Initialize the TaxAgent with an Anthropic client and knowledge base.
        
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
            with open(f"{SYSTEM_PROMPTS_PATH}/tax_agent.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            # Fallback system prompt if file doesn't exist
            return """You are a specialized AI financial advisor focusing on tax optimization and planning.
            Your goal is to help users understand the tax implications of financial decisions,
            optimize their tax strategies, plan for tax events, and navigate tax-advantaged accounts.
            Provide practical, actionable advice based on the user's financial situation and goals.
            Always emphasize that you're providing general tax information and not legal tax advice,
            and recommend consulting with a tax professional for specific situations."""
    
    def get_advice(self, user_financial_data: Dict, user_query: str) -> str:
        """
        Generate tax advice based on user financial data and query.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            user_query: The user's specific question or request
            
        Returns:
            Personalized tax advice with appropriate disclaimers
        """
        # Get relevant knowledge base information if available
        context = ""
        if self.knowledge_base:
            context = self.knowledge_base.query("tax planning " + user_query)
        
        # Format the user's financial data for the prompt
        formatted_data = self._format_financial_data(user_financial_data)
        
        # Create the full prompt
        prompt = f"""
        USER QUERY: {user_query}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        {context}
        
        Based on this information, provide personalized tax planning advice to help the user.
        Focus specifically on tax optimization strategies, tax-advantaged accounts, and tax planning.
        Be concrete and specific with your recommendations while including appropriate disclaimers
        about consulting with a tax professional.
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
        # Extract relevant tax information
        tax_data = {
            "income": user_financial_data.get("income", {}),
            "filing_status": user_financial_data.get("filing_status", "Unknown"),
            "dependents": user_financial_data.get("dependents", 0),
            "retirement_accounts": user_financial_data.get("retirement_accounts", {}),
            "tax_deductions": user_financial_data.get("tax_deductions", {}),
            "tax_credits": user_financial_data.get("tax_credits", {}),
            "investments": user_financial_data.get("investments", {}),
            "business_income": user_financial_data.get("business_income", {})
        }
        
        # Format as a readable string
        return json.dumps(tax_data, indent=2)
    
    def analyze_tax_situation(self, user_financial_data: Dict) -> Dict:
        """
        Analyze tax situation and identify optimization opportunities.
        
        Args:
            user_financial_data: Dictionary with income, deductions, filing status, etc.
            
        Returns:
            Analysis of tax situation with recommendations
        """
        # Format financial data for the prompt
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Please analyze the following tax situation:
        
        {formatted_data}
        
        Provide the following analysis:
        1. Estimated tax liability based on current information
        2. Potential tax deductions that may be underutilized
        3. Tax credits the user may qualify for
        4. Retirement account contribution opportunities
        5. Tax-loss harvesting possibilities (if investment data is available)
        6. Other tax optimization strategies tailored to their situation
        7. Recommendations prioritized by potential tax savings
        
        Focus on actionable insights that can help reduce tax liability legally and appropriately.
        Include appropriate disclaimers about consulting with a tax professional.
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
            "filing_status": user_financial_data.get("filing_status", "Unknown")
        }
    
    def recommend_tax_advantaged_accounts(self, user_financial_data: Dict) -> Dict:
        """
        Recommend tax-advantaged accounts based on financial situation.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            
        Returns:
            Recommendations for tax-advantaged accounts with rationales
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Based on the user's financial situation, please recommend appropriate tax-advantaged accounts:
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide specific recommendations for:
        1. Retirement accounts (e.g., Traditional IRA, Roth IRA, 401(k), SEP IRA, Solo 401(k))
        2. Health-related accounts (e.g., HSA, FSA)
        3. Education accounts (e.g., 529 plans, Coverdell ESAs)
        4. Other tax-advantaged vehicles relevant to their situation
        
        For each recommendation, include:
        - Eligibility assessment
        - Contribution limits based on their situation
        - Tax benefits specific to their tax bracket
        - Prioritization guidance
        - Implementation steps
        
        Focus on practical, actionable recommendations tailored to this specific financial situation.
        Include appropriate disclaimers about consulting with a tax professional.
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
            "recommendations": response.content,
            "income_level": self._categorize_income_level(user_financial_data)
        }
    
    def _categorize_income_level(self, user_financial_data: Dict) -> str:
        """Categorize income level for context."""
        total_income = sum(amount for source, amount in user_financial_data.get("income", {}).items())
        
        if total_income < 50000:
            return "low"
        elif total_income < 100000:
            return "moderate"
        elif total_income < 200000:
            return "high"
        else:
            return "very high"
    
    def estimate_tax_liability(self, income_data: Dict, deductions: Dict, credits: Dict, filing_status: str) -> Dict:
        """
        Estimate tax liability based on financial information.
        
        Args:
            income_data: Dictionary with income sources and amounts
            deductions: Dictionary with deduction categories and amounts
            credits: Dictionary with tax credits and amounts
            filing_status: Tax filing status
            
        Returns:
            Estimated tax liability with breakdown
        """
        # Format inputs for the prompt
        formatted_income = json.dumps(income_data, indent=2)
        formatted_deductions = json.dumps(deductions, indent=2)
        formatted_credits = json.dumps(credits, indent=2)
        
        prompt = f"""
        Please estimate tax liability based on the following information:
        
        INCOME:
        {formatted_income}
        
        DEDUCTIONS:
        {formatted_deductions}
        
        CREDITS:
        {formatted_credits}
        
        FILING STATUS:
        {filing_status}
        
        Provide a detailed estimate that includes:
        1. Gross income calculation
        2. Adjusted Gross Income (AGI) after applicable adjustments
        3. Standard or itemized deduction recommendation (whichever is higher)
        4. Taxable income calculation
        5. Federal income tax estimate with tax bracket breakdown
        6. Self-employment tax estimate (if applicable)
        7. Estimated tax credits
        8. Final estimated tax liability
        9. Effective tax rate
        
        Note: This is an estimate based on general tax principles and the limited information provided.
        Include appropriate disclaimers about consulting with a tax professional.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Calculate total income and deductions for context
        total_income = sum(amount for source, amount in income_data.items())
        total_deductions = sum(amount for category, amount in deductions.items())
        total_credits = sum(amount for credit, amount in credits.items())
        
        # In a real implementation, you would parse the response into structured data
        return {
            "tax_estimate": response.content,
            "total_income": total_income,
            "total_deductions": total_deductions,
            "total_credits": total_credits,
            "filing_status": filing_status
        }
    
    def get_perspective(self, user_financial_data: Dict, topic: str) -> str:
        """
        Get this agent's perspective on a financial topic for debate.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            topic: The financial topic to provide perspective on
            
        Returns:
            Tax agent's perspective on the topic
        """
        # Format financial data
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        As a tax optimization and planning expert, provide your professional perspective on this financial topic:
        
        TOPIC: {topic}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a thoughtful, nuanced perspective that:
        1. Emphasizes tax implications and considerations
        2. Highlights how this topic impacts tax planning and optimization
        3. Considers both short-term and long-term tax consequences
        4. Offers practical recommendations from a tax efficiency perspective
        
        Your perspective should be balanced but focus on your area of expertise.
        Include appropriate disclaimers about consulting with a tax professional.
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
            Tax agent's response for this debate round
        """
        prompt = f"""
        As a tax optimization and planning expert, respond to the other financial experts in this debate:
        
        TOPIC: {topic}
        
        DEBATE CONTEXT (WHAT OTHER EXPERTS HAVE SAID):
        {debate_context}
        
        This is round {round_num} of the debate. Please:
        1. Address key points raised by other experts
        2. Clarify or strengthen your position where needed
        3. Find areas of agreement while maintaining your tax expertise perspective
        4. Contribute new insights from a tax efficiency and planning perspective
        
        Focus on how this topic specifically relates to tax implications, tax-advantaged strategies, and tax planning.
        Include appropriate disclaimers about consulting with a tax professional when necessary.
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
        Generate multiple strategies to achieve a financial goal from a tax perspective.
        
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
        As a tax optimization and planning expert, generate {num_options} different strategies to achieve this financial goal:
        
        GOAL: {goal}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        For each strategy, provide:
        1. A clear name/title for the strategy
        2. A brief description (1-2 sentences)
        3. Specific action steps focused on tax optimization and planning
        4. Estimated tax implications and benefits
        5. Implementation timeline and considerations
        
        Generate diverse strategies with different approaches to tax efficiency.
        Focus on tax aspects but consider the whole financial picture.
        Include appropriate disclaimers about consulting with a tax professional.
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
                "id": f"tax_strategy_{i+1}",
                "name": f"Tax Strategy Option {i+1}",
                "description": "Strategy description would be extracted from Claude response",
                "source": "tax_agent",
                "content": response.content,
                "goal": goal
            })
        
        return strategies
    
    def evaluate_strategy(self, strategy: Dict, user_financial_data: Dict, goal: str) -> Dict:
        """
        Evaluate a strategy from a tax perspective.
        
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
        As a tax optimization and planning expert, evaluate this financial strategy:
        
        GOAL: {goal}
        
        STRATEGY:
        {strategy_content}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide an evaluation that includes:
        1. How this strategy impacts the user's tax situation
        2. Strengths from a tax perspective
        3. Weaknesses or risks from a tax perspective
        4. A rating from 1-10 on how well this serves the user's tax optimization needs
        5. Suggestions to improve the strategy from a tax standpoint
        
        Focus on tax efficiency, compliance, and long-term tax planning.
        Include appropriate disclaimers about consulting with a tax professional.
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
            "source": "tax_agent",
            "strategy_id": strategy.get("id", "unknown")
        }
    
    def analyze_tax_implications(self, financial_decision: Dict, user_financial_data: Dict) -> Dict:
        """
        Analyze tax implications of a specific financial decision.
        
        Args:
            financial_decision: Details of the financial decision being considered
            user_financial_data: User's financial information
            
        Returns:
            Analysis of tax implications with recommendations
        """
        # Format decision and financial data
        formatted_decision = json.dumps(financial_decision, indent=2)
        formatted_data = self._format_financial_data(user_financial_data)
        
        prompt = f"""
        Please analyze the tax implications of this financial decision:
        
        FINANCIAL DECISION:
        {formatted_decision}
        
        USER FINANCIAL DATA:
        {formatted_data}
        
        Provide a comprehensive analysis that includes:
        1. Immediate tax implications (current tax year)
        2. Long-term tax implications (future tax years)
        3. Alternative approaches with better tax outcomes
        4. Specific tax forms or schedules impacted
        5. Documentation requirements for tax compliance
        6. Strategies to minimize negative tax consequences
        7. Overall tax efficiency assessment
        
        Focus on providing actionable insights to help the user make a tax-efficient decision.
        Include appropriate disclaimers about consulting with a tax professional.
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
            "tax_analysis": response.content,
            "decision_type": financial_decision.get("type", "Unknown")
        }
    
    def chat_response(self, user_query: str, user_financial_data: Dict, chat_history: List[Dict]) -> str:
        """
        Generate a conversational response to a user query about taxes.
        
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
        
        Please respond to the user's query about tax optimization and planning.
        Be conversational but informative, and provide specific advice based on their financial data.
        Focus on practical, actionable recommendations related to tax efficiency and planning.
        Include appropriate disclaimers about consulting with a tax professional when necessary.
        """
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content