# agents/agent_manager.py
import json
from typing import Dict, List, Any, Optional

from config import get_anthropic_client, AGENT_TYPES, AGENT_INTERACTION_SETTINGS, DEFAULT_MODEL
from utils.rag_utils import FinancialRAG

class AgentManager:
    """
    Coordinates interactions between specialized financial agents and implements
    multi-agent design patterns like voting-based cooperation and debate-based cooperation.
    """
    
    def __init__(self, client=None, knowledge_base=None):
        """
        Initialize the AgentManager with an Anthropic client and specialized agents.
        
        Args:
            client: Anthropic API client. If None, a new client will be created.
            knowledge_base: RAG knowledge base for financial information.
        """
        self.client = client or get_anthropic_client()
        self.knowledge_base = knowledge_base
        self.agents = {}
        
        # Initialize each specialized agent
        self._initialize_agents()
        
        # Track user feedback for agent recommendations
        self.user_feedback = {}
    
    def _initialize_agents(self):
        """Initialize all specialized financial agents."""
        from agents.budget_agent import BudgetAgent
        from agents.investment_agent import InvestmentAgent
        from agents.debt_agent import DebtAgent
        from agents.savings_agent import SavingsAgent
        from agents.tax_agent import TaxAgent
        
        self.agents = {
            "budget": BudgetAgent(self.client, self.knowledge_base),
            "investment": InvestmentAgent(self.client, self.knowledge_base),
            "debt": DebtAgent(self.client, self.knowledge_base),
            "savings": SavingsAgent(self.client, self.knowledge_base),
            "tax": TaxAgent(self.client, self.knowledge_base)
        }
    
    def get_agent(self, agent_type: str):
        """Get a specific agent by type."""
        if agent_type not in self.agents:
            raise ValueError(f"Agent type '{agent_type}' not found.")
        return self.agents[agent_type]
    
    def get_holistic_advice(self, user_financial_data: Dict, user_query: str) -> Dict:
        """
        Get comprehensive financial advice by consulting all specialized agents.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            user_query: The user's specific question or request
            
        Returns:
            Dictionary with consolidated advice and agent-specific recommendations
        """
        # Get advice from each specialized agent
        agent_responses = {}
        for agent_type, agent in self.agents.items():
            agent_responses[agent_type] = agent.get_advice(user_financial_data, user_query)
        
        # Implement voting-based cooperation for consensus
        consensus = self._voting_cooperation(agent_responses, user_query)
        
        return {
            "consensus": consensus,
            "agent_responses": agent_responses
        }
    
    def _voting_cooperation(self, agent_responses: Dict, user_query: str) -> str:
        """
        Implement voting-based cooperation to reach consensus among agents.
        
        This pattern is useful for investment recommendations, budget allocations,
        and other scenarios where a consensus approach is valuable.
        
        Args:
            agent_responses: Dictionary of responses from each agent
            user_query: The original user query for context
            
        Returns:
            Consensus recommendation based on agent votes
        """
        threshold = AGENT_INTERACTION_SETTINGS["voting_threshold"]
        
        # Prepare prompt for consensus generation
        prompt = f"""
        I need to generate a consensus recommendation based on input from multiple financial expert agents.
        
        User Query: {user_query}
        
        Expert Agent Recommendations:
        """
        
        for agent_type, response in agent_responses.items():
            prompt += f"\n\n{agent_type.upper()} AGENT RECOMMENDATION:\n{response}"
        
        prompt += f"""
        
        Please analyze these recommendations and generate a consensus view that:
        1. Identifies points of agreement between at least {threshold * 100}% of the agents
        2. Highlights key recommendations that have strong support
        3. Notes any significant disagreements and explains the different perspectives
        4. Provides a balanced, integrated recommendation that considers all relevant advice
        
        The consensus should be comprehensive yet concise, focusing on actionable advice.
        """
        
        # Generate consensus using Claude
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system="You are a financial meta-advisor tasked with finding consensus among specialized financial experts.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def debate_based_cooperation(self, user_financial_data: Dict, topic: str, 
                              agent_types: List[str]) -> Dict:
        """
        Implement debate-based cooperation between agents to explore different perspectives.
        
        This pattern is useful for complex financial decisions with multiple valid approaches.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            topic: The financial topic to debate (e.g., "retirement investment strategy")
            agent_types: List of agent types to participate in the debate
            
        Returns:
            Dictionary with debate summary, key points, and final recommendation
        """
        rounds = AGENT_INTERACTION_SETTINGS["debate_rounds"]
        debate_history = []
        
        # Validate agent types
        for agent_type in agent_types:
            if agent_type not in self.agents:
                raise ValueError(f"Agent type '{agent_type}' not found.")
        
        # Initialize debate with each agent's perspective
        initial_perspectives = {}
        for agent_type in agent_types:
            agent = self.agents[agent_type]
            perspective = agent.get_perspective(user_financial_data, topic)
            initial_perspectives[agent_type] = perspective
            debate_history.append({
                "round": 0,
                "agent": agent_type,
                "content": perspective
            })
        
        # Conduct debate rounds
        for round_num in range(1, rounds + 1):
            for agent_type in agent_types:
                agent = self.agents[agent_type]
                
                # Prepare debate context from previous rounds
                debate_context = self._format_debate_context(debate_history, agent_type)
                
                # Get agent's response for this round
                response = agent.respond_to_debate(debate_context, topic, round_num)
                
                # Add to debate history
                debate_history.append({
                    "round": round_num,
                    "agent": agent_type,
                    "content": response
                })
        
        # Generate final summary and recommendation
        debate_summary = self._generate_debate_summary(debate_history, topic)
        
        return {
            "topic": topic,
            "debate_history": debate_history,
            "summary": debate_summary
        }
    
    def _format_debate_context(self, debate_history: List[Dict], current_agent: str) -> str:
        """Format the debate history as context for the next round."""
        context = "Previous debate contributions:\n\n"
        
        for entry in debate_history:
            agent_type = entry["agent"]
            if agent_type != current_agent:  # Only include other agents' perspectives
                context += f"ROUND {entry['round']} - {agent_type.upper()} AGENT:\n{entry['content']}\n\n"
        
        return context
    
    def _generate_debate_summary(self, debate_history: List[Dict], topic: str) -> str:
        """Generate a summary of the debate with key points and final recommendation."""
        # Format debate history
        formatted_debate = f"TOPIC: {topic}\n\n"
        
        for entry in debate_history:
            formatted_debate += f"ROUND {entry['round']} - {entry['agent'].upper()} AGENT:\n{entry['content']}\n\n"
        
        prompt = f"""
        Please analyze this debate between financial expert agents and provide:
        1. A summary of the key points made by each agent
        2. Areas of agreement and disagreement
        3. A balanced final recommendation that integrates the strongest arguments
        
        Debate transcript:
        {formatted_debate}
        """
        
        # Generate summary using Claude
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system="You are a financial meta-advisor tasked with summarizing debates between specialized financial experts.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content
    
    def multi_path_plan_generator(self, user_financial_data: Dict, goal: str) -> Dict:
        """
        Generate multiple alternative financial strategies to achieve a goal.
        
        This pattern is useful for presenting users with different approaches based on
        their risk tolerance, timeline, and priorities.
        
        Args:
            user_financial_data: Dictionary containing user's financial information
            goal: The financial goal (e.g., "save for house down payment")
            
        Returns:
            Dictionary with multiple strategy paths and their pros/cons
        """
        num_options = AGENT_INTERACTION_SETTINGS["multi_path_options"]
        
        # Identify which agents are relevant for this goal
        relevant_agents = self._identify_relevant_agents_for_goal(goal)
        
        # Get strategy suggestions from relevant agents
        agent_strategies = {}
        for agent_type in relevant_agents:
            agent = self.agents[agent_type]
            strategies = agent.generate_strategies(user_financial_data, goal, num_options)
            agent_strategies[agent_type] = strategies
        
        # Consolidate and diversify strategies
        consolidated_strategies = self._consolidate_strategies(agent_strategies, num_options)
        
        # Evaluate each strategy with all relevant agents
        evaluated_strategies = self._evaluate_strategies(consolidated_strategies, user_financial_data, goal, relevant_agents)
        
        return {
            "goal": goal,
            "strategies": evaluated_strategies
        }
    
    def _identify_relevant_agents_for_goal(self, goal: str) -> List[str]:
        """Identify which agents are most relevant for a specific financial goal."""
        # Map common goals to relevant agent types
        goal_to_agents = {
            "retirement": ["investment", "tax", "savings"],
            "house": ["savings", "debt", "budget"],
            "debt_payoff": ["debt", "budget"],
            "emergency_fund": ["savings", "budget"],
            "education": ["savings", "investment", "tax"],
            "budget": ["budget"],
            "investment": ["investment", "tax"]
        }
        
        # Find the best match for the goal
        for key, agents in goal_to_agents.items():
            if key in goal.lower():
                return agents
        
        # Default to all agents if no specific match
        return list(self.agents.keys())
    
    def _consolidate_strategies(self, agent_strategies: Dict, num_options: int) -> List[Dict]:
        """Consolidate and diversify strategies from different agents."""
        all_strategies = []
        
        # Collect all strategies
        for agent_type, strategies in agent_strategies.items():
            for strategy in strategies:
                strategy["source_agent"] = agent_type
                all_strategies.append(strategy)
        
        # Use Claude to select diverse strategies
        if len(all_strategies) > num_options:
            prompt = f"""
            I have {len(all_strategies)} financial strategies for a user's goal. Please select the {num_options} most diverse 
            and complementary approaches that provide different risk levels and approaches.
            
            The strategies are:
            
            {json.dumps(all_strategies, indent=2)}
            
            Please return the indices of the {num_options} most diverse strategies, with a brief explanation of why each was selected.
            """
            
            response = self.client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=1024,
                system="You are a financial meta-advisor tasked with selecting diverse financial strategies.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Process response to extract strategy indices
            # This is a simplified implementation and might need refinement
            selected_indices = self._extract_indices_from_response(response.content, len(all_strategies), num_options)
            consolidated_strategies = [all_strategies[i] for i in selected_indices]
        else:
            consolidated_strategies = all_strategies
        
        return consolidated_strategies
    
    def _extract_indices_from_response(self, response: str, max_index: int, num_expected: int) -> List[int]:
        """Extract strategy indices from Claude's response."""
        # This is a simplified implementation
        # In a real implementation, you would use regex or more sophisticated parsing
        try:
            # Try to find numbers in the response
            indices = []
            for word in response.split():
                if word.isdigit() and int(word) < max_index:
                    indices.append(int(word))
            
            # Deduplicate and limit to expected number
            indices = list(set(indices))[:num_expected]
            
            # If we didn't find enough indices, add sequential ones up to max_index
            while len(indices) < num_expected and len(indices) < max_index:
                for i in range(max_index):
                    if i not in indices:
                        indices.append(i)
                        break
                        
            return indices[:num_expected]
        except:
            # Fallback to sequential indices if parsing fails
            return list(range(min(num_expected, max_index)))
    
    def _evaluate_strategies(self, strategies: List[Dict], user_financial_data: Dict, 
                             goal: str, agent_types: List[str]) -> List[Dict]:
        """Have each relevant agent evaluate all the strategies."""
        evaluated_strategies = []
        
        for strategy in strategies:
            evaluations = {}
            
            for agent_type in agent_types:
                agent = self.agents[agent_type]
                evaluation = agent.evaluate_strategy(strategy, user_financial_data, goal)
                evaluations[agent_type] = evaluation
            
            # Add evaluations to the strategy
            strategy["evaluations"] = evaluations
            
            # Generate a balanced pros/cons summary
            strategy["analysis"] = self._generate_strategy_analysis(strategy, evaluations, goal)
            
            evaluated_strategies.append(strategy)
        
        return evaluated_strategies
    
    def _generate_strategy_analysis(self, strategy: Dict, evaluations: Dict, goal: str) -> Dict:
        """Generate a balanced analysis of a strategy's pros and cons."""
        prompt = f"""
        Please analyze this financial strategy for the goal: {goal}
        
        Strategy: {json.dumps(strategy, indent=2)}
        
        Expert evaluations: {json.dumps(evaluations, indent=2)}
        
        Provide a balanced analysis with:
        1. Top 3 pros of this strategy
        2. Top 3 cons or risks
        3. Who this strategy is most suitable for (risk profile, timeline, etc.)
        4. A brief summary (2-3 sentences) of the overall approach
        """
        
        response = self.client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system="You are a financial meta-advisor tasked with analyzing financial strategies.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # In a real implementation, you would parse the response into structured data
        # This simplified version just returns the raw text
        return response.content
    
    def incorporate_human_feedback(self, strategy_id: str, feedback: Dict) -> None:
        """
        Incorporate human feedback to improve future recommendations.
        
        Args:
            strategy_id: Identifier for the strategy being rated
            feedback: Dictionary containing rating and comments
        """
        # Store feedback for learning
        self.user_feedback[strategy_id] = feedback
    
    def get_agent_chat_response(self, agent_type: str, user_query: str, 
                              user_financial_data: Dict, chat_history: List[Dict]) -> str:
        """
        Get a response from a specific agent in the chat interface.
        
        Args:
            agent_type: Type of agent to respond
            user_query: User's question or request
            user_financial_data: User's financial data
            chat_history: List of previous chat messages
            
        Returns:
            Agent's response to the user query
        """
        if agent_type not in self.agents:
            raise ValueError(f"Agent type '{agent_type}' not found.")
        
        agent = self.agents[agent_type]
        return agent.chat_response(user_query, user_financial_data, chat_history)