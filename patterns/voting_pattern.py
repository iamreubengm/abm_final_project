from typing import List, Dict, Any, Optional, Union, Callable
import json
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from utils.api_client import ClaudeAPIClient

logger = logging.getLogger(__name__)

class VotingPattern:
    """
    Implementation of the Voting-Based Cooperation pattern.
    
    This pattern enables multiple agents to vote on options or solutions,
    with each agent providing its perspective based on its specialization.
    """
    
    def __init__(self, 
                coordinator_system_prompt: Optional[str] = None,
                api_client: Optional[ClaudeAPIClient] = None):
        """
        Initialize the voting pattern.
        
        Args:
            coordinator_system_prompt: Optional system prompt for the coordinator agent
            api_client: Claude API client. If not provided, a new one will be created.
        """
        self.api_client = api_client or ClaudeAPIClient()
        
        # Define coordinator system prompt if not provided
        if coordinator_system_prompt is None:
            coordinator_system_prompt = (
                "You are a neutral Coordinator responsible for managing a voting process between multiple AI agents. "
                "Your role is to collect votes, tally them, and determine the final decision. "
                "You must be impartial and objective, giving equal weight to each agent's vote unless specifically instructed otherwise. "
                "You should not insert your own preferences or biases into the voting process. "
                "Present the results clearly, including vote counts and the rationale provided by each agent. "
                "If there is a tie, use the confidence scores provided by the agents to break it."
            )
        
        self.coordinator_system_prompt = coordinator_system_prompt
        logger.info("Initialized Voting Pattern")
    
    def conduct_vote(self, 
                   agents: List[BaseAgent], 
                   question: str, 
                   options: Optional[List[str]] = None,
                   explanation_required: bool = True,
                   confidence_required: bool = True,
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Conduct a vote among multiple agents.
        
        Args:
            agents: List of agent instances
            question: The question or decision to vote on
            options: Optional list of specific options to vote on
            explanation_required: Whether agents must explain their votes
            confidence_required: Whether agents must provide confidence scores
            weights: Optional dictionary mapping agent names to voting weights
            
        Returns:
            Dictionary with voting results
        """
        if not agents:
            raise ValueError("No agents provided for voting")
        
        # Prepare the voting prompt
        voting_prompt = f"QUESTION: {question}\n\n"
        
        if options:
            voting_prompt += "OPTIONS:\n"
            for i, option in enumerate(options):
                voting_prompt += f"{i+1}. {option}\n"
            
            voting_prompt += (
                "\nPlease vote for ONE of the options above by providing the option number. "
                "Your vote should be based on your specialized knowledge and expertise."
            )
        else:
            voting_prompt += (
                "Please provide your answer to this question. "
                "Your response should be based on your specialized knowledge and expertise."
            )
        
        if explanation_required:
            voting_prompt += "\nEXPLANATION: Please explain the rationale for your vote."
        
        if confidence_required:
            voting_prompt += (
                "\nCONFIDENCE: Please provide a confidence score for your vote (0.0-1.0), "
                "where 1.0 means you're absolutely certain and 0.0 means you're completely uncertain."
            )
        
        # Define schema for structured responses
        if options:
            vote_schema = {
                "type": "object",
                "properties": {
                    "vote": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": len(options),
                        "description": "Option number selected"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation for the vote"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score (0.0-1.0)"
                    }
                },
                "required": ["vote"]
            }
        else:
            vote_schema = {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your answer to the question"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation for your answer"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence score (0.0-1.0)"
                    }
                },
                "required": ["answer"]
            }
        
        if explanation_required:
            vote_schema["required"].append("explanation")
        
        if confidence_required:
            vote_schema["required"].append("confidence")
        
        # Collect votes from each agent
        votes = []
        for agent in agents:
            try:
                # Clear agent history to ensure independence
                agent.clear_history()
                
                # Add the voting prompt to the agent's history
                agent.add_to_history({"role": "user", "content": voting_prompt})
                
                # Get structured response from the agent
                vote_response = agent.api_client.generate_structured_response(
                    messages=agent.get_conversation_history(),
                    system_prompt=agent.system_prompt,
                    output_schema=vote_schema
                )
                
                # Add the agent's identity to the vote
                vote_response["agent_name"] = agent.name
                vote_response["agent_description"] = agent.description
                
                # Add the agent's weight if provided
                if weights and agent.name in weights:
                    vote_response["weight"] = weights[agent.name]
                else:
                    vote_response["weight"] = 1.0
                
                votes.append(vote_response)
                logger.info(f"Collected vote from {agent.name}")
                
            except Exception as e:
                logger.error(f"Error collecting vote from {agent.name}: {str(e)}")
                # Add a failed vote placeholder
                votes.append({
                    "agent_name": agent.name,
                    "agent_description": agent.description,
                    "error": str(e),
                    "weight": weights.get(agent.name, 1.0) if weights else 1.0
                })
        
        # Prepare the tabulation prompt for the coordinator
        tabulation_prompt = (
            f"Voting results for the question: {question}\n\n"
            f"Here are the votes from each agent:\n\n"
        )
        
        for i, vote in enumerate(votes):
            tabulation_prompt += f"AGENT {i+1}: {vote['agent_name']} ({vote['agent_description']})\n"
            
            if "error" in vote:
                tabulation_prompt += f"ERROR: {vote['error']}\n"
                continue
            
            if options:
                option_idx = vote.get("vote")
                if option_idx and 1 <= option_idx <= len(options):
                    option_text = options[option_idx - 1]
                    tabulation_prompt += f"VOTE: Option {option_idx} - {option_text}\n"
                else:
                    tabulation_prompt += f"VOTE: Invalid option {option_idx}\n"
            else:
                tabulation_prompt += f"ANSWER: {vote.get('answer', 'No answer provided')}\n"
            
            if explanation_required and "explanation" in vote:
                tabulation_prompt += f"EXPLANATION: {vote['explanation']}\n"
                
            if confidence_required and "confidence" in vote:
                tabulation_prompt += f"CONFIDENCE: {vote['confidence']}\n"
                
            if "weight" in vote and vote["weight"] != 1.0:
                tabulation_prompt += f"WEIGHT: {vote['weight']}\n"
                
            tabulation_prompt += "\n"
        
        tabulation_prompt += (
            "Please tally the votes and determine the final result. "
            "If using weighted voting, multiply each vote by the agent's weight. "
            "If there is a tie, use confidence scores to break it. "
            "Present a clear summary of the results including vote counts and the winning option/answer."
        )
        
        # Define schema for coordinator's response
        coordinator_schema = {
            "type": "object",
            "properties": {
                "tallied_votes": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "number"},
                            "weighted_count": {"type": "number"},
                            "agents": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "winning_option": {
                    "type": "string",
                    "description": "The winning option or answer"
                },
                "vote_count": {
                    "type": "number",
                    "description": "Number of votes for the winning option"
                },
                "weighted_vote_count": {
                    "type": "number",
                    "description": "Weighted vote count for the winning option"
                },
                "total_votes": {
                    "type": "number",
                    "description": "Total number of valid votes cast"
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of the voting results"
                },
                "is_tie": {
                    "type": "boolean",
                    "description": "Whether there was a tie that needed to be broken"
                },
                "tie_breaking_method": {
                    "type": "string",
                    "description": "Method used to break tie, if applicable"
                }
            },
            "required": ["winning_option", "vote_count", "total_votes", "summary"]
        }
        
        # Get the coordinator's response
        coordinator_response = self.api_client.generate_structured_response(
            messages=[{"role": "user", "content": tabulation_prompt}],
            system_prompt=self.coordinator_system_prompt,
            output_schema=coordinator_schema
        )
        
        # Combine all results
        result = {
            "question": question,
            "options": options,
            "votes": votes,
            "result": coordinator_response,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Voting completed. Winning option: {coordinator_response.get('winning_option')}")
        return result
    
    def get_consensus(self, 
                     agents: List[BaseAgent], 
                     question: str,
                     min_consensus_percentage: float = 0.6,
                     max_rounds: int = 3) -> Dict[str, Any]:
        """
        Attempt to reach consensus among agents through multiple rounds of voting.
        
        Args:
            agents: List of agent instances
            question: The question or issue to reach consensus on
            min_consensus_percentage: Minimum percentage of agents required for consensus
            max_rounds: Maximum number of voting rounds
            
        Returns:
            Dictionary with consensus results
        """
        if not agents:
            raise ValueError("No agents provided for consensus")
        
        min_required_votes = max(2, int(len(agents) * min_consensus_percentage))
        
        # Initial voting round with open-ended responses
        round_results = []
        
        initial_result = self.conduct_vote(
            agents=agents,
            question=question,
            options=None,  # Open-ended in first round
            explanation_required=True,
            confidence_required=True
        )
        
        round_results.append(initial_result)
        
        # Extract all unique answers
        unique_answers = {}
        for vote in initial_result["votes"]:
            if "answer" in vote and vote["answer"]:
                answer = vote["answer"].strip()
                if answer in unique_answers:
                    unique_answers[answer]["count"] += 1
                    unique_answers[answer]["confidence"] += vote.get("confidence", 0.5)
                    unique_answers[answer]["agents"].append(vote["agent_name"])
                else:
                    unique_answers[answer] = {
                        "count": 1,
                        "confidence": vote.get("confidence", 0.5),
                        "agents": [vote["agent_name"]]
                    }
        
        # Check if we already have consensus
        for answer, data in unique_answers.items():
            if data["count"] >= min_required_votes:
                return {
                    "question": question,
                    "consensus_reached": True,
                    "consensus_answer": answer,
                    "vote_count": data["count"],
                    "total_agents": len(agents),
                    "supporting_agents": data["agents"],
                    "rounds_required": 1,
                    "round_results": round_results,
                    "timestamp": datetime.now().isoformat()
                }
        
        # If no consensus, proceed with additional rounds
        current_round = 1
        while current_round < max_rounds:
            current_round += 1
            
            # Sort answers by vote count and confidence
            sorted_answers = sorted(
                unique_answers.items(),
                key=lambda x: (x[1]["count"], x[1]["confidence"]),
                reverse=True
            )
            
            # Take top 3 options for next round
            top_options = [answer for answer, _ in sorted_answers[:min(3, len(sorted_answers))]]
            
            # Conduct next voting round with specific options
            round_result = self.conduct_vote(
                agents=agents,
                question=f"{question}\n\nBased on previous voting, please select from these options:",
                options=top_options,
                explanation_required=True,
                confidence_required=True
            )
            
            round_results.append(round_result)
            
            # Check if consensus reached
            winning_option = round_result["result"].get("winning_option")
            vote_count = round_result["result"].get("vote_count", 0)
            
            if vote_count >= min_required_votes:
                # Get list of supporting agents
                supporting_agents = []
                for vote in round_result["votes"]:
                    if "vote" in vote and vote["vote"]:
                        option_idx = vote["vote"]
                        if 1 <= option_idx <= len(top_options) and top_options[option_idx-1] == winning_option:
                            supporting_agents.append(vote["agent_name"])
                
                return {
                    "question": question,
                    "consensus_reached": True,
                    "consensus_answer": winning_option,
                    "vote_count": vote_count,
                    "total_agents": len(agents),
                    "supporting_agents": supporting_agents,
                    "rounds_required": current_round,
                    "round_results": round_results,
                    "timestamp": datetime.now().isoformat()
                }
        
        # If we've reached max rounds without consensus
        final_result = round_results[-1]
        winning_option = final_result["result"].get("winning_option")
        vote_count = final_result["result"].get("vote_count", 0)
        
        # Get list of supporting agents for final result
        supporting_agents = []
        if "options" in final_result and final_result["options"]:
            winning_idx = final_result["options"].index(winning_option) + 1 if winning_option in final_result["options"] else None
            if winning_idx:
                for vote in final_result["votes"]:
                    if "vote" in vote and vote["vote"] == winning_idx:
                        supporting_agents.append(vote["agent_name"])
        
        return {
            "question": question,
            "consensus_reached": False,
            "best_answer": winning_option,
            "vote_count": vote_count,
            "total_agents": len(agents),
            "supporting_agents": supporting_agents,
            "rounds_required": max_rounds,
            "round_results": round_results,
            "reason": "Maximum rounds reached without sufficient consensus",
            "timestamp": datetime.now().isoformat()
        }