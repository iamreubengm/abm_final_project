from typing import Dict, Any, List, Optional, Callable
import json
from datetime import datetime

from personal_finance_portal.config import config
from sqlalchemy.orm import Session
from personal_finance_portal.data.models import AgentInteraction

class HumanReflectionPattern:
    """
    Implementation of the Human Reflection pattern.
    Collects and incorporates user feedback on financial advice.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the human reflection pattern with configuration"""
        self.feedback_prompt = config.patterns["human_reflection"].parameters.get(
            "feedback_prompt", "How would you rate this advice from 1-5?"
        )
        self.db_session = db_session
        
    def get_feedback(
        self, 
        advice: str,
        context: Dict[str, Any] = None,
        custom_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate a feedback request for the user
        
        Args:
            advice: The financial advice to get feedback on
            context: Additional context (optional)
            custom_prompt: Custom feedback prompt (optional)
            
        Returns:
            Dict with the formatted feedback request
        """
        prompt = custom_prompt or self.feedback_prompt
        
        feedback_request = {
            "advice": advice,
            "prompt": prompt,
            "feedback_options": [
                {"rating": 1, "label": "Not helpful at all"},
                {"rating": 2, "label": "Slightly helpful"},
                {"rating": 3, "label": "Moderately helpful"},
                {"rating": 4, "label": "Very helpful"},
                {"rating": 5, "label": "Extremely helpful"}
            ],
            "comment_prompt": "Would you like to add any comments on how this advice could be improved?",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if context:
            feedback_request["context"] = context
            
        return feedback_request
    
    def record_feedback(
        self,
        user_id: int,
        agent_type: str,
        user_query: str,
        agent_response: str,
        feedback_rating: int,
        feedback_comment: Optional[str] = None,
        patterns_used: Optional[List[str]] = None
    ) -> bool:
        """
        Record user feedback in the database
        
        Args:
            user_id: The ID of the user providing feedback
            agent_type: The type of agent (budget, investment, etc.)
            user_query: The original user query
            agent_response: The agent's response
            feedback_rating: Numerical rating (typically 1-5)
            feedback_comment: Optional text comment
            patterns_used: List of pattern names used in the response
            
        Returns:
            Boolean indicating success
        """
        if not self.db_session:
            print("Warning: No database session provided. Feedback not recorded.")
            return False
            
        try:
            # Create a JSON representation of patterns used
            patterns_json = json.dumps(patterns_used) if patterns_used else None
            
            # Create the interaction record
            interaction = AgentInteraction(
                user_id=user_id,
                agent_type=agent_type,
                user_query=user_query,
                agent_response=agent_response,
                patterns_used=patterns_json,
                feedback_rating=feedback_rating,
                feedback_comment=feedback_comment
            )
            
            # Add to database
            self.db_session.add(interaction)
            self.db_session.commit()
            
            return True
        except Exception as e:
            print(f"Error recording feedback: {str(e)}")
            return False
    
    def analyze_feedback_history(
        self,
        user_id: Optional[int] = None,
        agent_type: Optional[str] = None,
        min_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze feedback history to identify patterns
        
        Args:
            user_id: Filter by specific user (optional)
            agent_type: Filter by specific agent type (optional)
            min_date: Minimum date for feedback (optional)
            
        Returns:
            Dict with feedback analysis
        """
        if not self.db_session:
            return {"error": "No database session provided"}
            
        try:
            # Build query based on filters
            query = self.db_session.query(AgentInteraction)
            
            if user_id is not None:
                query = query.filter(AgentInteraction.user_id == user_id)
                
            if agent_type:
                query = query.filter(AgentInteraction.agent_type == agent_type)
                
            if min_date:
                query = query.filter(AgentInteraction.created_at >= min_date)
                
            # Execute query
            interactions = query.all()
            
            if not interactions:
                return {"message": "No feedback data found matching the criteria"}
                
            # Analyze feedback
            total_interactions = len(interactions)
            rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            pattern_ratings = {}
            agent_ratings = {}
            
            for interaction in interactions:
                # Count ratings
                rating = interaction.feedback_rating
                if rating is not None:
                    if rating in rating_counts:
                        rating_counts[rating] += 1
                        
                    # Track agent type ratings
                    agent = interaction.agent_type
                    if agent not in agent_ratings:
                        agent_ratings[agent] = {"count": 0, "sum": 0, "avg": 0}
                    agent_ratings[agent]["count"] += 1
                    agent_ratings[agent]["sum"] += rating
                    
                    # Track pattern ratings
                    if interaction.patterns_used:
                        patterns = json.loads(interaction.patterns_used)
                        for pattern in patterns:
                            if pattern not in pattern_ratings:
                                pattern_ratings[pattern] = {"count": 0, "sum": 0, "avg": 0}
                            pattern_ratings[pattern]["count"] += 1
                            pattern_ratings[pattern]["sum"] += rating
            
            # Calculate averages
            average_rating = 0
            total_rated = sum(rating_counts.values())
            if total_rated > 0:
                weighted_sum = sum(rating * count for rating, count in rating_counts.items())
                average_rating = weighted_sum / total_rated
                
            # Calculate pattern averages
            for pattern, data in pattern_ratings.items():
                if data["count"] > 0:
                    data["avg"] = data["sum"] / data["count"]
                    
            # Calculate agent averages
            for agent, data in agent_ratings.items():
                if data["count"] > 0:
                    data["avg"] = data["sum"] / data["count"]
            
            # Prepare analysis
            analysis = {
                "total_interactions": total_interactions,
                "total_with_ratings": total_rated,
                "average_rating": average_rating,
                "rating_distribution": rating_counts,
                "agent_ratings": agent_ratings,
                "pattern_ratings": pattern_ratings
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing feedback: {str(e)}")
            return {"error": f"Error analyzing feedback: {str(e)}"}
            
    def incorporate_feedback(
        self,
        agent_function: Callable,
        original_query: str,
        original_response: str,
        feedback: Dict[str, Any]
    ) -> str:
        """
        Incorporate user feedback to improve a response
        
        Args:
            agent_function: The agent function to generate improved response
            original_query: The original user query
            original_response: The original agent response
            feedback: User feedback dictionary
            
        Returns:
            Improved response incorporating feedback
        """
        # Format the improvement prompt
        prompt = self._create_improvement_prompt(
            original_query, original_response, feedback
        )
        
        # Generate improved response
        improved_response = agent_function(prompt)
        
        return improved_response
        
    def _create_improvement_prompt(
        self,
        original_query: str,
        original_response: str,
        feedback: Dict[str, Any]
    ) -> str:
        """Create a prompt for improving a response based on feedback"""
        prompt_parts = [
            "Please improve this financial advice based on user feedback:",
            f"\nORIGINAL QUESTION: {original_query}",
            f"\nORIGINAL RESPONSE: {original_response}",
            "\nUSER FEEDBACK:"
        ]
        
        # Add numerical rating
        if "rating" in feedback:
            prompt_parts.append(f"Rating: {feedback['rating']}/5")
            
        # Add text comment if available
        if "comment" in feedback and feedback["comment"]:
            prompt_parts.append(f"Comment: {feedback['comment']}")
            
        prompt_parts.append("\nPlease provide an improved response that addresses the user feedback.")
        prompt_parts.append("Be specific about what has been improved based on the feedback.")
        
        return "\n".join(prompt_parts)