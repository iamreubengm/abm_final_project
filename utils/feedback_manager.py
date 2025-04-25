from typing import Dict, Any, Optional
import streamlit as st
from datetime import datetime

class FeedbackManager:
    """
    Manages feedback collection and storage in session state.
    """
    
    def __init__(self):
        """Initialize the feedback manager."""
        pass
        
    def record_insight_feedback(
        self,
        user_id: int,
        insight: Dict[str, Any],
        feedback: Dict[str, Any]
    ) -> bool:
        """
        Record feedback for an AI insight in session state.
        
        Args:
            user_id: The ID of the user providing feedback
            insight: The insight being rated
            feedback: The feedback data (rating and comment)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Store feedback in session state
            if "insight_feedback" not in st.session_state:
                st.session_state.insight_feedback = {}
            
            feedback_key = f"{insight['title']}_{datetime.utcnow().isoformat()}"
            st.session_state.insight_feedback[feedback_key] = {
                "user_id": user_id,
                "insight": insight,
                "feedback": feedback
            }
            return True
        except Exception as e:
            print(f"Error recording feedback: {str(e)}")
            return False
        
    def get_feedback_analysis(
        self,
        user_id: Optional[int] = None,
        min_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get analysis of feedback history from session state.
        
        Args:
            user_id: Filter by specific user (optional)
            min_date: Minimum date for feedback (optional)
            
        Returns:
            Dict with feedback analysis
        """
        if "insight_feedback" not in st.session_state:
            return {"message": "No feedback data found"}
            
        feedback_data = st.session_state.insight_feedback
        
        # Filter by user_id if provided
        if user_id:
            feedback_data = {
                k: v for k, v in feedback_data.items() 
                if v["user_id"] == user_id
            }
            
        # Filter by date if provided
        if min_date:
            feedback_data = {
                k: v for k, v in feedback_data.items() 
                if v["feedback"]["timestamp"] >= min_date
            }
            
        if not feedback_data:
            return {"message": "No feedback data found matching the criteria"}
            
        # Analyze feedback
        total_feedback = len(feedback_data)
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        pattern_ratings = {}
        
        for feedback in feedback_data.values():
            rating = feedback["feedback"]["rating"]
            if rating in rating_counts:
                rating_counts[rating] += 1
                
            # Track pattern ratings
            pattern = feedback["insight"]["pattern"]
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
                
        return {
            "total_feedback": total_feedback,
            "average_rating": average_rating,
            "rating_distribution": rating_counts,
            "pattern_ratings": pattern_ratings
        } 