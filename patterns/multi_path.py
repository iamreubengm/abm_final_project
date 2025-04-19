from typing import List, Dict, Any, Optional, Callable
import json
from dataclasses import dataclass, asdict

from personal_finance_portal.config import config

@dataclass
class FinancialPlan:
    """Represents a financial plan path"""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    timeline: str
    pros: List[str]
    cons: List[str]
    risk_level: str  # low, medium, high
    expected_outcome: str
    suitable_for: List[str]

class MultiPathPlanGenerator:
    """
    Implementation of the Multi-Path Plan Generator pattern.
    Generates multiple approaches to achieve a financial goal.
    """
    
    def __init__(self):
        """Initialize the multi-path generator with configuration"""
        self.num_paths = config.patterns["multi_path"].parameters.get("num_paths", 3)
        
    def generate_plans(
        self,
        goal: str,
        context: Dict[str, Any],
        agent_function: Callable,
        num_paths: Optional[int] = None
    ) -> List[FinancialPlan]:
        """
        Generate multiple financial plans to achieve a goal
        
        Args:
            goal: The financial goal to plan for
            context: User financial context
            agent_function: The agent function that generates plans
            num_paths: Number of paths to generate (overrides default)
            
        Returns:
            List of financial plans as FinancialPlan objects
        """
        if num_paths is None:
            num_paths = self.num_paths
            
        # Create the plan generation prompt
        prompt = self._create_plan_prompt(goal, context, num_paths)
        
        # Ask the agent to generate plans
        response = agent_function(prompt)
        
        # Parse the plans from the response
        plans = self._parse_plans(response)
        
        # Convert to FinancialPlan objects
        plan_objects = []
        for plan in plans:
            try:
                plan_obj = FinancialPlan(
                    name=plan.get("name", "Unnamed Plan"),
                    description=plan.get("description", ""),
                    steps=plan.get("steps", []),
                    timeline=plan.get("timeline", ""),
                    pros=plan.get("pros", []),
                    cons=plan.get("cons", []),
                    risk_level=plan.get("risk_level", "medium"),
                    expected_outcome=plan.get("expected_outcome", ""),
                    suitable_for=plan.get("suitable_for", [])
                )
                plan_objects.append(plan_obj)
            except Exception as e:
                print(f"Error creating plan object: {str(e)}")
        
        return plan_objects
    
    def _create_plan_prompt(
        self,
        goal: str,
        context: Dict[str, Any],
        num_paths: int
    ) -> str:
        """Create the prompt for plan generation"""
        prompt_parts = [
            f"Generate {num_paths} different financial plans to achieve the following goal:",
            f"GOAL: {goal}\n",
            "USER CONTEXT:"
        ]
        
        # Add context information
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
        
        prompt_parts.append("\nFor each plan, provide the following information:")
        prompt_parts.append("1. A unique name for the plan")
        prompt_parts.append("2. A brief description of the approach")
        prompt_parts.append("3. Specific steps to implement the plan")
        prompt_parts.append("4. Expected timeline")
        prompt_parts.append("5. Pros and cons of this approach")
        prompt_parts.append("6. Risk level (low, medium, high)")
        prompt_parts.append("7. Expected outcome if the plan is followed")
        prompt_parts.append("8. Who this plan is most suitable for")
        
        prompt_parts.append("\nMake each plan distinctly different in approach.")
        prompt_parts.append("Format your response as a JSON array with each plan as an object.")
        
        prompt_example = """{
  "plans": [
    {
      "name": "Plan Name",
      "description": "Brief description of the plan",
      "steps": [
        {"step": 1, "action": "First action", "timeline": "Month 1"},
        {"step": 2, "action": "Second action", "timeline": "Month 2-3"}
      ],
      "timeline": "6 months",
      "pros": ["Pro 1", "Pro 2"],
      "cons": ["Con 1", "Con 2"],
      "risk_level": "medium",
      "expected_outcome": "Expected result of following this plan",
      "suitable_for": ["Young professionals", "Risk-tolerant investors"]
    }
  ]
}"""
        
        prompt_parts.append(f"\nExample format:\n{prompt_example}")
        
        return "\n".join(prompt_parts)
    
    def _parse_plans(self, response: str) -> List[Dict[str, Any]]:
        """Parse the plans from the agent's response"""
        # Try to find JSON in the response
        try:
            # Look for JSON blocks in markdown format
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            else:
                # Assume the entire response might be JSON
                json_text = response
                
            data = json.loads(json_text)
            
            # Check if the plans are in a "plans" key or directly in a list
            if isinstance(data, dict) and "plans" in data:
                return data["plans"]
            elif isinstance(data, list):
                return data
            else:
                # Try to extract any list from the data
                for key, value in data.items():
                    if isinstance(value, list):
                        return value
                        
                raise ValueError("Could not find plans list in response")
                
        except Exception as e:
            print(f"Error parsing plans: {str(e)}")
            # Fallback: Try to create a basic plan structure from the text
            return [{"name": "Fallback Plan", 
                   "description": "A plan extracted from unstructured text.",
                   "steps": [{"step": 1, "action": "Review the full response for details"}],
                   "timeline": "Varies",
                   "pros": ["Based on agent's analysis"],
                   "cons": ["Unstructured format"],
                   "risk_level": "medium",
                   "expected_outcome": "Achieving financial goal with manual planning",
                   "suitable_for": ["All users"]}]
            
    def compare_plans(self, plans: List[FinancialPlan]) -> Dict[str, Any]:
        """
        Compare multiple financial plans and summarize their differences
        
        Args:
            plans: List of financial plans to compare
            
        Returns:
            Dictionary with comparison data
        """
        if not plans:
            return {"error": "No plans to compare"}
            
        comparison = {
            "total_plans": len(plans),
            "plan_names": [plan.name for plan in plans],
            "risk_distribution": {},
            "timeline_summary": {},
            "common_steps": [],
            "unique_approaches": []
        }
        
        # Analyze risk levels
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        for plan in plans:
            risk = plan.risk_level.lower()
            if risk in risk_counts:
                risk_counts[risk] += 1
        comparison["risk_distribution"] = risk_counts
        
        # Find common steps across plans
        all_steps = []
        for plan in plans:
            plan_steps = [step.get("action", "") for step in plan.steps]
            all_steps.append(set(plan_steps))
            
        if all_steps:
            common = all_steps[0]
            for steps in all_steps[1:]:
                common = common.intersection(steps)
            comparison["common_steps"] = list(common)
        
        # Identify unique approaches
        for plan in plans:
            unique_aspects = {
                "plan_name": plan.name,
                "distinguishing_feature": "",
                "unique_steps": []
            }
            
            # Find steps unique to this plan
            plan_steps = set([step.get("action", "") for step in plan.steps])
            for other_plan in plans:
                if other_plan.name == plan.name:
                    continue
                other_steps = set([step.get("action", "") for step in other_plan.steps])
                unique_steps = plan_steps - other_steps
                if unique_steps:
                    unique_aspects["unique_steps"].extend(list(unique_steps))
            
            # Try to identify a distinguishing feature
            if plan.description:
                unique_aspects["distinguishing_feature"] = plan.description.split(".")[0]
            
            comparison["unique_approaches"].append(unique_aspects)
        
        return comparison