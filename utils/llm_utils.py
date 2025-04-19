# utils/llm_utils.py
from typing import Dict, List, Any, Optional, Union, Callable
import json
import time
import os
from anthropic import Anthropic, RateLimitError, APIStatusError, APIConnectionError

from config import DEFAULT_MODEL, DEFAULT_MAX_TOKENS

class LLMUtils:
    """
    Utility class for common LLM interaction functions.
    
    This class provides utilities for working with LLMs, including prompting,
    response parsing, error handling, and structured output generation.
    """
    
    def __init__(self, client: Optional[Anthropic] = None):
        """
        Initialize LLMUtils with an optional Anthropic client.
        
        Args:
            client: Optional Anthropic API client. If None, a new client will be created.
        """
        self.client = client or self._create_client()
    
    def _create_client(self) -> Anthropic:
        """Create an Anthropic client using API key from environment."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        return Anthropic(api_key=api_key)
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                        model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS,
                        temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt to send to the model
            system_prompt: System prompt for setting context and behavior
            model: Model to use for generation
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content
        except RateLimitError:
            # Handle rate limiting with exponential backoff
            return self._retry_with_backoff(
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
            ).content
        except (APIStatusError, APIConnectionError) as e:
            print(f"API error: {e}")
            return f"I encountered an issue connecting to the AI service. Please try again later. Error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "I encountered an unexpected error processing your request. Please try again."
    
    def _retry_with_backoff(self, operation: Callable, max_retries: int = 5, 
                          initial_backoff: float = 1.0) -> Any:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Function to retry
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            
        Returns:
            Result of the operation if successful
            
        Raises:
            Exception: If all retries fail
        """
        backoff = initial_backoff
        retries = 0
        
        while retries < max_retries:
            try:
                return operation()
            except RateLimitError:
                # Exponential backoff
                time.sleep(backoff)
                backoff *= 2
                retries += 1
                print(f"Rate limited. Retrying in {backoff} seconds (attempt {retries}/{max_retries})")
        
        # If we've exhausted retries, try one more time and let any error propagate
        return operation()
    
    def get_structured_output(self, prompt: str, output_format: Dict, 
                            system_prompt: str = "", temperature: float = 0.2) -> Dict:
        """
        Get a structured JSON output from the LLM.
        
        Args:
            prompt: User prompt to send to the model
            output_format: Dictionary describing the expected structure
            system_prompt: System prompt for setting context and behavior
            temperature: Sampling temperature (usually lower for structured output)
            
        Returns:
            Structured output as a dictionary
        """
        # Create full prompt with output format instructions
        full_prompt = f"""
        {prompt}
        
        Please format your response as a valid JSON object with the following structure:
        
        ```json
        {json.dumps(output_format, indent=2)}
        ```
        
        Return only the JSON object in your response, with no additional text.
        """
        
        # Enhance system prompt
        enhanced_system = system_prompt
        if system_prompt:
            enhanced_system += "\n\n"
        enhanced_system += "You are a helpful assistant that returns structured data in valid JSON format."
        
        try:
            response = self.generate_response(
                prompt=full_prompt,
                system_prompt=enhanced_system,
                temperature=temperature
            )
            
            # Extract JSON from response
            return self._extract_json(response)
        except Exception as e:
            print(f"Error getting structured output: {e}")
            # Return empty structure matching the output format
            return self._create_empty_structure(output_format)
    
    def _extract_json(self, text: str) -> Dict:
        """
        Extract JSON from text.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON as dictionary
        """
        # Try to find JSON in the response
        try:
            # Check if the entire text is valid JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON by looking for starting and ending braces
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                try:
                    json_str = text[start_idx:end_idx+1]
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract JSON from code blocks
            import re
            pattern = r"```(?:json)?\s*([\s\S]*?)```"
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
            
            # If all extraction attempts fail, raise error
            raise ValueError("Could not extract valid JSON from response")
    
    def _create_empty_structure(self, structure_template: Dict) -> Dict:
        """
        Create an empty structure matching the template.
        
        Args:
            structure_template: Template of the structure
            
        Returns:
            Empty structure with the same schema
        """
        result = {}
        
        for key, value in structure_template.items():
            if isinstance(value, dict):
                result[key] = self._create_empty_structure(value)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    result[key] = [self._create_empty_structure(value[0])]
                else:
                    result[key] = []
            elif isinstance(value, str):
                result[key] = ""
            elif isinstance(value, (int, float)):
                result[key] = 0
            elif isinstance(value, bool):
                result[key] = False
            else:
                result[key] = None
        
        return result
    
    def classify_user_intent(self, user_query: str) -> Dict[str, float]:
        """
        Classify user intent from a query.
        
        Args:
            user_query: User's query text
            
        Returns:
            Dictionary with intent categories and confidence scores
        """
        prompt = f"""
        Classify the financial intent of the following user query into one or more of these categories:
        - budget_planning: Questions about creating or managing a budget
        - debt_management: Questions about managing or paying off debt
        - investment_advice: Questions about investments or portfolio management
        - savings_goals: Questions about saving for specific goals
        - tax_planning: Questions about tax optimization or planning
        - general_question: General financial questions that don't fit other categories
        
        Return a JSON object with each category and a confidence score from 0.0 to 1.0.
        
        USER QUERY: {user_query}
        """
        
        output_format = {
            "budget_planning": 0.0,
            "debt_management": 0.0,
            "investment_advice": 0.0,
            "savings_goals": 0.0,
            "tax_planning": 0.0,
            "general_question": 0.0
        }
        
        result = self.get_structured_output(
            prompt=prompt,
            output_format=output_format,
            temperature=0.1
        )
        
        return result
    
    def extract_financial_data(self, user_text: str) -> Dict:
        """
        Extract structured financial data from user text.
        
        Args:
            user_text: Text containing financial information
            
        Returns:
            Dictionary with extracted financial data
        """
        prompt = f"""
        Extract structured financial data from the following text. If a piece of information 
        is not mentioned, leave the corresponding field empty.
        
        TEXT: {user_text}
        """
        
        output_format = {
            "income": {
                "salary": None,
                "self_employment": None,
                "investments": None,
                "other": None
            },
            "expenses": {
                "housing": None,
                "transportation": None,
                "food": None,
                "utilities": None,
                "insurance": None,
                "healthcare": None,
                "personal": None,
                "entertainment": None,
                "other": None
            },
            "debts": [
                {
                    "type": None,
                    "name": None,
                    "balance": None,
                    "interest_rate": None,
                    "minimum_payment": None
                }
            ],
            "savings": {
                "emergency_fund": None,
                "retirement_accounts": None,
                "other_savings": None
            },
            "investments": {
                "stocks": None,
                "bonds": None,
                "real_estate": None,
                "other": None
            },
            "financial_goals": [
                {
                    "goal": None,
                    "target_amount": None,
                    "timeline": None,
                    "priority": None
                }
            ]
        }
        
        result = self.get_structured_output(
            prompt=prompt,
            output_format=output_format,
            temperature=0.2
        )
        
        # Clean up the result by removing None values
        return self._clean_financial_data(result)
    
    def _clean_financial_data(self, data: Dict) -> Dict:
        """
        Clean financial data by removing None values and empty containers.
        
        Args:
            data: Raw extracted financial data
            
        Returns:
            Cleaned financial data
        """
        if not isinstance(data, dict):
            return data
        
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively clean dictionaries
                cleaned_dict = self._clean_financial_data(value)
                if cleaned_dict:  # Only include non-empty dictionaries
                    result[key] = cleaned_dict
            elif isinstance(value, list):
                # Clean each item in the list
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = self._clean_financial_data(item)
                        if cleaned_item:  # Only include non-empty items
                            cleaned_list.append(cleaned_item)
                    elif item is not None:
                        cleaned_list.append(item)
                
                if cleaned_list:  # Only include non-empty lists
                    result[key] = cleaned_list
            elif value is not None:
                result[key] = value
        
        return result
    
    def generate_financial_explanation(self, concept: str, complexity_level: str = "intermediate") -> str:
        """
        Generate an educational explanation of a financial concept.
        
        Args:
            concept: Financial concept to explain
            complexity_level: Desired complexity level (basic, intermediate, advanced)
            
        Returns:
            Educational explanation of the concept
        """
        prompt = f"""
        Please explain this financial concept: {concept}
        
        Target the explanation at a {complexity_level} level of understanding.
        Include:
        1. A clear definition
        2. Why it matters
        3. How it applies to personal finance
        4. An example or illustration
        
        Make the explanation educational and helpful for someone learning about personal finance.
        """
        
        system_prompt = """
        You are a financial educator who specializes in making complex financial concepts 
        accessible and understandable. Provide clear, accurate explanations with concrete examples.
        """
        
        return self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
    
    def summarize_financial_advice(self, detailed_advice: str, max_length: int = 200) -> str:
        """
        Summarize complex financial advice into a concise format.
        
        Args:
            detailed_advice: Detailed financial advice to summarize
            max_length: Maximum length of the summary in characters
            
        Returns:
            Concise summary of the advice
        """
        prompt = f"""
        Summarize the following financial advice in a concise, actionable format:
        
        {detailed_advice}
        
        The summary should be no more than {max_length} characters and focus on the key 
        actionable points while preserving the most important information.
        """
        
        system_prompt = """
        You are a financial communication specialist who excels at distilling complex 
        financial information into concise, actionable summaries without losing the 
        core message or key details.
        """
        
        summary = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_length // 2  # Approximate token count
        )
        
        return summary
    
    def generate_comparison_table(self, options: List[Dict], criteria: List[str]) -> str:
        """
        Generate a markdown comparison table for financial options.
        
        Args:
            options: List of option dictionaries with attributes
            criteria: List of criteria to compare
            
        Returns:
            Markdown table comparing the options
        """
        # Validate inputs
        if not options or not criteria:
            return "Insufficient data for comparison."
        
        # Create a description of the options
        options_desc = []
        for i, option in enumerate(options):
            option_str = f"Option {i+1}: {option.get('name', f'Option {i+1}')}\n"
            for key, value in option.items():
                if key != 'name':
                    option_str += f"- {key}: {value}\n"
            options_desc.append(option_str)
        
        # Create prompt
        prompt = f"""
        Create a markdown comparison table for the following financial options:
        
        {"".join(options_desc)}
        
        Compare these options based on the following criteria:
        {", ".join(criteria)}
        
        Format the table with:
        - A header row with "Option" and each criterion
        - A row for each option with its values for each criterion
        - Proper markdown table syntax
        
        Also provide a brief paragraph highlighting the key differences between the options.
        """
        
        system_prompt = """
        You are a financial analyst who specializes in creating clear, informative comparisons 
        of financial options. Present information in well-structured tables with accurate, 
        balanced assessments.
        """
        
        return self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
    
    def parse_financial_query(self, user_query: str) -> Dict:
        """
        Parse a financial query to identify key components and parameters.
        
        Args:
            user_query: User's financial query
            
        Returns:
            Dictionary with parsed query components
        """
        prompt = f"""
        Parse the following financial query to identify its key components and parameters:
        
        USER QUERY: {user_query}
        
        Extract the following information (if present):
        - Main topic (e.g., budgeting, investing, debt)
        - Specific action or request
        - Time frame mentioned
        - Amount(s) mentioned
        - Rate(s) mentioned
        - Goal(s) mentioned
        - Constraints or limitations
        - User's current financial situation details
        
        Return the results in a JSON structure.
        """
        
        output_format = {
            "main_topic": "",
            "action": "",
            "time_frame": "",
            "amounts": [],
            "rates": [],
            "goals": [],
            "constraints": [],
            "current_situation": ""
        }
        
        result = self.get_structured_output(
            prompt=prompt,
            output_format=output_format,
            temperature=0.1
        )
        
        return result