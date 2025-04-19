import os
import json
from anthropic import Anthropic
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class ClaudeAPIClient:
    """
    Client for interacting with Anthropic's Claude API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Claude API client.
        
        Args:
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
            model: Claude model to use. If not provided, will use DEFAULT_MODEL env var or fallback to default.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.model = model or os.environ.get("DEFAULT_MODEL", "claude-3-opus-20240229")
        self.client = Anthropic(api_key=self.api_key)
        
        logger.info(f"Initialized Claude API client with model: {self.model}")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         system_prompt: Optional[str] = None,
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> str:
        """
        Generate a response from Claude.
        
        Args:
            messages: List of message dictionaries with role and content keys
            system_prompt: Optional system prompt to guide Claude's behavior
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature for response generation (0.0-1.0)
            
        Returns:
            The generated response text
        """
        try:
            # Prepare default system prompt if none provided
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful, honest, and accurate financial advisor AI assistant. "
                    "Provide clear advice based on financial best practices. "
                    "When you don't know something, admit it rather than making up information. "
                    "Always consider the user's financial goals and risk tolerance in your advice."
                )
            
            # Create the API request
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error generating response from Claude API: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def generate_structured_response(self, 
                                    messages: List[Dict[str, str]], 
                                    system_prompt: Optional[str] = None,
                                    output_schema: Dict[str, Any] = None,
                                    max_tokens: int = 1000,
                                    temperature: float = 0.3) -> Dict[str, Any]:
        """
        Generate a structured response from Claude using JSON mode.
        
        Args:
            messages: List of message dictionaries with role and content keys
            system_prompt: Optional system prompt to guide Claude's behavior
            output_schema: Schema defining the expected output structure
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature for response generation (0.0-1.0)
            
        Returns:
            Dict containing the structured response
        """
        try:
            # Create a system prompt that requests JSON output
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful, honest, and accurate financial advisor AI assistant. "
                    "Provide clear advice based on financial best practices. "
                    "You will respond with a JSON object that strictly follows the specified schema."
                )
            else:
                system_prompt += " Respond with a JSON object that strictly follows the specified schema."
            
            # Add the schema to the last user message
            schema_message = {
                "role": "user", 
                "content": f"Please provide a response following this JSON schema: {json.dumps(output_schema)}"
            }
            
            # Add a description of the schema to the last message if not already present
            last_message_content = messages[-1]["content"]
            if "JSON schema" not in last_message_content:
                schema_desc = f"\n\nPlease format your response as JSON following this schema: {json.dumps(output_schema)}"
                messages[-1]["content"] = last_message_content + schema_desc
            
            # Make the API request
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract and parse JSON from the response
            response_text = response.content[0].text
            
            # Try to find JSON in the response
            try:
                # First attempt: try to parse the entire response as JSON
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Second attempt: try to extract JSON blocks from markdown or text
                if "```json" in response_text:
                    json_block = response_text.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_block)
                elif "```" in response_text:
                    json_block = response_text.split("```")[1].split("```")[0].strip()
                    return json.loads(json_block)
                else:
                    # If we can't find JSON, raise an error
                    raise ValueError("Could not extract JSON from Claude's response")
                
        except Exception as e:
            logger.error(f"Error generating structured response from Claude API: {str(e)}")
            return {"error": str(e), "message": "Failed to generate structured response"}