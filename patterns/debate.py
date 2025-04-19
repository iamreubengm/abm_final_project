from typing import List, Dict, Any, Callable, Optional
import json
from dataclasses import dataclass

from personal_finance_portal.config import config

@dataclass
class DebateMessage:
    """Represents a message in the debate"""
    agent_id: str
    content: str
    role: str  # proposal, critique, defense, revision, conclusion


class DebateBasedCooperation:
    """
    Implementation of the Debate-Based Cooperation pattern.
    Facilitates debate between agents to refine financial advice.
    """
    
    def __init__(self):
        """Initialize the debate pattern with configuration"""
        self.max_rounds = config.patterns["debate"].parameters.get("max_rounds", 3)
        
    def run_debate(
        self,
        topic: str,
        context: Dict[str, Any],
        agents: Dict[str, Callable],
        max_rounds: Optional[int] = None,
        initial_proposal: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a debate between agents to reach a refined conclusion
        
        Args:
            topic: The financial topic to debate
            context: User financial context
            agents: Dictionary of agent functions with role names as keys
            max_rounds: Maximum debate rounds (overrides default)
            initial_proposal: Optional starting proposal
            
        Returns:
            Dict with debate results and transcript
        """
        if max_rounds is None:
            max_rounds = self.max_rounds
            
        transcript: List[DebateMessage] = []
        
        # Create initial proposal if not provided
        if initial_proposal is None:
            # Use the proposer agent to create an initial proposal
            if "proposer" in agents:
                proposal_prompt = self._create_proposal_prompt(topic, context)
                initial_proposal = agents["proposer"](proposal_prompt)
            else:
                # Use the first agent as a fallback
                first_agent_id = list(agents.keys())[0]
                proposal_prompt = self._create_proposal_prompt(topic, context)
                initial_proposal = agents[first_agent_id](proposal_prompt)
        
        # Add initial proposal to transcript
        transcript.append(DebateMessage(
            agent_id="proposer" if "proposer" in agents else list(agents.keys())[0],
            content=initial_proposal,
            role="proposal"
        ))
        
        current_proposal = initial_proposal
        
        # Run debate rounds
        for round_num in range(1, max_rounds + 1):
            # Critiques
            critiques = {}
            for agent_id, agent_func in agents.items():
                if agent_id == transcript[-1].agent_id:
                    continue  # Skip the agent that made the proposal
                
                critique_prompt = self._create_critique_prompt(
                    topic, context, current_proposal, round_num
                )
                critique = agent_func(critique_prompt)
                
                critiques[agent_id] = critique
                transcript.append(DebateMessage(
                    agent_id=agent_id,
                    content=critique,
                    role="critique"
                ))
            
            # Defense from the proposer
            proposer_id = transcript[0].agent_id
            defense_prompt = self._create_defense_prompt(
                topic, context, current_proposal, critiques, round_num
            )
            defense = agents[proposer_id](defense_prompt)
            
            transcript.append(DebateMessage(
                agent_id=proposer_id,
                content=defense,
                role="defense"
            ))
            
            # Revision based on critiques and defense
            revision_prompt = self._create_revision_prompt(
                topic, context, current_proposal, critiques, defense, round_num
            )
            
            # Choose a different agent for the revision
            reviser_id = None
            for agent_id in agents.keys():
                if agent_id != proposer_id:
                    reviser_id = agent_id
                    break
                    
            if reviser_id is None:
                reviser_id = proposer_id  # Fallback to the proposer
                
            revision = agents[reviser_id](revision_prompt)
            
            transcript.append(DebateMessage(
                agent_id=reviser_id,
                content=revision,
                role="revision"
            ))
            
            # Update the current proposal
            current_proposal = revision
            
            # Check if we've reached consensus (could implement a more 
            # sophisticated check here)
            if self._check_consensus(critiques):
                break
        
        # Final conclusion
        conclusion_prompt = self._create_conclusion_prompt(
            topic, context, transcript, current_proposal
        )
        
        # Choose an agent for the conclusion
        conclusion_agent_id = list(agents.keys())[0]  # Default to first agent
        
        conclusion = agents[conclusion_agent_id](conclusion_prompt)
        
        transcript.append(DebateMessage(
            agent_id=conclusion_agent_id,
            content=conclusion,
            role="conclusion"
        ))
        
        # Format the debate results
        result = {
            "topic": topic,
            "rounds": max_rounds,
            "initial_proposal": initial_proposal,
            "final_proposal": current_proposal,
            "conclusion": conclusion,
            "transcript": [
                {
                    "agent_id": msg.agent_id,
                    "content": msg.content,
                    "role": msg.role
                }
                for msg in transcript
            ],
            "agents_involved": list(agents.keys())
        }
        
        return result
    
    def _create_proposal_prompt(self, topic: str, context: Dict[str, Any]) -> str:
        """Create a prompt for the initial proposal"""
        prompt_parts = [
            "You are a financial advisor tasked with creating an initial proposal on the following topic:",
            f"TOPIC: {topic}\n",
            "USER CONTEXT:"
        ]
        
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
            
        prompt_parts.append("\nProvide a thoughtful, detailed proposal that addresses this financial topic.")
        prompt_parts.append("Consider different perspectives and approaches.")
        prompt_parts.append("Format your proposal clearly with appropriate headings and structure.")
        
        return "\n".join(prompt_parts)
    
    def _create_critique_prompt(
        self, 
        topic: str, 
        context: Dict[str, Any],
        proposal: str,
        round_num: int
    ) -> str:
        """Create a prompt for critiquing a proposal"""
        prompt_parts = [
            "You are a critical financial analyst reviewing a proposal on the following topic:",
            f"TOPIC: {topic}\n",
            "USER CONTEXT:"
        ]
        
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
            
        prompt_parts.append(f"\nPROPOSAL (Round {round_num}):")
        prompt_parts.append(proposal)
        
        prompt_parts.append("\nYour task is to critically evaluate this proposal.")
        prompt_parts.append("Identify potential weaknesses, risks, or improvements.")
        prompt_parts.append("Be specific and constructive in your criticism.")
        prompt_parts.append("Format your critique in clear paragraphs addressing different aspects.")
        
        return "\n".join(prompt_parts)
    
    def _create_defense_prompt(
        self,
        topic: str,
        context: Dict[str, Any],
        proposal: str,
        critiques: Dict[str, str],
        round_num: int
    ) -> str:
        """Create a prompt for defending against critiques"""
        prompt_parts = [
            "You are a financial advisor defending your proposal on the following topic:",
            f"TOPIC: {topic}\n",
            "USER CONTEXT:"
        ]
        
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
            
        prompt_parts.append(f"\nYOUR PROPOSAL (Round {round_num}):")
        prompt_parts.append(proposal)
        
        prompt_parts.append("\nCRITIQUES:")
        for agent_id, critique in critiques.items():
            prompt_parts.append(f"Critique from {agent_id}:")
            prompt_parts.append(critique)
            prompt_parts.append("")
            
        prompt_parts.append("\nYour task is to defend your proposal against these critiques.")
        prompt_parts.append("Address the main points raised by each critique.")
        prompt_parts.append("Acknowledge valid criticisms and explain your reasoning.")
        prompt_parts.append("Format your defense clearly, addressing each major criticism.")
        
        return "\n".join(prompt_parts)
    
    def _create_revision_prompt(
        self,
        topic: str,
        context: Dict[str, Any],
        proposal: str,
        critiques: Dict[str, str],
        defense: str,
        round_num: int
    ) -> str:
        """Create a prompt for revising a proposal"""
        prompt_parts = [
            "You are a financial advisor revising a proposal based on debate:",
            f"TOPIC: {topic}\n",
            "USER CONTEXT:"
        ]
        
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
            
        prompt_parts.append(f"\nORIGINAL PROPOSAL (Round {round_num}):")
        prompt_parts.append(proposal)
        
        prompt_parts.append("\nCRITIQUES:")
        for agent_id, critique in critiques.items():
            prompt_parts.append(f"Critique from {agent_id}:")
            prompt_parts.append(critique)
            prompt_parts.append("")
            
        prompt_parts.append("\nDEFENSE:")
        prompt_parts.append(defense)
        
        prompt_parts.append("\nYour task is to create an improved, revised proposal.")
        prompt_parts.append("Incorporate valid criticisms and maintain strengths from the original.")
        prompt_parts.append("Be specific about what you've changed and why.")
        prompt_parts.append("Format your revision clearly with appropriate structure.")
        
        return "\n".join(prompt_parts)
    
    def _create_conclusion_prompt(
        self,
        topic: str,
        context: Dict[str, Any],
        transcript: List[DebateMessage],
        final_proposal: str
    ) -> str:
        """Create a prompt for generating a conclusion"""
        prompt_parts = [
            "You are a financial advisor summarizing a debate on the following topic:",
            f"TOPIC: {topic}\n",
            "USER CONTEXT:"
        ]
        
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
        
        prompt_parts.append("\nFINAL PROPOSAL:")
        prompt_parts.append(final_proposal)
        
        prompt_parts.append("\nYour task is to provide a conclusion that:")
        prompt_parts.append("1. Summarizes the key points from the debate")
        prompt_parts.append("2. Highlights how the proposal evolved")
        prompt_parts.append("3. Emphasizes the strengths of the final recommendation")
        prompt_parts.append("4. Acknowledges any remaining limitations")
        prompt_parts.append("Format your conclusion clearly and concisely.")
        
        return "\n".join(prompt_parts)
    
    def _check_consensus(self, critiques: Dict[str, str]) -> bool:
        """
        Check if there's a consensus among critiques
        A very basic implementation that could be enhanced
        
        Returns:
            Boolean indicating whether consensus was reached
        """
        # Simple heuristic: check if critiques contain positive phrases
        consensus_phrases = [
            "agree", "concur", "support", "endorse", "acceptable", 
            "reasonable", "sound", "valid", "good point", "makes sense"
        ]
        
        agreement_count = 0
        for critique in critiques.values():
            critique_lower = critique.lower()
            if any(phrase in critique_lower for phrase in consensus_phrases):
                agreement_count += 1
                
        # Consider it consensus if more than half of critics agree
        return agreement_count > len(critiques) / 2