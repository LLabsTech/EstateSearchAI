from typing import List
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from models.property import PropertyMatch
from .base import LLMHandler

class ClaudeHandler(LLMHandler):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model,
            temperature=0.7
        )

    async def generate_response(self, query: str, matches: List[PropertyMatch]) -> str:
        system_prompt = self._create_system_prompt()
        property_context = self._create_property_context(matches)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
User Query: {query}

Available properties:
{property_context}

Please analyze these properties and suggest the best matches for the user's requirements.
            """.strip())
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
