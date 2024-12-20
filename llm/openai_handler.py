from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from models.property import PropertyMatch
from .base import LLMHandler

class OpenAIHandler(LLMHandler):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            api_key=api_key,
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
