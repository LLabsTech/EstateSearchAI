from typing import List
from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from models.property import PropertyMatch
from .base import LLMHandler

class LlamaHandler(LLMHandler):
    def __init__(self, model_path: str):
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            n_ctx=2048,
            verbose=False
        )

    async def generate_response(self, query: str, matches: List[PropertyMatch]) -> str:
        system_prompt = self._create_system_prompt()
        property_context = self._create_property_context(matches)

        # Llama expects a specific format
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

User Query: {query}

Available properties:
{property_context}

Please analyze these properties and suggest the best matches for the user's requirements.[/INST]"""

        response = await self.llm.ainvoke(prompt)
        return response
