from abc import ABC, abstractmethod
from typing import List
from models.property import PropertyMatch

class LLMHandler(ABC):
    @abstractmethod
    async def generate_response(self, query: str, matches: List[PropertyMatch]) -> str:
        """Generate response based on property matches"""
        pass

    def _create_system_prompt(self) -> str:
        return """You are a knowledgeable and helpful real estate assistant. 
        Your goal is to help users find properties that match their requirements.
        Always be clear and concise in your recommendations while highlighting key features that match the user's query.
        If discussing prices, be sure to mention both the price and the payment frequency (sale/month).
        Be honest about both advantages and limitations of each property."""

    def _create_property_context(self, matches: List[PropertyMatch]) -> str:
        properties_text = []
        
        for i, match in enumerate(matches, 1):
            prop = match.property
            similarity_percentage = round(match.similarity * 100, 1)
            
            property_text = f"""
Property {i} (Match score: {similarity_percentage}%):
Name: {prop.property_name}
Type: {prop.type}
Location: {prop.town}, {prop.province or prop.country}
Price: {prop.price} {prop.currency} ({prop.price_freq})
Details: {prop.beds} bedrooms, {prop.baths} bathrooms
Area: {prop.surface_area_built}m² built
Features: {', '.join(prop.features) if prop.features else 'None listed'}
Description: {prop.desc[:300]}...
            """.strip()
            properties_text.append(property_text)
        
        return "\n\n".join(properties_text)
