from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class Property(BaseModel):
    id: str
    date: str
    ref: str
    price: float
    currency: str
    price_freq: str
    new_build: bool = False
    type: str
    town: str
    province: Optional[str]
    country: str
    beds: Optional[int]
    baths: Optional[int]
    surface_area_built: Optional[float]
    surface_area_plot: Optional[float]
    desc: Dict[str, str] = Field(default_factory=dict)
    features: List[str] = Field(default_factory=list)
    pool: bool = False
    property_name: str
    images: List[Dict[str, str]] = Field(default_factory=list)
    
    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    def to_embedding_text(self) -> str:
        """Convert property to text for embedding"""
        features_text = ", ".join(self.features) if self.features else "No special features"
        description = self.desc.get('es', '') if self.desc else ''
        
        # Build area text
        area_parts = []
        if self.surface_area_built:
            area_parts.append(f"{self.surface_area_built}m² built")
        if self.surface_area_plot:
            area_parts.append(f"{self.surface_area_plot}m² plot")
        area_text = ", ".join(area_parts) if area_parts else "Area N/A"
        
        return f"""
        Property Name: {self.property_name}
        Type: {self.type}
        Location: {self.town}, {self.province if self.province else ''}, {self.country}
        Price: {self.price} {self.currency} ({self.price_freq})
        Details: {self.beds} bedrooms, {self.baths} bathrooms
        Area: {area_text}
        Features: {features_text}
        Pool: {'Yes' if self.pool else 'No'}
        Description: {description}
        """.strip()
    
    def to_display_text(self) -> str:
        """Convert property to display format with markdown"""
        # Build area text
        area_parts = []
        if self.surface_area_built:
            area_parts.append(f"{self.surface_area_built}m² built")
        if self.surface_area_plot:
            area_parts.append(f"{self.surface_area_plot}m² plot")
        area_text = ", ".join(area_parts) if area_parts else "N/A"
        
        # Escape all text values for markdown
        name = self._escape_markdown(str(self.property_name))
        price = self._escape_markdown(str(self.price))
        currency = self._escape_markdown(str(self.currency))
        price_freq = self._escape_markdown(str(self.price_freq))
        town = self._escape_markdown(str(self.town))
        province = self._escape_markdown(str(self.province)) if self.province else ''
        country = self._escape_markdown(str(self.country))
        features = [self._escape_markdown(f) for f in self.features]
        ref = self._escape_markdown(str(self.ref))
        area_text = self._escape_markdown(area_text)
        
        return f"""
*🏠 {name}*
*💰 Price:* {price} {currency}{' per ' + price_freq if price_freq != 'sale' else ''}
*📍 Location:* {town}, {province if province else country}
*🛏️ Bedrooms:* {self.beds if self.beds else 'N/A'}
*🚿 Bathrooms:* {self.baths if self.baths else 'N/A'}
*📐 Area:* {area_text}
*✨ Features:* {', '.join(features) if features else 'No special features'}
*🏊‍♂️ Pool:* {'Yes' if self.pool else 'No'}
*🔍 Reference:* {ref}
        """.strip()

class PropertyMatch(BaseModel):
    property: Property
    similarity: float