from typing import Optional, List
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
    desc: str
    features: List[str] = Field(default_factory=list)
    pool: bool = False
    property_name: str
    
    def to_embedding_text(self) -> str:
        """Convert property to text for embedding"""
        features_text = ", ".join(self.features) if self.features else "No special features"
        return f"""
        Property Name: {self.property_name}
        Type: {self.type}
        Location: {self.town}, {self.province if self.province else ''}, {self.country}
        Price: {self.price} {self.currency} ({self.price_freq})
        Details: {self.beds} bedrooms, {self.baths} bathrooms
        Area: {self.surface_area_built}m² built
        Features: {features_text}
        Pool: {'Yes' if self.pool else 'No'}
        Description: {self.desc}
        """.strip()
    
    def to_display_text(self) -> str:
        """Convert property to display format"""
        return f"""
🏠 {self.property_name}
💰 Price: {self.price} {self.currency}
📍 Location: {self.town}, {self.province if self.province else self.country}
🛏️ Bedrooms: {self.beds if self.beds else 'N/A'}
🚿 Bathrooms: {self.baths if self.baths else 'N/A'}
📐 Area: {self.surface_area_built}m²
✨ Features: {', '.join(self.features) if self.features else 'No special features'}
🏊‍♂️ Pool: {'Yes' if self.pool else 'No'}
🔍 Reference: {self.ref}
        """.strip()

class PropertyMatch(BaseModel):
    property: Property
    similarity: float
