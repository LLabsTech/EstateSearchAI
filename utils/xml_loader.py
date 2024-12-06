from typing import List
import pandas as pd
from models.property import Property

def load_properties_from_xml(file_path: str) -> List[Property]:
    """Load and parse properties from XML file"""
    df = pd.read_xml(file_path, parser="etree")
    properties = []
    
    for _, row in df.iterrows():
        try:
            # Extract surface area
            surface_area = row.get('surface_area', {})
            if isinstance(surface_area, dict):
                surface_built = float(surface_area.get('built', 0))
                surface_plot = float(surface_area.get('plot', 0))
            else:
                surface_built = surface_plot = 0
                
            # Extract features
            features = row.get('features', {})
            feature_list = []
            if isinstance(features, dict) and 'feature' in features:
                if isinstance(features['feature'], list):
                    feature_list = features['feature']
                else:
                    feature_list = [features['feature']]
                    
            # Extract description
            desc = row.get('desc', {})
            if isinstance(desc, dict) and 'es' in desc:
                description = desc['es']
            else:
                description = str(desc)
            
            property_data = Property(
                id=str(row['id']),
                date=str(row['date']),
                ref=str(row['ref']),
                price=float(row['price']),
                currency=str(row['currency']),
                price_freq=str(row['price_freq']),
                new_build=bool(row['new_build']),
                type=str(row['type']),
                town=str(row['town']),
                province=str(row['province']) if 'province' in row else None,
                country=str(row['country']),
                beds=int(row['beds']) if 'beds' in row and pd.notna(row['beds']) else None,
                baths=int(row['baths']) if 'baths' in row and pd.notna(row['baths']) else None,
                surface_area_built=surface_built,
                surface_area_plot=surface_plot,
                desc=description,
                features=feature_list,
                pool=bool(row['pool']),
                property_name=str(row['property_name'])
            )
            
            properties.append(property_data)
            
        except Exception as e:
            print(f"Error processing property {row.get('id', 'unknown')}: {str(e)}")
            continue
    
    return properties
