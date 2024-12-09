from typing import List
import xml.etree.ElementTree as ET
from models.property import Property

def clean_numeric(value: str) -> float:
    """Convert string numbers to float, handling commas"""
    if value is None:
        return 0.0
    try:
        # Replace comma with dot and convert to float
        return float(str(value).replace(',', '.'))
    except (ValueError, AttributeError):
        return 0.0

def load_properties_from_xml(file_path: str) -> List[Property]:
    """Load and parse properties from XML file"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    properties = []
    
    for prop_elem in root.findall('.//property'):
        try:
            # Extract surface area
            surface_area_elem = prop_elem.find('surface_area')
            surface_built = surface_plot = 0.0
            if surface_area_elem is not None:
                built_elem = surface_area_elem.find('built')
                plot_elem = surface_area_elem.find('plot')
                if built_elem is not None and built_elem.text:
                    surface_built = clean_numeric(built_elem.text)
                if plot_elem is not None and plot_elem.text:
                    surface_plot = clean_numeric(plot_elem.text)
            
            # Extract features
            features_elem = prop_elem.find('features')
            feature_list = []
            if features_elem is not None:
                for feature in features_elem.findall('feature'):
                    if feature is not None and feature.text:
                        feature_list.append(feature.text.strip())
            
            # Extract description
            desc = {'es': ''}  # Default empty Spanish description
            desc_elem = prop_elem.find('desc')
            if desc_elem is not None:
                es_elem = desc_elem.find('es')
                if es_elem is not None and es_elem.text:
                    desc['es'] = es_elem.text.strip()
            
            # Extract images
            images = []
            images_elem = prop_elem.find('images')
            if images_elem is not None:
                for img in images_elem.findall('image'):
                    url_elem = img.find('url')
                    if url_elem is not None and url_elem.text:
                        if url_elem.text.startswith('http'):
                            images.append({'url': url_elem.text})
            
            # Create property object
            property_data = Property(
                id=prop_elem.find('id').text.strip() if prop_elem.find('id') is not None else '',
                date=prop_elem.find('date').text.strip() if prop_elem.find('date') is not None else '',
                ref=prop_elem.find('ref').text.strip() if prop_elem.find('ref') is not None else '',
                price=clean_numeric(prop_elem.find('price').text) if prop_elem.find('price') is not None else 0.0,
                currency=prop_elem.find('currency').text.strip() if prop_elem.find('currency') is not None else '',
                price_freq=prop_elem.find('price_freq').text.strip() if prop_elem.find('price_freq') is not None else '',
                new_build=bool(int(prop_elem.find('new_build').text)) if prop_elem.find('new_build') is not None else False,
                type=prop_elem.find('type').text.strip() if prop_elem.find('type') is not None else '',
                town=prop_elem.find('town').text.strip() if prop_elem.find('town') is not None else '',
                province=prop_elem.find('province').text.strip() if prop_elem.find('province') is not None else None,
                country=prop_elem.find('country').text.strip() if prop_elem.find('country') is not None else '',
                beds=int(prop_elem.find('beds').text) if prop_elem.find('beds') is not None else None,
                baths=int(prop_elem.find('baths').text) if prop_elem.find('baths') is not None else None,
                surface_area_built=surface_built,
                surface_area_plot=surface_plot,
                desc=desc,
                features=feature_list,
                pool=bool(int(prop_elem.find('pool').text)) if prop_elem.find('pool') is not None else False,
                property_name=prop_elem.find('property_name').text.strip() if prop_elem.find('property_name') is not None else '',
                images=images
            )
            
            properties.append(property_data)
            
        except Exception as e:
            print(f"Error processing property {prop_elem.find('id').text if prop_elem.find('id') is not None else 'unknown'}: {str(e)}")
            continue
    
    return properties