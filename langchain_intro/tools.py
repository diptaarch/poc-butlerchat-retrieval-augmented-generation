import random
import time

def get_hotel_information(hotel_code: str) -> dict:
    """Get hotel information for Archipelago International"""
    
    valid_hotels = {
        "aston_jakarta": {
            "name": "ASTON Madiun",
            "location": "Jakarta, Indonesia",
            "brand": "ASTON",
            "rooms": 156,
            "amenities": ["Fitness Center", "Swimming Pool", "Business Center", "Restaurant", "Bar"]
        },
        "aston_bali": {
            "name": "ASTON Denpasar Hotel & Convention Center",
            "location": "Bali, Indonesia",
            "brand": "ASTON",
            "rooms": 217,
            "amenities": ["Fitness Center", "Swimming Pool", "Conference Halls", "Restaurant", "Spa"]
        },
        "huxley_jakarta": {
            "name": "Huxley Jakarta",
            "location": "Jakarta, Indonesia",
            "brand": "Huxley",
            "rooms": 89,
            "amenities": ["Boutique Design", "Rooftop Bar", "Fitness Center", "Business Lounge"]
        },
        "alana_jakarta": {
            "name": "ALANA Jakarta Hotel",
            "location": "Jakarta, Indonesia",
            "brand": "ALANA",
            "rooms": 172,
            "amenities": ["Premium Pool", "Fine Dining", "Spa", "Business Center", "Fitness Center"]
        }
    }
    
    if hotel_code.lower() not in valid_hotels:
        return {"error": f"Hotel code '{hotel_code}' not found. Available: {', '.join(valid_hotels.keys())}"}
    
    # Simulate API call delay
    time.sleep(0.5)
    
    return valid_hotels[hotel_code.lower()]


def get_brand_details(brand_name: str) -> dict:
    """Get detailed information about an Archipelago brand"""
    
    brands = {
        "aston": {
            "name": "ASTON",
            "description": "Flagship brand of Archipelago International, launched in 1997. Diverse portfolio from luxury 5-star Grand ASTON to mid-range ASTON hotels.",
            "positioning": "Full-service hotels",
            "properties_count": "50+",
            "target_market": "Business and leisure travelers"
        },
        "huxley": {
            "name": "Huxley",
            "description": "Ultra-chic and sophisticated brand for discerning travelers seeking modern luxury and contemporary design.",
            "positioning": "Luxury lifestyle",
            "properties_count": "15+",
            "target_market": "Premium travelers"
        },
        "alana": {
            "name": "ALANA",
            "description": "Premium brand offering refined hospitality experiences with elegant accommodations and world-class facilities.",
            "positioning": "Premium hospitality",
            "properties_count": "20+",
            "target_market": "Upscale leisure travelers"
        },
        "kamuela": {
            "name": "Kamuela",
            "description": "Upscale resort properties with distinctive charm and personalized service across Southeast Asia.",
            "positioning": "Resort hospitality",
            "properties_count": "12+",
            "target_market": "Resort and leisure guests"
        },
        "favehotel": {
            "name": "FAVE Hotel",
            "description": "Lifestyle brand offering vibrant, youthful hospitality experiences with trendy design.",
            "positioning": "Lifestyle budget",
            "properties_count": "30+",
            "target_market": "Young travelers and backpackers"
        }
    }
    
    if brand_name.lower() not in brands:
        return {"error": f"Brand '{brand_name}' not found. Available: {', '.join(brands.keys())}"}
    
    # Simulate API call delay
    time.sleep(0.5)
    
    return brands[brand_name.lower()]
