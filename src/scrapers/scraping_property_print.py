import pandas as pd
from daftlistings import Daft, Location, SearchType, PropertyType, SortType, MapVisualization
import json

def safe_get(obj, *keys):
    """Safely get a value from a nested dictionary or object."""
    for key in keys:
        try:
            if isinstance(obj, dict):
                obj = obj[key]
            else:
                obj = getattr(obj, key)
            if callable(obj):
                obj = obj()
        except Exception:
            return None  # or return 0 if you prefer zero for numeric fields
    return obj

daft = Daft()
daft.search(max_pages=1)
daft.set_location(Location.DUBLIN)
daft.set_search_type(SearchType.RESIDENTIAL_SALE)
daft.set_property_type(PropertyType.HOUSE)
daft.set_sort_type(SortType.PUBLISH_DATE_DESC)

listings = daft.search()

for listing in listings:
    print(f"Title: {safe_get(listing, 'title')}")
    print(f"Price: {safe_get(listing, 'price')}")
    print(f"Daft Link: {safe_get(listing, 'daft_link')}")
    print(f"Latitude: {safe_get(listing, 'latitude')}")
    print(f"Longitude: {safe_get(listing, 'longitude')}")
    print(f"Monthly Price: {safe_get(listing, 'monthly_price') or 0}")  # Use 0 if None
    print(f"Bedrooms: {safe_get(listing, 'bedrooms')}")
    print(f"Bathrooms: {safe_get(listing, 'bathrooms')}")
    print(f"Floor Area Value: {safe_get(listing._result, 'floorArea', 'value')}")
    print(f"Floor Area Unit: {safe_get(listing._result, 'floorArea', 'unit')}")
    print(f"Publish Date: {safe_get(listing, 'publish_date')}")
    print(f"Sale Type: {safe_get(listing, 'sale_type')}")
    print(f"BER Rating: {safe_get(listing._result, 'ber', 'rating')}")
    print(f"BER EPI: {safe_get(listing._result, 'ber', 'epi')}")
    print(f"Category: {safe_get(listing, 'category')}")
    print(f"Featured Level: {safe_get(listing, 'featured_level')}")
    print(f"Agent: {safe_get(listing._result, 'seller', 'name')}")  
    print('-' * 50)  # Add a separator between listings
