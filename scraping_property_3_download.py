import pandas as pd
from daftlistings import Daft, Location, SearchType, PropertyType, SortType

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
            return None
    return obj

daft = Daft()
daft.set_location(Location.DUBLIN)
daft.set_search_type(SearchType.RESIDENTIAL_SALE)
daft.set_property_type(PropertyType.HOUSE)
daft.set_sort_type(SortType.PUBLISH_DATE_DESC)

listings = daft.search()

data = []
for listing in listings:
    data.append({
        "Title": safe_get(listing, 'title'),
        "Price": safe_get(listing, 'price'),
        "Daft Link": safe_get(listing, 'daft_link'),
        "Latitude": safe_get(listing, 'latitude'),
        "Longitude": safe_get(listing, 'longitude'),
        "Monthly Price": safe_get(listing, 'monthly_price') or 0,
        "Bedrooms": safe_get(listing, 'bedrooms'),
        "Bathrooms": safe_get(listing, 'bathrooms'),
        "Floor Area Value": safe_get(listing._result, 'floorArea', 'value'),
        "Floor Area Unit": safe_get(listing._result, 'floorArea', 'unit'),
        "Publish Date": safe_get(listing, 'publish_date'),
        "Sale Type": safe_get(listing, 'sale_type'),
        "BER Rating": safe_get(listing._result, 'ber', 'rating'),
        "BER EPI": safe_get(listing._result, 'ber', 'epi'),
        "Category": safe_get(listing, 'category'),
        "Featured Level": safe_get(listing, 'featured_level'),
        "Agent": safe_get(listing._result, 'seller', 'name')
    })

df = pd.DataFrame(data)

# Save to CSV
csv_filename = 'dublin_house_listings.csv'
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")
