import pandas as pd
from daftlistings import Daft, Location, SearchType, PropertyType, SortType

def get_largest_image(images):
    """Get the URL of the largest image from the images dictionary."""
    sizes = ['size720x480', 'size600x600', 'size400x300', 'size360x240', 'size300x200', 'size320x280', 'size680x392']
    for size in sizes:
        if size in images:
            return images[size]
    return None  # Return None if no image is found

daft = Daft()
daft.search(max_pages=1)
daft.set_location(Location.DUBLIN)
daft.set_search_type(SearchType.RESIDENTIAL_SALE)
daft.set_property_type(PropertyType.HOUSE)
daft.set_sort_type(SortType.PUBLISH_DATE_DESC)

listings = daft.search()

property_data = []

for listing in listings:
    title = listing._result.get('title', 'No Title')
    
    images = listing._result.get('media', {}).get('images', [])
    image_urls = [get_largest_image(img) for img in images if get_largest_image(img)]
    
    property_data.append({
        'title': title,
        'images': image_urls
    })

# Create a DataFrame
df = pd.DataFrame(property_data)

# Save to CSV
csv_filename = 'dublin_property_images.csv'
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# Print the first few rows of the DataFrame
print(df.head())

# Print the results
""" for property in property_data:
    print(f"Title: {property['title']}")
    print("Images:")
    for url in property['images']:
        print(f"  - {url}")
    print()  # Empty line for readability between properties """
    
