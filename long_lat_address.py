import requests

# Function to get latitude and longitude using Google Maps Geocoding API
def get_latitude_longitude(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    # Set up the parameters for the request
    params = {
        "address": address,
        "key": api_key
    }
    
    # Make a request to the Google Maps Geocoding API
    response = requests.get(base_url, params=params)
    results = response.json()

    # Check if the response contains results
    if results["status"] == "OK":
        # Extract latitude and longitude from the first result
        location = results["results"][0]["geometry"]["location"]
        latitude = location["lat"]
        longitude = location["lng"]
        return latitude, longitude
    else:
        print(f"Error fetching data: {results['status']}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Your Google Maps Geocoding API key
    api_key = "AIzaSyBiAEpA-8G_WDKPH4lnnon95tbyYIRgkbM"
    address = "Apartment 9, Chestnut House, Grace Park Court, Beaumont, Dublin 9, D09KT67"
    
    # Get the latitude and longitude
    lat, lng = get_latitude_longitude(address, api_key)
    
    if lat and lng:
        print(f"The latitude of '{address}' is: {lat}")
        print(f"The longitude of '{address}' is: {lng}")
    else:
        print("Failed to get latitude and longitude for the given address.")
