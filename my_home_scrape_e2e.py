import csv
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Google Maps Geocoding API function to get latitude and longitude
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

# Function to perform a Google search and extract the results
def google_search(query):
    # Set up Selenium Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Navigate to Google
    driver.get(f"https://www.google.com/search?q={query}")
    
    # Wait for the page to load
    time.sleep(3)
    
    # Find the search result links
    search_results = driver.find_elements(By.CSS_SELECTOR, 'div.g a')

    urls = []
    for result in search_results:
        url = result.get_attribute('href')
        urls.append(url)
    
    # Close the browser
    driver.quit()

    return urls

# Function to check if any of the URLs are from myhome.ie
def find_myhome_links(urls):
    myhome_links = [url for url in urls if 'myhome.ie' in url]
    return myhome_links

# Function to safely extract text from web elements
def safe_get_text(element):
    return element.text.strip() if element else ""

# Function to safely find elements
def safe_find_element(driver, by, value):
    try:
        return driver.find_element(by, value)
    except:
        return None

# Function to parse and extract details from the MyHome property page
def parse_listing(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    data = {}

    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I ACCEPT')]"))
        ).click()
        time.sleep(2)
    except:
        pass

    try:
        data['Address'] = safe_get_text(safe_find_element(driver, By.CSS_SELECTOR, 'h1.h4.fw-bold'))
        data['Asking Price'] = safe_get_text(safe_find_element(driver, By.CSS_SELECTOR, 'b.brochure__price'))
        data['Beds'] = safe_get_text(safe_find_element(driver, By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'beds')]"))
        data['Baths'] = safe_get_text(safe_find_element(driver, By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'baths')]"))

        # Floor Area
        floor_area_element = safe_find_element(driver, By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]")
        if floor_area_element:
            sup_element = safe_find_element(floor_area_element, By.TAG_NAME, 'sup')
            data['Floor Area Value'] = safe_get_text(floor_area_element).replace(safe_get_text(sup_element), '') + safe_get_text(sup_element)

        # BER Rating
        ber_info = safe_get_text(safe_find_element(driver, By.CSS_SELECTOR, 'p.brochure__details--description-content'))
        if ber_info:
            ber_parts = ber_info.split('\n')
            data['BER Rating'] = ber_parts[0].replace('BER:', '').strip()
            data['BER Number'] = ber_parts[1].replace('BER No:', '').strip()
            data['Energy Performance Indicator'] = ber_parts[2].replace('Energy Performance Indicator:', '').strip()

    except Exception as e:
        print(f"Error extracting data: {e}")
    
    driver.quit()
    return data

# Main function to read CSV, search for myhome.ie links, scrape data, and get latitude/longitude
def process_csv(input_csv, output_csv, api_key):
    with open(input_csv, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Open the output CSV file
        with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            fieldnames = [
                'Address', 'Asking Price', 'Beds', 'Baths', 'Property Type',
                'Energy Rating', 'Eircode', 'Local Property Tax', 'Agency Name',
                'Agency Contact', 'Price Changes', 'MyHome Link', 'Scraped Beds',
                'Scraped Baths', 'Scraped Asking Price', 'Floor Area Value',
                'BER Rating', 'BER Number', 'Energy Performance Indicator',
                'Latitude', 'Longitude'
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                address = row['Address']
                print(f"Searching for MyHome link for address: {address}")

                # Perform Google search for the address
                search_query = f"{address} myhome.ie"
                urls = google_search(search_query)
                myhome_links = find_myhome_links(urls)

                if myhome_links:
                    # Use the first MyHome link found
                    myhome_link = myhome_links[0]
                    print(f"Found MyHome link: {myhome_link}")

                    # Scrape the property data from the MyHome link
                    scraped_data = parse_listing(myhome_link)

                    # Get latitude and longitude using the Google Maps Geocoding API
                    latitude, longitude = get_latitude_longitude(address, api_key)

                    # Write both original and scraped data to the output CSV
                    row['MyHome Link'] = myhome_link
                    row['Scraped Beds'] = scraped_data.get('Beds', '')
                    row['Scraped Baths'] = scraped_data.get('Baths', '')
                    row['Scraped Asking Price'] = scraped_data.get('Asking Price', '')
                    row['Floor Area Value'] = scraped_data.get('Floor Area Value', '')
                    row['BER Rating'] = scraped_data.get('BER Rating', '')
                    row['BER Number'] = scraped_data.get('BER Number', '')
                    row['Energy Performance Indicator'] = scraped_data.get('Energy Performance Indicator', '')
                    row['Latitude'] = latitude if latitude else ''
                    row['Longitude'] = longitude if longitude else ''

                    writer.writerow(row)
                else:
                    print(f"No MyHome link found for address: {address}")
                    row['MyHome Link'] = ''
                    writer.writerow(row)

# Call the function with the input CSV, desired output CSV, and your API key
api_key = "AIzaSyBiAEpA-8G_WDKPH4lnnon95tbyYIRgkbM"
process_csv('property_data.csv', 'output_property_data.csv', api_key)
