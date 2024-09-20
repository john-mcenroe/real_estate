import csv
import requests
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException

def google_search(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
            driver.get(f"https://www.google.com/search?q={query}")
            time.sleep(5)  # Increased wait time
            search_results = driver.find_elements(By.CSS_SELECTOR, 'div.g a')
            urls = [result.get_attribute('href') for result in search_results]
            driver.quit()
            return urls
        except WebDriverException as e:
            print(f"WebDriver error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Returning empty list.")
                return []
            time.sleep(2 ** attempt)  # Exponential backoff
        finally:
            try:
                driver.quit()
            except:
                pass
    return []

def find_myhome_links(urls):
    return [url for url in urls if url and 'myhome.ie' in url]

def safe_get_text(driver, selector, by=By.CSS_SELECTOR, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
        return element.text.strip()
    except (TimeoutException, NoSuchElementException):
        print(f"Element not found: {selector}")
        return ""

def parse_listing(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    
    data = {'MyHome_Link': url}
    
    try:
        # Wait for the page to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.h4.fw-bold'))
        )
        
        # Try to handle the privacy popup
        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I ACCEPT')]"))
            ).click()
            time.sleep(2)
        except:
            print("Privacy popup not found or could not be handled.")

        # Extract data
        data['MyHome_Address'] = safe_get_text(driver, 'h1.h4.fw-bold')
        data['MyHome_Asking_Price'] = safe_get_text(driver, 'b.brochure__price')
        data['MyHome_Beds'] = safe_get_text(driver, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'beds')]", By.XPATH)
        data['MyHome_Baths'] = safe_get_text(driver, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'baths')]", By.XPATH)

        floor_area_element = driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]")
        sup_elements = floor_area_element.find_elements(By.TAG_NAME, 'sup')
        sup_text = safe_get_text(driver, 'sup', By.TAG_NAME) if sup_elements else ''
        floor_area_value = safe_get_text(driver, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]", By.XPATH).replace(sup_text, '') + sup_text
        data['MyHome_Floor_Area_Value'] = floor_area_value

        ber_info = safe_get_text(driver, 'p.brochure__details--description-content')
        if ber_info:
            ber_parts = ber_info.split('\n')
            data['MyHome_BER_Rating'] = ber_parts[0].replace('BER:', '').strip() if len(ber_parts) > 0 else ''
            data['MyHome_BER_Number'] = ber_parts[1].replace('BER No:', '').strip() if len(ber_parts) > 1 else ''
            data['MyHome_Energy_Performance_Indicator'] = ber_parts[2].replace('Energy Performance Indicator:', '').strip() if len(ber_parts) > 2 else ''

        # Set default values for fields that might not always be present
        data['MyHome_Monthly_Price'] = '0'
        data['MyHome_Floor_Area_Unit'] = 'm²'
        data['MyHome_Publish_Date'] = ''
        data['MyHome_Sale_Type'] = ''
        data['MyHome_Category'] = ''
        data['MyHome_Featured_Level'] = ''

    except Exception as e:
        print(f"Error extracting data: {e}")
    
    driver.quit()
    return data

def get_latitude_longitude(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    params = {
        "address": address,
        "key": api_key
    }
    
    response = requests.get(base_url, params=params)
    results = response.json()

    if results["status"] == "OK":
        location = results["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        print(f"Error fetching data: {results['status']}")
        return None, None

def process_csv(input_file, output_file, api_key, limit=3):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        myhome_fields = [
            'MyHome_Address', 'MyHome_Asking_Price', 'MyHome_Beds', 'MyHome_Baths', 
            'MyHome_Floor_Area_Value', 'MyHome_BER_Rating', 'MyHome_BER_Number', 
            'MyHome_Energy_Performance_Indicator', 'MyHome_Latitude', 'MyHome_Longitude', 
            'MyHome_Monthly_Price', 'MyHome_Floor_Area_Unit', 'MyHome_Publish_Date', 
            'MyHome_Sale_Type', 'MyHome_Category', 'MyHome_Featured_Level', 'MyHome_Link'
        ]
        fieldnames = reader.fieldnames + myhome_fields
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            if i >= limit:
                break
            
            address = row['Address']
            print(f"Processing record {i+1}: {address}")
            
            try:
                urls = google_search(address)
                myhome_links = find_myhome_links(urls)

                if myhome_links:
                    print(f"Found MyHome.ie link: {myhome_links[0]}")
                    listing_data = parse_listing(myhome_links[0])
                    row.update(listing_data)
                    
                    # Get latitude and longitude
                    lat, lng = get_latitude_longitude(address, api_key)
                    row['MyHome_Latitude'] = lat
                    row['MyHome_Longitude'] = lng
                else:
                    print("No MyHome.ie link found")
                    # Add empty values for MyHome fields
                    for field in myhome_fields:
                        row[field] = ''
            except Exception as e:
                print(f"Error processing record: {e}")
                # Add empty values for MyHome fields
                for field in myhome_fields:
                    row[field] = ''
            
            writer.writerow(row)
            print(f"Processed: {address}")
            print("---")

if __name__ == "__main__":
    input_file = "scraped_dublin/scraped_property_results_Dublin_page_1.csv"
    output_file = "scraped_dublin_metadata/scraped_property_results_metadata_Dublin_page_1_test.csv"
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        print("Error: Google Maps API key not found. Please set the GOOGLE_MAPS_API_KEY environment variable.")
    else:
        process_csv(input_file, output_file, api_key, limit=3)
        print("Processing complete. Results saved to", output_file)