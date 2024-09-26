## Active in use
## Goes through defined CSV file and iteratively gets myhome values for every address in the file. 

import csv
import requests
import os
import re
import logging
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def google_search(query, max_retries=3):
    options = webdriver.ChromeOptions()
    # Uncomment the next line to run Chrome in headless mode
    # options.add_argument('--headless')
    driver_service = Service(ChromeDriverManager().install())

    for attempt in range(max_retries):
        try:
            driver = webdriver.Chrome(service=driver_service, options=options)
            driver.get(f"https://www.google.com/search?q={query}")

            # Wait until search results are present
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.g'))
            )

            search_results = driver.find_elements(By.CSS_SELECTOR, 'div.g a')
            urls = [result.get_attribute('href') for result in search_results]
            driver.quit()
            return urls
        except Exception as e:
            logging.error(f"Error during Google search on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached. Returning empty list.")
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
        logging.warning(f"Element not found: {selector}")
        return ""

def parse_listing(url):
    options = webdriver.ChromeOptions()
    # Uncomment the next line to run Chrome in headless mode
    # options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    data = {'MyHome_Link': url}

    try:
        # Wait for the page to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.h4.fw-bold'))
        )

        # Handle the privacy popup
        try:
            accept_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I ACCEPT')]"))
            )
            accept_button.click()
        except TimeoutException:
            logging.info("Privacy popup not found or already handled.")
        except Exception as e:
            logging.error(f"Error handling privacy popup: {e}")

        # Extract data
        data['MyHome_Address'] = safe_get_text(driver, 'h1.h4.fw-bold')
        data['MyHome_Asking_Price'] = safe_get_text(driver, 'b.brochure__price')
        data['MyHome_Beds'] = safe_get_text(driver, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'beds')]", By.XPATH)
        data['MyHome_Baths'] = safe_get_text(driver, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'baths')]", By.XPATH)

        # Floor area parsing
        try:
            floor_area_element = driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]")
            floor_area_text = floor_area_element.text.strip()
            match = re.search(r'([\d.,]+)\s*m', floor_area_text)
            if match:
                floor_area_value = match.group(1)
                data['MyHome_Floor_Area_Value'] = floor_area_value
                data['MyHome_Floor_Area_Unit'] = 'm²'
            else:
                data['MyHome_Floor_Area_Value'] = ''
                data['MyHome_Floor_Area_Unit'] = ''
        except NoSuchElementException:
            data['MyHome_Floor_Area_Value'] = ''
            data['MyHome_Floor_Area_Unit'] = ''

        # Extract BER information
        try:
            ber_elements = driver.find_elements(By.XPATH, "//p[contains(@class, 'brochure__details--description-content')]")
            data['MyHome_BER_Rating'] = ''
            data['MyHome_BER_Number'] = ''
            data['MyHome_Energy_Performance_Indicator'] = ''
            for ber_element in ber_elements:
                ber_text = ber_element.text.strip()
                if 'BER' in ber_text:
                    ber_parts = ber_text.split('\n')
                    for part in ber_parts:
                        if 'BER:' in part:
                            data['MyHome_BER_Rating'] = part.replace('BER:', '').strip()
                        elif 'BER No:' in part:
                            data['MyHome_BER_Number'] = part.replace('BER No:', '').strip()
                        elif 'Energy Performance Indicator:' in part:
                            data['MyHome_Energy_Performance_Indicator'] = part.replace('Energy Performance Indicator:', '').strip()
                    break
        except Exception as e:
            logging.error(f"Error extracting BER info: {e}")

        # Set default values for optional fields
        data.setdefault('MyHome_Monthly_Price', '0')
        data.setdefault('MyHome_Publish_Date', '')
        data.setdefault('MyHome_Sale_Type', '')
        data.setdefault('MyHome_Category', '')
        data.setdefault('MyHome_Featured_Level', '')

    except Exception as e:
        logging.error(f"Error extracting data: {e}")
    finally:
        driver.quit()
    return data

def get_latitude_longitude(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    try:
        response = requests.get(base_url, params=params)
        results = response.json()
        if results["status"] == "OK":
            location = results["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            logging.error(f"Error fetching geocode data for '{address}': {results['status']}")
            return None, None
    except Exception as e:
        logging.error(f"Exception during geocoding for '{address}': {e}")
        return None, None

def create_snapshot(output_file, snapshot_number):
    snapshot_dir = os.path.join(os.path.dirname(output_file), 'snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_file = os.path.join(snapshot_dir, f'snapshot_{snapshot_number}.csv')
    shutil.copy2(output_file, snapshot_file)
    logging.info(f"Created snapshot: {snapshot_file}")

def process_csv(input_file, output_file, api_key, limit=500):
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
            logging.info(f"Processing record {i+1}: {address}")

            try:
                urls = google_search(address)
                myhome_links = find_myhome_links(urls)

                if myhome_links:
                    logging.info(f"Found MyHome.ie link: {myhome_links[0]}")
                    listing_data = parse_listing(myhome_links[0])
                    row.update(listing_data)

                    # Get latitude and longitude
                    lat, lng = get_latitude_longitude(address, api_key)
                    row['MyHome_Latitude'] = lat
                    row['MyHome_Longitude'] = lng
                else:
                    logging.warning("No MyHome.ie link found")
                    # Add empty values for MyHome fields
                    for field in myhome_fields:
                        if field not in row:
                            row[field] = ''
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                # Add empty values for MyHome fields
                for field in myhome_fields:
                    if field not in row:
                        row[field] = ''

            writer.writerow(row)
            logging.info(f"Processed: {address}")
            logging.info("---")

            # Create a snapshot every 10 records
            if (i + 1) % 10 == 0:
                create_snapshot(output_file, (i + 1) // 10)

        # Create a final snapshot if the total number of records is not divisible by 10
        if (i + 1) % 10 != 0:
            create_snapshot(output_file, ((i + 1) // 10) + 1)

if __name__ == "__main__":
    input_file = "data/processed/scraped_dublin/scraped_property_results_Dublin_page_24.csv"
    output_file = "data/processed/scraped_dublin_metadata/scraped_property_results_metadata_Dublin_page_1.csv"
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')

    if not api_key:
        logging.error("Google Maps API key not found. Please set the GOOGLE_MAPS_API_KEY environment variable.")
    else:
        process_csv(input_file, output_file, api_key, limit=2500)
        logging.info(f"Processing complete. Results saved to {output_file}")