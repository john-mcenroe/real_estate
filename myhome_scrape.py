from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Function to safely extract text from web elements
def safe_get_text(element):
    return element.text.strip() if element else ""

# Function to parse and extract details from the MyHome property page
def parse_listing(url):
    # Initialize the Chrome driver using webdriver-manager
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Open the URL
    driver.get(url)

    # Wait for the page to load or wait for the privacy consent button
    try:
        # Wait until the privacy consent button is available and click it
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I ACCEPT')]"))
        ).click()

        print("Privacy popup accepted")

        # Optionally, wait a bit for the page to fully load after the popup is dismissed
        time.sleep(2)

    except Exception as e:
        print(f"Error accepting privacy popup: {e}")
        driver.quit()
        return {}

    # Initialize an empty dictionary for the listing data
    data = {}

    try:
        # Extracting Address
        data['Address'] = safe_get_text(driver.find_element(By.CSS_SELECTOR, 'h1.h4.fw-bold'))

        # Extracting Asking Price
        data['Asking Price'] = safe_get_text(driver.find_element(By.CSS_SELECTOR, 'b.brochure__price'))

        # Extracting Beds and Baths
        data['Beds'] = safe_get_text(driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'beds')]"))
        data['Baths'] = safe_get_text(driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'baths')]"))

        # Extracting Floor Area (handling the separate <sup> element)
        floor_area_element = driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]")
        sup_element = floor_area_element.find_element(By.TAG_NAME, 'sup')
        data['Floor Area Value'] = safe_get_text(floor_area_element).replace(safe_get_text(sup_element), '') + safe_get_text(sup_element)

        # Extracting BER Rating, BER Number, and Energy Performance Indicator
        ber_info = safe_get_text(driver.find_element(By.CSS_SELECTOR, 'p.brochure__details--description-content'))
        if ber_info:
            ber_parts = ber_info.split('\n')
            data['BER Rating'] = ber_parts[0].replace('BER:', '').strip()
            data['BER Number'] = ber_parts[1].replace('BER No:', '').strip()
            data['Energy Performance Indicator'] = ber_parts[2].replace('Energy Performance Indicator:', '').strip()

        # Add placeholders for fields you can't extract immediately
        data['Latitude'] = ''
        data['Longitude'] = ''
        data['Monthly Price'] = 0
        data['Floor Area Unit'] = 'm²'  # Assumed from the information in your HTML
        data['Publish Date'] = ''
        data['Sale Type'] = ''
        data['Category'] = ''
        data['Featured Level'] = ''
        data['Daft Link'] = url

    except Exception as e:
        print(f"Error extracting data: {e}")
    
    # Close the browser
    driver.quit()

    return data

# Example usage
url = "https://www.myhome.ie/residential/brochure/13-mapas-avenue-dalkey-co-dublin/4795687"
listing_data = parse_listing(url)

# Output the listing data
print(listing_data)
