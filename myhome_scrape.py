from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def safe_get_text(element):
    return element.text.strip() if element else ""

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
        data['MyHome_Address'] = safe_get_text(driver.find_element(By.CSS_SELECTOR, 'h1.h4.fw-bold'))

        # Extracting Asking Price
        data['MyHome_Asking_Price'] = safe_get_text(driver.find_element(By.CSS_SELECTOR, 'b.brochure__price'))

        # Extracting Beds and Baths (handling the <span> elements correctly)
        data['MyHome_Beds'] = safe_get_text(driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'beds')]"))
        data['MyHome_Baths'] = safe_get_text(driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'baths')]"))

        # Extracting Floor Area (handling the separate <sup> element)
        floor_area_element = driver.find_element(By.XPATH, "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]")

        # Check if <sup> exists and handle it
        sup_elements = floor_area_element.find_elements(By.TAG_NAME, 'sup')  # Find all <sup> elements within the floor area element
        sup_text = safe_get_text(sup_elements[0]) if sup_elements else ''  # Use the first <sup> element if found

        # Extract and format the floor area value
        floor_area_value = safe_get_text(floor_area_element).replace(sup_text, '') + sup_text
        data['MyHome_Floor_Area_Value'] = floor_area_value

        # Extracting BER Rating (handling the <img> element)
        ber_image_element = driver.find_element(By.XPATH, "//img[contains(@class, 'info-strip--divider') and contains(@alt, 'Energy Rating')]")
        ber_rating = ber_image_element.get_attribute('src').split('/')[-1].split('.')[0]  # Extract from image URL
        data['MyHome_BER_Rating'] = ber_rating

        # Extracting BER Number, and Energy Performance Indicator (if available)
        ber_info = safe_get_text(driver.find_element(By.CSS_SELECTOR, 'p.brochure__details--description-content'))
        if ber_info:
            ber_parts = ber_info.split('\n')
            if len(ber_parts) >= 2:  # Ensure enough parts are present
                data['MyHome_BER_Number'] = ber_parts[0].replace('BER No:', '').strip()
                data['MyHome_Energy_Performance_Indicator'] = ber_parts[1].replace('Energy Performance Indicator:', '').strip()

        # Add placeholders for fields you can't extract immediately
        data['MyHome_Latitude'] = ''
        data['MyHome_Longitude'] = ''
        data['MyHome_Monthly_Price'] = 0
        data['MyHome_Floor_Area_Unit'] = 'm²'  # Assumed from the information in your HTML
        data['MyHome_Publish_Date'] = ''
        data['MyHome_Sale_Type'] = ''
        data['MyHome_Category'] = ''
        data['MyHome_Featured_Level'] = ''
        data['MyHome_Link'] = url

    except Exception as e:
        print(f"Error extracting data: {e}")

    # Close the browser
    driver.quit()
    return data

# Example usage
url = "https://www.myhome.ie/residential/brochure/taramar-middle-third-killester-dublin-5/4786863"
listing_data = parse_listing(url)

# Output the listing data
print(listing_data)