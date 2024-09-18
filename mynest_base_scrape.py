import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize the Chrome WebDriver using webdriver-manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Navigate to the URL
region = "Dublin"  # Parameterized region name
url = f"https://mynest.ie/priceregister/{region}"
driver.get(url)

# Explicitly wait for the table to load
wait = WebDriverWait(driver, 10)

# Set to store unique addresses and URLs
unique_addresses = set()

def extract_data():
    # Keep track of which rows have been processed
    processed_addresses = set()

    while True:
        try:
            # Wait for the rows to load
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')))
            
            # Re-fetch the list of rows after every navigation
            rows = driver.find_elements(By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')
            
            # Process each row one by one
            for row in rows:
                address = row.text.strip()  # Extract the text and clean up spaces

                # Skip the row if it's already processed
                if address in processed_addresses:
                    continue

                try:
                    print(f"Processing: {address}")
                    
                    # Click the address to go to the detailed page
                    row.click()

                    # Wait for 2 seconds to ensure the page is loaded
                    time.sleep(1)

                    # Get the current URL
                    url = driver.current_url

                    # Store the URL and address
                    unique_addresses.add((address, url))
                    print(f"Address: {address}, URL: {url}")  # Print for feedback

                    # Add the address to processed list
                    processed_addresses.add(address)

                    # Go back to the main listing page
                    driver.back()

                    # Wait for the rows to reload
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')))
                    
                    # Pause to ensure the page is fully loaded
                    time.sleep(2)

                except Exception as e:
                    print(f"Error processing {address}: {e}")

            # Exit the loop once all rows have been processed
            break

        except Exception as e:
            print(f"Error locating elements: {e}")
            continue  # In case of error, re-attempt to locate elements

# Scrape the data
extract_data()

# Close the driver after scraping is done
driver.quit()

# Save the output results to a CSV file
csv_filename = f"scraped_property_results_{region}.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Address', 'URL'])  # Write the header
    for address, url in unique_addresses:
        writer.writerow([address, url])

print(f"\nUnique Extracted Addresses and URLs saved to {csv_filename}")
