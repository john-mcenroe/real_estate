#### ACTIVE IN USE #####
## Goes through mynest starting at some page and gets as much data as it can. 

import csv
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)

# Initialize Chrome options for headless mode and other settings
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
chrome_options.add_argument("--window-size=1920,1080")  # Set window size to standard
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

try:
    # Initialize the Chrome WebDriver using webdriver-manager with specified options
    logging.info("Initializing headless Chrome WebDriver.")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
except WebDriverException as e:
    logging.error(f"Failed to initialize WebDriver: {e}")
    exit(1)

# Set parameters
region = "Dublin"  # Parameterized region name
start_page = 25  # Starting page number
url = f"https://mynest.ie/priceregister/{region}/{start_page}"
logging.info(f"Navigating to URL: {url}")
driver.get(url)

# Create the output directory
output_dir = f"scraped_{region}"
os.makedirs(output_dir, exist_ok=True)
logging.info(f"Output directory set to: {output_dir}")

# Explicitly wait for the table to load
wait = WebDriverWait(driver, 10)

# Set to store property data
property_data = []

# Set to store unique addresses and URLs
unique_addresses = set()

# Set the record limit
record_limit = 5
record_count = 0

# Variable to keep track of the current page number
current_page = start_page
max_pages_to_test = 500

# Function to scrape property details
def scrape_property_data():
    try:
        logging.debug("Scraping property details.")
        # Scrape individual fields
        asking_price = driver.find_element(By.XPATH, "//p[contains(text(), 'Asking Price')]/following-sibling::h4").text
        beds = driver.find_element(By.XPATH, "//p[contains(text(), 'Beds')]/following-sibling::h4").text
        baths = driver.find_element(By.XPATH, "//p[contains(text(), 'Baths')]/following-sibling::h4").text
        property_type = driver.find_element(By.XPATH, "//p[contains(text(), 'Property Type')]/following-sibling::h4").text
        energy_rating = driver.find_element(By.XPATH, "//p[contains(text(), 'Energy Rating')]/following-sibling::h4").text
        eircode = driver.find_element(By.XPATH, "//p[contains(text(), 'Eircode')]/following-sibling::h4").text
        lpt = driver.find_element(By.XPATH, "//p[contains(text(), 'Local Property Tax')]/following-sibling::h4").text
        agency_name = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Name')]/following-sibling::h4").text
        agency_contact = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Contact')]/following-sibling::h4").text
        address = driver.find_element(By.XPATH, "//h1[@class='card-title']").text

        logging.debug(f"Extracted data for address: {address}")

        # Extract price change history if available
        price_changes = []
        try:
            price_changes_table = driver.find_element(By.XPATH, "//table[contains(@class, 'table-hover table-striped')]")
            rows = price_changes_table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                change = cols[0].text.strip()
                price = cols[1].text.strip()
                date = cols[3].text.strip()
                price_changes.append({'Change': change, 'Price': price, 'Date': date})
            logging.debug(f"Price changes found: {price_changes}")
        except NoSuchElementException:
            logging.warning(f"Price changes not found for {address}.")
        except Exception as e:
            logging.error(f"Unexpected error when extracting price changes for {address}: {e}")

        # Return all the scraped details as a dictionary
        return {
            'Address': address,
            'Asking Price': asking_price,
            'Beds': beds,
            'Baths': baths,
            'Property Type': property_type,
            'Energy Rating': energy_rating,
            'Eircode': eircode,
            'Local Property Tax': lpt,
            'Agency Name': agency_name,
            'Agency Contact': agency_contact,
            'Price Changes': price_changes
        }

    except NoSuchElementException as e:
        logging.error(f"Element not found while scraping property data: {e}")
        return None
    except Exception as e:
        logging.error(f"Error occurred while scraping property data: {e}")
        return None

# Function to save the current data to a CSV file
def save_to_csv(filename):
    filepath = os.path.join(output_dir, filename)
    try:
        logging.info(f"Saving data to CSV file: {filepath}")
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Address', 'Asking Price', 'Beds', 'Baths', 'Property Type',
                'Energy Rating', 'Eircode', 'Local Property Tax', 'Agency Name',
                'Agency Contact', 'Price Changes', 'URL'
            ])  # Write the header
            for data in property_data:
                address = data['Address']
                url = next((url for addr, url in unique_addresses if addr == address), '')
                writer.writerow([
                    address, data['Asking Price'], data['Beds'], data['Baths'],
                    data['Property Type'], data['Energy Rating'], data['Eircode'],
                    data['Local Property Tax'], data['Agency Name'], data['Agency Contact'],
                    "; ".join([f"{p['Change']}, {p['Price']}, {p['Date']}" for p in data['Price Changes']]),
                    url
                ])
        logging.info(f"Data successfully saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save data to CSV: {e}")

# Function to handle pagination and scraping
def extract_data():
    global record_count, current_page

    # Keep track of processed addresses
    processed_addresses = set()

    while current_page <= max_pages_to_test and record_count < record_limit:
        try:
            logging.info(f"Processing page {current_page}")
            # Wait for the rows to load
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')))
            logging.debug("Page elements loaded.")

            # Re-fetch the list of rows after every navigation
            rows = driver.find_elements(By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')

            logging.info(f"Found {len(rows)} property listings on page {current_page}.")

            # Process each row one by one
            for index, row in enumerate(rows, start=1):
                if record_count >= record_limit:
                    logging.info(f"Record limit of {record_limit} reached.")
                    return

                address = row.text.strip()  # Extract the text and clean up spaces

                # Skip the row if it's already processed
                if address in processed_addresses:
                    logging.debug(f"Skipping already processed address: {address}")
                    continue

                try:
                    logging.info(f"Processing ({index}/{len(rows)}): {address}")

                    # Click the address to go to the detailed page
                    row.click()
                    logging.debug(f"Clicked on address: {address}")

                    # Wait for the detailed page to load
                    wait.until(EC.presence_of_element_located((By.XPATH, "//h1[@class='card-title']")))
                    logging.debug("Detailed page loaded.")

                    # Get the current URL
                    url = driver.current_url
                    logging.debug(f"Current URL: {url}")

                    # Store the URL and address
                    unique_addresses.add((address, url))

                    # Scrape property data
                    data = scrape_property_data()
                    if data:
                        property_data.append(data)
                        logging.info(f"Data scraped for {address}. Total records collected: {record_count + 1}")

                    # Add the address to the processed list
                    processed_addresses.add(address)

                    # Increment the record count
                    record_count += 1

                    # Go back to the main listing page
                    driver.back()
                    logging.debug("Navigated back to the main listing page.")

                    # Wait for the rows to reload
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')))
                    logging.debug("Main listing page reloaded.")

                    # Pause to ensure the page is fully loaded
                    time.sleep(1)

                except TimeoutException:
                    logging.error(f"Timeout while processing address: {address}")
                    driver.back()
                    continue
                except Exception as e:
                    logging.error(f"Error processing {address}: {e}")
                    driver.back()
                    continue

            # Save the current data to a CSV file after processing each page
            csv_filename = f"scraped_property_results_{region}_page_{current_page}.csv"
            save_to_csv(csv_filename)

            # Check if we need to go to the next page
            try:
                next_page_number = current_page + 1
                next_page_link = driver.find_element(By.XPATH, f"//a[contains(@class, 'pagination-item') and text()='{next_page_number}']")
                if next_page_link.is_displayed() and next_page_link.is_enabled():
                    next_page_link.click()
                    logging.info(f"Navigated to page {next_page_number}")
                    current_page = next_page_number
                    time.sleep(2)  # Wait for the next page to load
                else:
                    logging.info("No more pages to process or next button not clickable.")
                    break

            except NoSuchElementException:
                logging.info("Next page link not found. Reached the last available page.")
                break

            except Exception as e:
                logging.error(f"Error in pagination: {e}")
                break

        except TimeoutException:
            logging.error("Timeout while locating elements on the page.")
            continue  # In case of error, re-attempt to locate elements
        except Exception as e:
            logging.error(f"Unexpected error during data extraction: {e}")
            continue

# Scrape the data
logging.info("Starting data extraction process.")
extract_data()

# Close the driver after scraping is done
try:
    driver.quit()
    logging.info("WebDriver closed successfully.")
except Exception as e:
    logging.error(f"Error while closing WebDriver: {e}")

# Save the final output results to a CSV file
final_csv_filename = f"scraped_property_results_{region}_final.csv"
save_to_csv(final_csv_filename)

logging.info(f"\nFinal property details saved to {os.path.join(output_dir, final_csv_filename)}")
