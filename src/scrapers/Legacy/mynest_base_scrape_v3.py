import os
import csv
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------- Configuration -----------------------------

# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Console Handler for INFO and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File Handler for DEBUG and above
file_handler = logging.FileHandler('scraper.log')
file_handler.setLevel(logging.DEBUG)

# Log Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add Handlers to Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configuration Parameters (Can be set via environment variables)
REGION = os.getenv("REGION", "Dublin")
START_PAGE = int(os.getenv("START_PAGE", "25"))  # Starting page number
MAX_PAGES_TO_TEST = int(os.getenv("MAX_PAGES_TO_TEST", "500"))
RECORD_LIMIT = int(os.getenv("RECORD_LIMIT", "5"))  # Set to 5 as per original script
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "True").lower() in ["true", "1", "yes"]  # Headless by default
SNAPSHOT_INTERVAL = int(os.getenv("SNAPSHOT_INTERVAL", "5"))  # Number of records after which to create a snapshot
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"scraped_{REGION}")

# Snapshot Directory
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ----------------------------- Helper Functions -----------------------------

def init_driver(headless=True):
    """Initialize the Selenium WebDriver with optional headless mode."""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")  # Suppress driver logs
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")  # Attempt to avoid detection

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        driver.set_page_load_timeout(30)
        logger.debug("WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logger.critical(f"Error initializing WebDriver: {e}")
        return None

def save_to_csv(filename, property_data):
    """Save scraped property data to a CSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fieldnames = [
        'Address', 'Asking Price', 'Beds', 'Baths', 'Property Type',
        'Energy Rating', 'Eircode', 'Local Property Tax', 'Agency Name',
        'Agency Contact', 'Price Changes', 'URL'
    ]
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for data in property_data:
                writer.writerow(data)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to CSV: {e}")

def create_snapshot(snapshot_number, property_data):
    """Create a snapshot of the current scraping progress."""
    snapshot_file = os.path.join(SNAPSHOT_DIR, f"snapshot_{snapshot_number}.csv")
    fieldnames = [
        'Address', 'Asking Price', 'Beds', 'Baths', 'Property Type',
        'Energy Rating', 'Eircode', 'Local Property Tax', 'Agency Name',
        'Agency Contact', 'Price Changes', 'URL'
    ]
    try:
        with open(snapshot_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for data in property_data:
                writer.writerow(data)
        logger.info(f"Snapshot {snapshot_number} saved to {snapshot_file}")
    except Exception as e:
        logger.error(f"Failed to create snapshot {snapshot_number}: {e}")

def scrape_property_data(driver):
    """Scrape individual property details from the detailed page."""
    try:
        # Wait until the main elements are present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )

        # Scrape individual fields
        address = driver.find_element(By.TAG_NAME, "h1").text.strip()
        asking_price = driver.find_element(By.XPATH, "//p[contains(text(), 'Asking Price')]/following-sibling::h4").text.strip()
        beds = driver.find_element(By.XPATH, "//p[contains(text(), 'Beds')]/following-sibling::h4").text.strip()
        baths = driver.find_element(By.XPATH, "//p[contains(text(), 'Baths')]/following-sibling::h4").text.strip()
        property_type = driver.find_element(By.XPATH, "//p[contains(text(), 'Property Type')]/following-sibling::h4").text.strip()
        energy_rating = driver.find_element(By.XPATH, "//p[contains(text(), 'Energy Rating')]/following-sibling::h4").text.strip()
        eircode = driver.find_element(By.XPATH, "//p[contains(text(), 'Eircode')]/following-sibling::h4").text.strip()
        lpt = driver.find_element(By.XPATH, "//p[contains(text(), 'Local Property Tax')]/following-sibling::h4").text.strip()
        agency_name = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Name')]/following-sibling::h4").text.strip()
        agency_contact = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Contact')]/following-sibling::h4").text.strip()

        # Extract price change history if available
        price_changes = []
        try:
            price_changes_table = driver.find_element(By.XPATH, "//table[contains(@class, 'table-hover table-striped')]")
            rows = price_changes_table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 4:
                    change = cols[0].text.strip()
                    price = cols[1].text.strip()
                    date = cols[3].text.strip()
                    price_changes.append({'Change': change, 'Price': price, 'Date': date})
        except NoSuchElementException:
            logger.warning("Price Changes table not found.")
        except Exception as e:
            logger.warning(f"Error extracting price changes: {e}")

        # Return all the scraped details as a dictionary
        property_info = {
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
            'Price Changes': "; ".join([f"{p['Change']}, {p['Price']}, {p['Date']}" for p in price_changes]),
            'URL': driver.current_url
        }

        logger.debug(f"Scraped data for {address}: {property_info}")
        return property_info

    except NoSuchElementException as e:
        logger.error(f"Element not found while scraping property data: {e}")
        return None
    except TimeoutException as e:
        logger.error(f"Timeout while scraping property data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while scraping property data: {e}")
        return None

# ----------------------------- Main Scraping Function -----------------------------

def extract_data():
    """Main function to handle pagination and scrape property data."""
    driver = init_driver(headless=HEADLESS_MODE)
    if not driver:
        logger.critical("WebDriver could not be initialized. Exiting scraper.")
        return

    start_url = f"https://mynest.ie/priceregister/{REGION}/{START_PAGE}"
    try:
        driver.get(start_url)
        logger.info(f"Navigating to {start_url}")
    except Exception as e:
        logger.critical(f"Failed to load the initial URL: {e}")
        driver.quit()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    property_data = []
    unique_addresses = set()
    record_count = 0
    current_page = START_PAGE

    try:
        while current_page <= MAX_PAGES_TO_TEST and record_count < RECORD_LIMIT:
            logger.info(f"Processing page {current_page}")
            try:
                # Optimize wait by targeting specific elements
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content'))
                )
                rows = driver.find_elements(By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')
                logger.debug(f"Found {len(rows)} property listings on page {current_page}")

                if not rows:
                    logger.info(f"No property listings found on page {current_page}. Ending scraping.")
                    break

                for row in rows:
                    if record_count >= RECORD_LIMIT:
                        break

                    address = row.text.strip()  # Extract the text and clean up spaces

                    # Skip the row if it's already processed
                    if address in unique_addresses:
                        logger.debug(f"Duplicate address found: {address}. Skipping.")
                        continue

                    try:
                        logger.info(f"Processing: {address}")

                        # Open property in the same tab to maintain context
                        row.click()

                        # Wait for the detailed page to load
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
                        )

                        # Get the current URL
                        url = driver.current_url

                        # Scrape property data
                        data = scrape_property_data(driver)
                        if data:
                            property_data.append(data)
                            unique_addresses.add(address)
                            record_count += 1
                            logger.info(f"Scraped {record_count}/{RECORD_LIMIT} records.")

                            # Snapshotting
                            if record_count % SNAPSHOT_INTERVAL == 0:
                                snapshot_number = record_count // SNAPSHOT_INTERVAL
                                create_snapshot(snapshot_number, property_data)

                        # Save progress after each record
                        csv_filename = f"scraped_property_results_{REGION}_page_{current_page}.csv"
                        save_to_csv(csv_filename, property_data)

                        # Navigate back to the main listing page
                        driver.back()

                        # Wait for the listings to load again
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content'))
                        )

                    except StaleElementReferenceException as sere:
                        logger.warning(f"StaleElementReferenceException encountered: {sere}. Retrying.")
                        time.sleep(2)
                        continue  # Retry the same row
                    except NoSuchElementException as nse:
                        logger.error(f"Element not found while processing property: {nse}")
                        driver.back()
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content'))
                        )
                        continue
                    except Exception as e:
                        logger.error(f"Error processing property: {e}")
                        driver.back()
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content'))
                        )
                        continue

                # Save the current data to a CSV file after processing each page
                csv_filename = f"scraped_property_results_{REGION}_page_{current_page}.csv"
                save_to_csv(csv_filename, property_data)

                # Proceed to the next page
                try:
                    next_page_number = current_page + 1
                    next_page_link = driver.find_element(By.XPATH, f"//a[contains(@class, 'pagination-item') and text()='{next_page_number}']")
                    if next_page_link.is_displayed() and next_page_link.is_enabled():
                        next_page_link.click()
                        logger.info(f"Navigated to page {next_page_number}")
                        current_page = next_page_number
                        time.sleep(2)  # Wait for the next page to load
                    else:
                        logger.info("Next page link is not clickable. Ending scraping.")
                        break

                except NoSuchElementException:
                    logger.info("Next page link not found. Ending scraping.")
                    break
                except Exception as e:
                    logger.error(f"Error navigating to the next page: {e}")
                    break

            except TimeoutException:
                logger.error(f"Timeout while waiting for property listings on page {current_page}.")
                break
            except Exception as e:
                logger.error(f"Unexpected error on page {current_page}: {e}")
                break

    finally:
        driver.quit()
        logger.info("WebDriver closed.")

    # Save the final results
    final_csv_filename = f"scraped_property_results_{REGION}_final.csv"
    save_to_csv(final_csv_filename, property_data)
    logger.info(f"Final property details saved to {os.path.join(OUTPUT_DIR, final_csv_filename)}")

# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    logger.info("Starting the scraping process...")
    extract_data()
    logger.info("Scraping process completed.")
