import os
import re
import csv
import time
import shutil
import logging
import requests
import pandas as pd
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
)
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------- Configuration -----------------------------

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Create console handler for INFO and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler for DEBUG and above
file_handler = logging.FileHandler('scraper.log')
file_handler.setLevel(logging.DEBUG)

# Define log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Constants
IMAGE_DOWNLOAD_WORKERS = 5
SNAPSHOT_INTERVAL = 10  # Number of records after which to create a snapshot

# Columns to retain (Removed specified columns)
MYHOME_FIELDS = [
    "MyHome_Address",
    "MyHome_Asking_Price",
    "MyHome_Beds",
    "MyHome_Baths",
    "MyHome_Floor_Area_Value",
    "MyHome_BER_Rating",
    "MyHome_Latitude",
    "MyHome_Longitude",
    "MyHome_Link",
]

# ----------------------------- Helper Functions -----------------------------

def slugify(text):
    """Create a valid folder name from a URL or address."""
    return re.sub(r"[^\w\-]", "-", text.strip().lower())

def safe_get_text(driver, selector, by=By.CSS_SELECTOR, timeout=10):
    """Safely extract text from a web element."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
        text = element.text.strip()
        logger.debug(f"Extracted text for selector '{selector}': '{text}'")
        return text
    except (TimeoutException, NoSuchElementException):
        logger.warning(f"Element not found: {selector}")
        return ""

def download_image(image_url, save_folder, image_number):
    """Download an image from a URL."""
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image_path = os.path.join(save_folder, f"image_{image_number:03d}.jpg")
        with open(image_path, "wb") as file:
            for chunk in response.iter_content(8192):
                file.write(chunk)
        logger.debug(f"Downloaded {image_path}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download image from {image_url}: {e}")
        return False

def init_driver(headless=True):
    """Initialize the Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")  # Suppress driver logs

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        driver.set_page_load_timeout(30)
        logger.debug("WebDriver initialized successfully.")
        return driver
    except WebDriverException as e:
        logger.error(f"Error initializing WebDriver: {e}")
        return None

def google_search(query, driver, max_retries=3):
    """Perform a Google search and return a list of result URLs."""
    for attempt in range(max_retries):
        try:
            driver.get(f"https://www.google.com/search?q={query}")

            # Wait until search results are present
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.g"))
            )

            search_results = driver.find_elements(By.CSS_SELECTOR, "div.g a")
            urls = [result.get_attribute("href") for result in search_results]
            logger.info(f"Found {len(urls)} URLs from Google search for query '{query}'.")
            return urls
        except Exception as e:
            logger.error(
                f"Error during Google search on attempt {attempt + 1}: {e}"
            )
            if attempt == max_retries - 1:
                logger.error("Max retries reached. Returning empty list.")
                return []
            time.sleep(2 ** attempt)  # Exponential backoff
    return []

def find_myhome_links(urls):
    """Filter URLs to find those that contain 'myhome.ie'."""
    myhome_links = [url for url in urls if url and "myhome.ie" in url]
    logger.info(f"Filtered {len(myhome_links)} MyHome.ie links.")
    return myhome_links

def parse_listing(url, driver):
    """Parse listing data from a MyHome.ie property page."""
    data = {"MyHome_Link": url}

    try:
        driver.get(url)
        logger.info(f"Opened URL: {url}")

        # Handle the privacy popup
        try:
            accept_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'I ACCEPT')]")
                )
            )
            accept_button.click()
            logger.info("Privacy popup accepted.")
            time.sleep(2)  # Wait for the page to update after accepting
        except TimeoutException:
            logger.info("Privacy popup not found or already handled.")
        except Exception as e:
            logger.error(f"Error handling privacy popup: {e}")

        # Extract data
        data["MyHome_Address"] = safe_get_text(driver, "h1.h4.fw-bold")
        data["MyHome_Asking_Price"] = safe_get_text(driver, "b.brochure__price")
        data["MyHome_Beds"] = safe_get_text(
            driver,
            "//span[contains(@class, 'info-strip--divider') and contains(text(), 'beds')]",
            By.XPATH,
        )
        data["MyHome_Baths"] = safe_get_text(
            driver,
            "//span[contains(@class, 'info-strip--divider') and contains(text(), 'baths')]",
            By.XPATH,
        )

        # Floor area parsing
        try:
            floor_area_element = driver.find_element(
                By.XPATH,
                "//span[contains(@class, 'info-strip--divider') and contains(text(), 'm')]",
            )
            floor_area_text = floor_area_element.text.strip()
            match = re.search(r"([\d.,]+)\s*m", floor_area_text)
            if match:
                floor_area_value = match.group(1)
                data["MyHome_Floor_Area_Value"] = floor_area_value
                logger.debug(f"Extracted floor area: {floor_area_value} m²")
            else:
                data["MyHome_Floor_Area_Value"] = ""
                logger.warning("Floor area value not found.")
        except NoSuchElementException:
            data["MyHome_Floor_Area_Value"] = ""
            logger.warning("Floor area element not found.")

        # Extract BER information
        try:
            ber_elements = driver.find_elements(
                By.XPATH,
                "//p[contains(@class, 'brochure__details--description-content')]",
            )
            data["MyHome_BER_Rating"] = ""
            for ber_element in ber_elements:
                ber_text = ber_element.text.strip()
                if "BER" in ber_text:
                    ber_parts = ber_text.split("\n")
                    for part in ber_parts:
                        if "BER:" in part:
                            data["MyHome_BER_Rating"] = part.replace("BER:", "").strip()
                            logger.debug(f"Extracted BER Rating: {data['MyHome_BER_Rating']}")
                    break  # Assuming only one BER section exists
        except Exception as e:
            logger.error(f"Error extracting BER info: {e}")

        # Set default values for optional fields (Removed specified fields)
        # Since we have removed these columns, we won't set them

        logger.info(f"Extracted data for {url}: {data}")

    except Exception as e:
        logger.error(f"Error extracting data from {url}: {e}")

    return data

def get_latitude_longitude(address, api_key):
    """Get latitude and longitude for a given address using Google Geocoding API."""
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        if results["status"] == "OK":
            location = results["results"][0]["geometry"]["location"]
            logger.debug(f"Geocoded '{address}' to (lat: {location['lat']}, lng: {location['lng']})")
            return location["lat"], location["lng"]
        else:
            logger.error(
                f"Error fetching geocode data for '{address}': {results['status']}"
            )
            return None, None
    except Exception as e:
        logger.error(f"Exception during geocoding for '{address}': {e}")
        return None, None

def scrape_images(url, save_folder):
    """Scrape image URLs from a MyHome.ie property page."""
    image_urls = []

    # Initialize a separate driver for image scraping to avoid conflicts
    image_driver = init_driver(headless=True)
    if image_driver is None:
        logger.error("Image WebDriver could not be initialized. Skipping image scraping.")
        return image_urls

    try:
        image_driver.get(url)
        logger.info(f"Opened URL for images: {url}")

        # Handle the privacy popup
        try:
            consent_button = WebDriverWait(image_driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'I ACCEPT')]")
                )
            )
            consent_button.click()
            logger.info("Privacy popup accepted for images.")
            time.sleep(2)  # Wait for the page to update after accepting
        except TimeoutException:
            logger.warning("No privacy popup found for images or timeout waiting for it.")
        except Exception as e:
            logger.warning(f"Error accepting privacy popup for images: {e}")

        # Click the main image to trigger the carousel
        try:
            main_image = WebDriverWait(image_driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "img.gallery__main-image"))
            )
            main_image.click()
            logger.info("Clicked main image to open carousel for images.")
            time.sleep(2)  # Wait for the carousel to open
        except TimeoutException:
            logger.error("Main image not found or not clickable for images.")
            return image_urls
        except Exception as e:
            logger.error(f"Error clicking main image for images: {e}")
            return image_urls

        while True:
            try:
                image_element = WebDriverWait(image_driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "img.image-carousel__image")
                    )
                )
                image_url = image_element.get_attribute("src")

                if image_url in image_urls:
                    logger.info(f"Duplicate image URL encountered: {image_url}. Stopping.")
                    break
                else:
                    image_urls.append(image_url)
                    logger.info(f"Found image URL: {image_url}")

                # Click the next button
                next_button = WebDriverWait(image_driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "span.image-carousel__button--right")
                    )
                )
                next_button.click()
                time.sleep(0.5)  # Reduced sleep time for quicker scraping
            except TimeoutException:
                logger.info(
                    "Reached the end of the carousel or couldn't find the next button for images."
                )
                break
            except Exception as e:
                logger.error(f"Error navigating carousel for images: {e}")
                break

    except Exception as e:
        logger.error(f"Error extracting images from {url}: {e}")

    finally:
        image_driver.quit()
        logger.info("Image WebDriver closed after scraping images.")

    return image_urls

def create_snapshot(output_file, snapshot_number):
    """Create a snapshot of the current output CSV."""
    snapshot_dir = os.path.join(os.path.dirname(output_file), "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_file = os.path.join(
        snapshot_dir, f"snapshot_{snapshot_number}.csv"
    )
    try:
        shutil.copy2(output_file, snapshot_file)
        logger.info(f"Created snapshot: {snapshot_file}")
    except Exception as e:
        logger.error(f"Failed to create snapshot: {e}")

# ----------------------------- Main Processing Function -----------------------------

def process_csv(input_file, output_file, api_key, limit=500, headless=True):
    """Process the input CSV to enrich data with MyHome.ie listing details and images."""
    # Read the CSV into a DataFrame
    try:
        df = pd.read_csv(input_file)
        original_count = len(df)
        df = df.head(limit)
        logger.info(f"Loaded {len(df)} records from input CSV (limit set to {limit}).")
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        return

    # Define new fields to be added (already defined MYHOME_FIELDS without removed columns)

    # Initialize the output CSV with headers
    fieldnames = list(df.columns) + MYHOME_FIELDS
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f"Initialized output CSV with headers.")
    except Exception as e:
        logger.error(f"Failed to initialize output CSV: {e}")
        return

    # Prepare snapshot directory
    snapshot_dir = os.path.join(
        "data", "processed", "my_home_images_snapshots"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    # Initialize WebDriver for Google searches and MyHome parsing
    driver = init_driver(headless=headless)
    if driver is None:
        logger.error("WebDriver could not be initialized. Exiting process.")
        return

    results = []

    try:
        for i, row in df.iterrows():
            if i >= limit:
                break

            address = row.get("Address", "").strip()
            logger.info(f"Processing record {i+1}: {address}")

            if not address:
                logger.warning(f"No address found for record {i+1}. Skipping.")
                results.append({"Address": address, "Images_Found": False})
                # Add empty values for MyHome fields
                for field in MYHOME_FIELDS:
                    row[field] = ""
                continue

            try:
                # Perform Google search to find MyHome.ie links
                urls = google_search(address, driver)
                myhome_links = find_myhome_links(urls)

                if myhome_links:
                    myhome_url = myhome_links[0]
                    logger.info(f"Found MyHome.ie link: {myhome_url}")

                    # Parse listing data
                    listing_data = parse_listing(myhome_url, driver)

                    # Update row with listing data
                    for key, value in listing_data.items():
                        if key in MYHOME_FIELDS:
                            row[key] = value

                    # Get latitude and longitude
                    lat, lng = get_latitude_longitude(address, api_key)
                    row["MyHome_Latitude"] = lat
                    row["MyHome_Longitude"] = lng

                    # Scrape and download images using a separate driver
                    property_folder = os.path.join(
                        "data", "processed", "my_home_images", slugify(address)
                    )
                    os.makedirs(property_folder, exist_ok=True)
                    image_urls = scrape_images(myhome_url, property_folder)
                    logger.info(f"Found {len(image_urls)} image URLs for {address}")

                    # Download images concurrently
                    with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as executor:
                        futures = [
                            executor.submit(
                                download_image, img_url, property_folder, idx + 1
                            )
                            for idx, img_url in enumerate(image_urls)
                        ]
                        images_downloaded = 0
                        for future in as_completed(futures):
                            if future.result():
                                images_downloaded += 1
                        logger.info(
                            f"Downloaded {images_downloaded} out of {len(image_urls)} images for {address}"
                        )

                    results.append({"Address": address, "Images_Found": images_downloaded > 0})
                else:
                    logger.warning("No MyHome.ie link found")
                    # Add empty values for MyHome fields
                    for field in MYHOME_FIELDS:
                        row[field] = ""
                    results.append({"Address": address, "Images_Found": False})

            except Exception as e:
                logger.error(f"Error processing record {i+1}: {e}")
                # Add empty values for MyHome fields
                for field in MYHOME_FIELDS:
                    row[field] = ""
                results.append({"Address": address, "Images_Found": False})

            # Write the updated row to the output CSV
            try:
                with open(output_file, "a", newline="", encoding="utf-8") as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writerow(row.to_dict())
                logger.info(f"Written data for: {address}")
            except Exception as e:
                logger.error(f"Failed to write data for {address}: {e}")

            # Add to results for snapshot
            # Snapshot handling
            if (i + 1) % SNAPSHOT_INTERVAL == 0:
                snapshot_number = (i + 1) // SNAPSHOT_INTERVAL
                snapshot_df = pd.DataFrame(results)
                snapshot_path = os.path.join(
                    snapshot_dir, f"snapshot_{snapshot_number}.csv"
                )
                try:
                    snapshot_df.to_csv(snapshot_path, index=False)
                    logger.info(f"Saved snapshot to {snapshot_path}")
                except Exception as e:
                    logger.error(f"Failed to save snapshot: {e}")

    finally:
        driver.quit()
        logger.info("WebDriver closed.")

    # Save final snapshot if not already saved
    if len(results) % SNAPSHOT_INTERVAL != 0 and len(results) > 0:
        snapshot_number = (len(results) // SNAPSHOT_INTERVAL) + 1
        snapshot_df = pd.DataFrame(results)
        snapshot_path = os.path.join(
            snapshot_dir, f"snapshot_{snapshot_number}.csv"
        )
        try:
            snapshot_df.to_csv(snapshot_path, index=False)
            logger.info(f"Saved final snapshot to {snapshot_path}")
        except Exception as e:
            logger.error(f"Failed to save final snapshot: {e}")

    # Save final results
    final_df = pd.DataFrame(results)
    final_path = os.path.join(
        "data", "processed", "my_home_images_final_results.csv"
    )
    try:
        final_df.to_csv(final_path, index=False)
        logger.info(f"Saved final results to {final_path}")
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")

# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    # Configuration via environment variables or defaults
    input_file = os.getenv("INPUT_CSV_PATH", "data/processed/scraped_dublin/scraped_property_results_Dublin_page_24.csv")
    output_file = os.getenv("OUTPUT_CSV_PATH", "data/processed/scraped_dublin_metadata/scraped_property_results_metadata_Dublin_page_1_test.csv")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    limit = int(os.getenv("PROCESS_LIMIT", "2"))  # Set to 500 or desired number in production
    headless = os.getenv("HEADLESS_MODE", "True").lower() in ["true", "1", "yes"]

    if not api_key:
        logger.error(
            "Google Maps API key not found. Please set the GOOGLE_MAPS_API_KEY environment variable."
        )
    else:
        logger.info("Starting CSV processing...")
        process_csv(input_file, output_file, api_key, limit=limit, headless=headless)
        logger.info(f"Processing complete. Results saved to {output_file}")
