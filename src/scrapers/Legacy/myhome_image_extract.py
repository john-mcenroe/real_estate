import os
import re
import requests
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to safely extract text from web elements
def safe_get_text(element):
    return element.text.strip() if element else ""

# Function to download an image from a URL
def download_image(image_url, save_folder, image_number):
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image_path = os.path.join(save_folder, f'image_{image_number:03d}.jpg')
        with open(image_path, 'wb') as file:
            for chunk in response.iter_content(8192):
                file.write(chunk)
        logger.info(f"Downloaded {image_path}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download image from {image_url}: {e}")
        return False

# Function to create a valid folder name from a URL or address
def slugify(text):
    return re.sub(r'[^\w\-]', '-', text.strip().lower())

# Function to extract image URLs from the MyHome property page
def scrape_images(url, save_folder):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)

    image_urls = []

    try:
        driver.get(url)
        logger.info(f"Opened URL: {url}")

        # Wait for the page to load or wait for the privacy consent button
        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I ACCEPT')]"))
            ).click()
            logger.info("Privacy popup accepted")
            time.sleep(2)
        except Exception as e:
            logger.warning(f"No privacy popup found or error accepting it: {e}")

        # Click the main image to trigger the carousel
        main_image = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img.gallery__main-image"))
        )
        main_image.click()
        logger.info("Clicked main image to open carousel")
        time.sleep(2)

        while True:
            image_url = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'img.image-carousel__image'))
            ).get_attribute('src')
            
            if image_url in image_urls:
                logger.info(f"Duplicate image URL encountered: {image_url}. Stopping.")
                break
            else:
                image_urls.append(image_url)
                logger.info(f"Found image URL: {image_url}")

            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "span.image-carousel__button--right"))
                )
                next_button.click()
                time.sleep(0.5)  # Reduced sleep time for quicker scraping
            except Exception as e:
                logger.info(f"Reached the end of the carousel or couldn't find the next button: {e}")
                break

    except Exception as e:
        logger.error(f"Error extracting images: {e}")
    
    finally:
        driver.quit()
        logger.info("Browser closed")

    return image_urls

# Function to scrape the property details and download the images
def scrape_property(url):
    parsed_url = urlparse(url)
    address = parsed_url.path.split('/')[-2]
    
    # Create base folder "data/my_home_images"
    base_folder = os.path.join("data", "my_home_images")
    os.makedirs(base_folder, exist_ok=True)

    # Create a subfolder based on the address
    property_folder = os.path.join(base_folder, slugify(address))
    os.makedirs(property_folder, exist_ok=True)
    logger.info(f"Created folder: {property_folder}")

    # Scrape images
    image_urls = scrape_images(url, property_folder)
    logger.info(f"Found {len(image_urls)} image URLs")

    # Download images concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_image, url, property_folder, i+1) for i, url in enumerate(image_urls)]
        for future in as_completed(futures):
            future.result()

    logger.info(f"Finished processing {url}")
    return image_urls

# Example usage
if __name__ == "__main__":
    url = "https://www.myhome.ie/residential/brochure/12-greenville-court-stradbrook-road-blackrock-co-dublin/4749817"
    scrape_property(url)