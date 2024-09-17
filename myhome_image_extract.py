import os
import re
import requests
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

# Function to download an image from a URL
def download_image(image_url, save_folder, image_number):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        image_path = os.path.join(save_folder, f'image_{image_number}.jpg')
        with open(image_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded {image_path}")
    else:
        print(f"Failed to download image from {image_url}")

# Function to create a valid folder name from a URL
def slugify(url):
    return re.sub(r'\W+', '-', url.strip().lower())

# Function to extract image URLs from the MyHome property page
def scrape_images(url, save_folder):
    # Initialize the Chrome driver using webdriver-manager
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Open the URL
    driver.get(url)

    # Wait for the page to load or wait for the privacy consent button
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I ACCEPT')]"))
        ).click()
        print("Privacy popup accepted")
        time.sleep(2)

    except Exception as e:
        print(f"Error accepting privacy popup: {e}")
        driver.quit()
        return []

    image_urls = []

    try:
        # Click the main image to trigger the carousel
        main_image = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "img.gallery__main-image"))
        )
        main_image.click()
        time.sleep(2)

        while True:
            # Get the current image URL
            image_url = driver.find_element(By.CSS_SELECTOR, 'img.image-carousel__image').get_attribute('src')
            if image_url in image_urls:
                print(f"Duplicate image URL encountered: {image_url}. Stopping.")
                break
            else:
                image_urls.append(image_url)
                download_image(image_url, save_folder, len(image_urls))

            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "span.image-carousel__button--right"))
                )
                next_button.click()
                time.sleep(2)
            except Exception as e:
                print(f"Reached the end of the carousel or couldn't find the next button: {e}")
                break

    except Exception as e:
        print(f"Error extracting images: {e}")
    
    driver.quit()
    return image_urls

# Function to scrape the property details and download the images
def scrape_property(url):
    # Create base folder "my home images"
    base_folder = "my home images"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Create a subfolder based on the URL slug
    property_folder = os.path.join(base_folder, slugify(url))
    if not os.path.exists(property_folder):
        os.makedirs(property_folder)

    # Scrape images and save them in the property folder
    images = scrape_images(url, property_folder)
    print(f"All image URLs: {images}")

    return images

# Example usage
url = "https://www.myhome.ie/residential/brochure/16-brooklands-castlebar-co-mayo/4832101"
scrape_property(url)
