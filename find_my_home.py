from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Function to perform a Google search and extract the results
def google_search(query):
    # Set up Selenium Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Navigate to Google
    driver.get(f"https://www.google.com/search?q={query}")
    
    # Wait for the page to load
    time.sleep(3)
    
    # Find the search result links
    search_results = driver.find_elements(By.CSS_SELECTOR, 'div.g a')

    urls = []
    for result in search_results:  # Get all the results
        url = result.get_attribute('href')
        urls.append(url)
    
    # Close the browser
    driver.quit()

    return urls

# Function to check if any of the URLs are from myhome.ie
def find_myhome_links(urls):
    myhome_links = [url for url in urls if 'myhome.ie' in url]
    return myhome_links

# Main execution
query = "Apartment 9, Chestnut House, Grace Park Court, Beaumont, Dublin 9, D09KT67"
query = "13 Mapas Avenue, Dalkey, Co. Dublin, A96XW20"
urls = google_search(query)

# Check and print only MyHome.ie links
myhome_links = find_myhome_links(urls)

if myhome_links:
    print("Found MyHome.ie links:")
    for link in myhome_links:
        print(link)
else:
    print("No MyHome.ie links found in the search results.")
