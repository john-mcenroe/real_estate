from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time

def test_pagination():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    region = "Dublin"
    url = f"https://mynest.ie/priceregister/{region}"
    driver.get(url)
    wait = WebDriverWait(driver, 10)

    current_page = 1
    max_pages_to_test = 5

    def print_page_source():
        print("Page source:")
        print(driver.page_source[:1000])  # Print first 1000 characters of page source

    while current_page <= max_pages_to_test:
        print(f"\nCurrently on page {current_page}")
        
        # Wait for the pagination container to be present
        try:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "pagination-container")))
        except TimeoutException:
            print("Pagination container not found. Printing page source.")
            print_page_source()
            break

        # Check if we're on the correct page
        try:
            # Try different methods to find the active page
            active_page = driver.find_element(By.XPATH, f"//a[contains(@class, 'pagination-item') and contains(@class, 'active') and text()='{current_page}']")
            print(f"Confirmed: Active page is {active_page.text}")
        except NoSuchElementException:
            print(f"Warning: Could not confirm active page {current_page}")
            print("Printing all pagination items:")
            pagination_items = driver.find_elements(By.CLASS_NAME, "pagination-item")
            for item in pagination_items:
                print(f"Pagination item: {item.text} - Classes: {item.get_attribute('class')}")
        
        # Try to go to the next page
        try:
            next_page_number = current_page + 1
            next_page_link = driver.find_element(By.XPATH, f"//a[contains(@class, 'pagination-item') and text()='{next_page_number}']")
            if next_page_link.is_displayed() and next_page_link.is_enabled():
                print(f"Found next page link: {next_page_link.text}")
                next_page_link.click()
                print(f"Clicked on page {next_page_number}")
                current_page = next_page_number
                time.sleep(2)
            else:
                print("Next page link not clickable")
                break
        except NoSuchElementException:
            print("Next page link not found. Printing all pagination items:")
            pagination_items = driver.find_elements(By.CLASS_NAME, "pagination-item")
            for item in pagination_items:
                print(f"Pagination item: {item.text} - Classes: {item.get_attribute('class')}")
            break
        except Exception as e:
            print(f"Error in pagination: {e}")
            print_page_source()
            break

    driver.quit()
    print("Pagination test completed")

if __name__ == "__main__":
    test_pagination()