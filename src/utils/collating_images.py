from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys

def scrape_property_data(url):
    # Set up the Chrome WebDriver
    driver = webdriver.Chrome()  # Ensure you have the correct path to your chromedriver if needed
    driver.get(url)

    # Wait for the page to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # Extract Local Property Tax (LPT)
    try:
        lpt_element = driver.find_element(By.XPATH, "//p[contains(text(), 'Local Property Tax')]/following-sibling::h4")
        lpt_value = lpt_element.text.strip()
        print(f"Local Property Tax (LPT): {lpt_value}")
    except Exception as e:
        print(f"LPT not found: {e}")

    # Extract Price Changes using explicit wait
    try:
        price_changes_table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'table-hover table-striped')]"))
        )
        rows = price_changes_table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
        print("\nPrice Changes:")
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            change = cols[0].text.strip()
            price = cols[1].text.strip()
            date = cols[3].text.strip()
            print(f"Change: {change}, Price: {price}, Date: {date}")
    except Exception as e:
        print(f"Price Changes not found: {e}")

    # Close the WebDriver
    driver.quit()

# Main block to accept URL from command-line arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mynest_scraping.py <property_url>")
        sys.exit(1)

    url = sys.argv[1]
    scrape_property_data(url)
