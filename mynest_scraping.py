from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys

def scrape_property_data(url):
    # Set up the Chrome WebDriver
    driver = webdriver.Chrome()
    driver.get(url)

    # Wait for the page to load
    driver.implicitly_wait(5)

    # Extracting Fields as before
    try:
        asking_price = driver.find_element(By.XPATH, "//p[contains(text(), 'Asking Price')]/following-sibling::h4").text
        print(f"Asking Price: {asking_price}")

        beds = driver.find_element(By.XPATH, "//p[contains(text(), 'Beds')]/following-sibling::h4").text
        print(f"Beds: {beds}")

        baths = driver.find_element(By.XPATH, "//p[contains(text(), 'Baths')]/following-sibling::h4").text
        print(f"Baths: {baths}")

        property_type = driver.find_element(By.XPATH, "//p[contains(text(), 'Property Type')]/following-sibling::h4").text
        print(f"Property Type: {property_type}")

        energy_rating = driver.find_element(By.XPATH, "//p[contains(text(), 'Energy Rating')]/following-sibling::h4").text
        print(f"Energy Rating (BER): {energy_rating}")

        eircode = driver.find_element(By.XPATH, "//p[contains(text(), 'Eircode')]/following-sibling::h4").text
        print(f"Eircode: {eircode}")

        lpt = driver.find_element(By.XPATH, "//p[contains(text(), 'Local Property Tax')]/following-sibling::h4").text
        print(f"Local Property Tax (LPT): {lpt}")

        agency_name = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Name')]/following-sibling::h4").text
        agency_contact = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Contact')]/following-sibling::h4").text
        print(f"Agency Name: {agency_name}")
        print(f"Agency Contact: {agency_contact}")

        # Extracting the Address from the <h1> element with class 'card-title'
        address = driver.find_element(By.XPATH, "//h1[@class='card-title']").text
        print(f"Address: {address}")

    except Exception as e:
        print(f"Error occurred: {e}")

    # Extracting the Price Changes Table
    try:
        # Locate the price changes table
        price_changes_table = driver.find_element(By.XPATH, "//table[contains(@class, 'table-hover table-striped')]")
        rows = price_changes_table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header

        # Prepare a list to store price change data
        price_changes_data = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            change = cols[0].text.strip()
            price = cols[1].text.strip()
            date = cols[3].text.strip()

            # Append as a dictionary to the list
            price_changes_data.append({
                'Change': change,
                'Price': price,
                'Date': date
            })

        # Output the structured price changes data
        for entry in price_changes_data:
            print(f"Change: {entry['Change']}, Price: {entry['Price']}, Date: {entry['Date']}")

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