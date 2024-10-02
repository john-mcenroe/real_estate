from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_property_data(url):
    # Set up the Chrome WebDriver
    driver = webdriver.Chrome()
    driver.get(url)

    # Wait for the page to load
    driver.implicitly_wait(5)

    # Dictionary to store extracted data
    property_data = {}

    # Extracting fields
    try:
        asking_price = driver.find_element(By.XPATH, "//p[contains(text(), 'Asking Price')]/following-sibling::h4").text
        property_data['Asking Price'] = asking_price

        beds = driver.find_element(By.XPATH, "//p[contains(text(), 'Beds')]/following-sibling::h4").text
        property_data['Beds'] = beds

        baths = driver.find_element(By.XPATH, "//p[contains(text(), 'Baths')]/following-sibling::h4").text
        property_data['Baths'] = baths

        property_type = driver.find_element(By.XPATH, "//p[contains(text(), 'Property Type')]/following-sibling::h4").text
        property_data['Property Type'] = property_type

        energy_rating = driver.find_element(By.XPATH, "//p[contains(text(), 'Energy Rating')]/following-sibling::h4").text
        property_data['Energy Rating'] = energy_rating

        eircode = driver.find_element(By.XPATH, "//p[contains(text(), 'Eircode')]/following-sibling::h4").text
        property_data['Eircode'] = eircode

        lpt = driver.find_element(By.XPATH, "//p[contains(text(), 'Local Property Tax')]/following-sibling::h4").text
        property_data['Local Property Tax'] = lpt

        agency_name = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Name')]/following-sibling::h4").text
        property_data['Agency Name'] = agency_name

        agency_contact = driver.find_element(By.XPATH, "//label[contains(text(), 'Agency Contact')]/following-sibling::h4").text
        property_data['Agency Contact'] = agency_contact

        # Extracting the Address from the <h1> element with class 'card-title'
        address = driver.find_element(By.XPATH, "//h1[@class='card-title']").text
        property_data['Address'] = address

    except Exception as e:
        print(f"Error occurred while extracting main details: {e}")

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

        # Add price change data to property_data dictionary
        property_data['Price Changes'] = price_changes_data

    except Exception as e:
        print(f"Error occurred while extracting price changes: {e}")

    # Close the WebDriver
    driver.quit()

    # Return the extracted property data
    return property_data

# Usage example in another script
if __name__ == "__main__":
    # Example URL (replace with your own)
    url = "https://example.com/property-detail-page"
    result = scrape_property_data(url)

    # Printing the result for testing purposes
    print(result)
