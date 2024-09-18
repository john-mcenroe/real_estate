import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize the Chrome WebDriver using webdriver-manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Navigate to the URL
region = "Dublin"  # Parameterized region name
url = f"https://mynest.ie/priceregister/{region}"
driver.get(url)

# Explicitly wait for the table to load
wait = WebDriverWait(driver, 10)

# Set to store property data
property_data = []

# Add a counter for test mode
record_limit = 2  # Set the limit to 2 records for testing
record_count = 0  # Initialize the record count

# Function to scrape property details
def scrape_property_data():
    try:
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
        except Exception as e:
            print(f"Price Changes not found: {e}")

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

    except Exception as e:
        print(f"Error occurred while scraping property data: {e}")
        return None

def extract_data():
    global record_count  # Access the global counter for record tracking

    # Keep track of processed addresses
    processed_addresses = set()

    while True:
        try:
            # Wait for the rows to load
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')))
            
            # Re-fetch the list of rows after every navigation
            rows = driver.find_elements(By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')
            
            # Process each row one by one
            for row in rows:
                address = row.text.strip()  # Extract the text and clean up spaces

                # Skip the row if it's already processed
                if address in processed_addresses:
                    continue

                try:
                    print(f"Processing: {address}")
                    
                    # Click the address to go to the detailed page
                    row.click()

                    # Wait for 2 seconds to ensure the page is loaded
                    time.sleep(2)

                    # Scrape property data
                    data = scrape_property_data()
                    if data:
                        property_data.append(data)
                        print(f"Data scraped for {address}: {data}")

                    # Add the address to the processed list
                    processed_addresses.add(address)

                    # Increment the record count and stop after reaching the limit
                    record_count += 1
                    if record_count >= record_limit:
                        print(f"Stopping after {record_limit} records.")
                        return  # Exit the function after hitting the record limit

                    # Go back to the main listing page
                    driver.back()

                    # Wait for the rows to reload
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.fancy-Rtable-cell--content.fancy-title-content')))
                    
                    # Pause to ensure the page is fully loaded
                    time.sleep(2)

                except Exception as e:
                    print(f"Error processing {address}: {e}")

            # Exit the loop once all rows have been processed
            break

        except Exception as e:
            print(f"Error locating elements: {e}")
            continue  # In case of error, re-attempt to locate elements

# Scrape the data
extract_data()

# Close the driver after scraping is done
driver.quit()

# Save the output results to a CSV file
csv_filename = f"scraped_property_results_{region}.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Address', 'Asking Price', 'Beds', 'Baths', 'Property Type',
        'Energy Rating', 'Eircode', 'Local Property Tax', 'Agency Name',
        'Agency Contact', 'Price Changes'
    ])  # Write the header
    for data in property_data:
        writer.writerow([
            data['Address'], data['Asking Price'], data['Beds'], data['Baths'],
            data['Property Type'], data['Energy Rating'], data['Eircode'],
            data['Local Property Tax'], data['Agency Name'], data['Agency Contact'],
            "; ".join([f"{p['Change']}, {p['Price']}, {p['Date']}" for p in data['Price Changes']])
        ])

print(f"\nProperty details saved to {csv_filename}")
