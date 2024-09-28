import pandas as pd
import re
import logging

# Setup logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to clean currency fields
def clean_currency(value):
    """Removes currency symbols and commas, and converts to float."""
    if isinstance(value, str):
        value = value.replace('€', '').replace(',', '').strip()
        try:
            return float(value)
        except ValueError:
            return None
    return value

# Updated function to extract sale price and date from 'Price Changes'
def extract_sale_info(price_changes):
    """Extracts the sold price and sold date from the 'Price Changes' field."""
    if isinstance(price_changes, str):
        logging.debug(f"Extracting sale info from: {price_changes}")
        
        # Adjust regex to capture the sale info, trying different formats
        sale_patterns = [
            r"Sold, €([0-9,]+), [A-Za-z]{3} \d{2} \d{4}",  # Standard with full date and day
            r"Sold, €([0-9,]+), [A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4}",  # With day of the week
        ]
        
        for pattern in sale_patterns:
            sale_match = re.search(pattern, price_changes)
            if sale_match:
                sale_price = clean_currency(sale_match.group(1))  # Clean currency
                sale_date = sale_match.group(0).split(", ")[2]  # Extract date
                logging.info(f"Matched sale info: Price={sale_price}, Date={sale_date}")
                return sale_price, sale_date
        
        logging.warning(f"No match for sale info: {price_changes}")
    return None, None

# Updated function to extract first list price and date from 'Price Changes'
def extract_first_list_info(price_changes):
    """Extracts the first listing price and date from the 'Price Changes' field."""
    if isinstance(price_changes, str):
        logging.debug(f"Extracting first list info from: {price_changes}")
        
        # Adjust regex to capture the first list price and date, allowing different formats
        list_patterns = [
            r"Created, €([0-9,]+), [A-Za-z]{3} \d{2} \d{4}",  # Standard with full date
            r"Created, €([0-9,]+), [A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4}",  # With day of the week
            r"Created, €([0-9,]+), [A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4};",  # Ending with semicolon
        ]
        
        for pattern in list_patterns:
            list_match = re.search(pattern, price_changes)
            if list_match:
                list_price = clean_currency(list_match.group(1))  # Clean currency
                list_date = list_match.group(0).split(", ")[2]  # Extract the date
                logging.info(f"Matched first list info: Price={list_price}, Date={list_date}")
                return list_price, list_date
        
        logging.warning(f"No match for first list info: {price_changes}")
    return None, None

# Safe file reading with error handling
input_file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin_metadata/scraped_property_results_metadata_Dublin_page_1.csv'

try:
    logging.info(f"Reading CSV file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    logging.info("CSV file read successfully.")
except FileNotFoundError:
    logging.error(f"File not found: {input_file_path}")
    raise
except pd.errors.EmptyDataError:
    logging.error("No data found in CSV file.")
    raise
except Exception as e:
    logging.error(f"Error occurred while reading CSV: {e}")
    raise

# Apply the functions to extract sale and list price/date
try:
    logging.info("Applying sale price and date extraction.")
    df['Sale Price'], df['Sale Date'] = zip(*df['Price Changes'].apply(extract_sale_info))

    logging.info("Applying first list price and date extraction.")
    df['First List Price'], df['First List Date'] = zip(*df['Price Changes'].apply(extract_first_list_info))
except Exception as e:
    logging.error(f"Error occurred during data extraction: {e}")
    raise

# Clean 'Asking Price' and 'Local Property Tax' columns by stripping currency symbols
try:
    logging.info("Cleaning 'Asking Price' and 'Local Property Tax' columns.")
    df['Cleaned Asking Price'] = df['Asking Price'].apply(clean_currency)
    df['Cleaned Local Property Tax'] = df['Local Property Tax'].apply(clean_currency)
    logging.info("Currency cleaning completed.")
except KeyError as e:
    logging.error(f"Missing column: {e}")
    raise
except Exception as e:
    logging.error(f"Error occurred during currency cleaning: {e}")
    raise

# Convert 'Sale Date' and 'First List Date' to datetime objects
try:
    logging.info("Converting 'Sale Date' and 'First List Date' to datetime.")
    df['Sale Date'] = pd.to_datetime(df['Sale Date'], format='%b %d %Y', errors='coerce')
    df['First List Date'] = pd.to_datetime(df['First List Date'], format='%b %d %Y', errors='coerce')
    logging.info("Date conversion successful.")
except Exception as e:
    logging.error(f"Error during date conversion: {e}")
    raise

# Save the updated DataFrame to a new CSV file
output_file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/post_juypter_processing/scraped_property_results_metadata_Dublin_page_1_2024_10_28.csv'

try:
    logging.info(f"Saving processed data to CSV: {output_file_path}")
    df.to_csv(output_file_path, index=False)
    logging.info(f"Processed data saved to {output_file_path}")
except Exception as e:
    logging.error(f"Error while saving CSV: {e}")
    raise
