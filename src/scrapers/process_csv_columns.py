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
            logging.warning(f"Unable to convert value to float: {value}")
            return None
    return value

# Updated function to extract sale price and date from 'Price Changes'
def extract_sale_info(price_changes):
    """Extracts the sold price and sold date from the 'Price Changes' field."""
    if isinstance(price_changes, str):
        logging.debug(f"Extracting sale info from: {price_changes}")
        
        # Adjust regex to capture the sale info, allowing an optional semicolon at the end
        sale_patterns = [
            r"Sold, €([0-9,]+), ([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4})",  # With day of the week
            r"Sold, €([0-9,]+), ([A-Za-z]{3} \d{2} \d{4})",  # Without day of the week
        ]
        
        for pattern in sale_patterns:
            sale_match = re.search(pattern, price_changes)
            if sale_match:
                sale_price = clean_currency(sale_match.group(1))  # Clean currency
                sale_date_str = sale_match.group(2).strip()  # Extract date string
                logging.info(f"Matched sale info: Price={sale_price}, Date={sale_date_str}")
                return sale_price, sale_date_str
        
        logging.warning(f"No match for sale info: {price_changes}")
    return None, None

# Updated function to extract first list price and date from 'Price Changes'
def extract_first_list_info(price_changes):
    """Extracts the first listing price and date from the 'Price Changes' field."""
    if isinstance(price_changes, str):
        logging.debug(f"Extracting first list info from: {price_changes}")
        
        # Adjust regex to capture the first list price and date, allowing an optional semicolon at the end
        list_patterns = [
            r"Created, €([0-9,]+), ([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4})",  # With day of the week
            r"Created, €([0-9,]+), ([A-Za-z]{3} \d{2} \d{4})",  # Without day of the week
        ]
        
        for pattern in list_patterns:
            list_match = re.search(pattern, price_changes)
            if list_match:
                list_price = clean_currency(list_match.group(1))  # Clean currency
                list_date_str = list_match.group(2).strip()  # Extract date string
                logging.info(f"Matched first list info: Price={list_price}, Date={list_date_str}")
                return list_price, list_date_str
        
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
    sale_info = df['Price Changes'].apply(extract_sale_info)
    df['Sale Price'], df['Sale Date'] = zip(*sale_info)
    
    logging.info("Applying first list price and date extraction.")
    list_info = df['Price Changes'].apply(extract_first_list_info)
    df['First List Price'], df['First List Date'] = zip(*list_info)
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
    
    # Function to parse date strings with or without day of the week
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            # Try parsing with day of the week
            return pd.to_datetime(date_str, format='%a %b %d %Y', errors='coerce')
        except:
            pass
        try:
            # Try parsing without day of the week
            return pd.to_datetime(date_str, format='%b %d %Y', errors='coerce')
        except:
            pass
        # Fallback to infer format
        return pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')
    
    df['Sale Date'] = df['Sale Date'].apply(parse_date)
    df['First List Date'] = df['First List Date'].apply(parse_date)
    
    logging.info("Date conversion successful.")
except Exception as e:
    logging.error(f"Error during date conversion: {e}")
    raise

# Now, handle the latitude and longitude fields
try:
    logging.info("Extracting 'Latitude' and 'Longitude' columns.")
    df['Latitude'] = df['MyHome_Latitude']  # Assuming 'MyHome_Latitude' is the column in the CSV
    df['Longitude'] = df['MyHome_Longitude']  # Assuming 'MyHome_Longitude' is the column in the CSV
    logging.info("'Latitude' and 'Longitude' extraction successful.")
except KeyError as e:
    logging.error(f"Missing column for latitude or longitude: {e}")
    raise
except Exception as e:
    logging.error(f"Error occurred while extracting latitude and longitude: {e}")
    raise

# Optional: Validate and log the extracted dates
try:
    logging.debug("Validating extracted dates.")
    for index, row in df.iterrows():
        logging.debug(f"Row {index}: Sale Date - {row['Sale Date']}, First List Date - {row['First List Date']}")
except Exception as e:
    logging.error(f"Error during date validation: {e}")
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
