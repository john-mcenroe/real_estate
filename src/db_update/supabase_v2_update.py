import os
import pandas as pd
from supabase import create_client, Client
import logging
import json
import sys
import re
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions
def clean_currency(value):
    if isinstance(value, str):
        value = value.replace('€', '').replace(',', '').strip()
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Unable to convert value to float: {value}")
            return None
    elif pd.isna(value):
        return None
    return value

def extract_sale_info(price_changes):
    if isinstance(price_changes, str):
        sale_patterns = [
            r"Sold, €([0-9,]+), ([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4})",
            r"Sold, €([0-9,]+), ([A-Za-z]{3} \d{2} \d{4})",
        ]
        for pattern in sale_patterns:
            sale_match = re.search(pattern, price_changes)
            if sale_match:
                sale_price = clean_currency(sale_match.group(1))
                sale_date_str = sale_match.group(2).strip()
                return sale_price, sale_date_str
    return None, None

def extract_first_list_info(price_changes):
    if isinstance(price_changes, str):
        list_patterns = [
            r"Created, €([0-9,]+), ([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4})",
            r"Created, €([0-9,]+), ([A-Za-z]{3} \d{2} \d{4})",
        ]
        for pattern in list_patterns:
            list_match = re.search(pattern, price_changes)
            if list_match:
                list_price = clean_currency(list_match.group(1))
                list_date_str = list_match.group(2).strip()
                return list_price, list_date_str
    return None, None

def parse_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        return datetime.strptime(date_str, '%a %b %d %Y')
    except:
        try:
            return datetime.strptime(date_str, '%b %d %Y')
        except:
            return None

def compute_price_per_square_metre(sale_price, area, area_unit):
    if pd.isna(sale_price) or pd.isna(area) or area == 0:
        return None
    
    if area_unit:
        area_unit = area_unit.lower()
        if area_unit in ['sqft', 'square feet', 'ft²', 'ft2']:
            area = area * 0.092903  # Convert square feet to square metres
    
    try:
        return round(sale_price / area, 2)
    except:
        return None

def parse_integer(value):
    if not value or value == '--':
        return None
    try:
        return int(value.split()[0])
    except ValueError:
        logging.warning(f"Unable to parse integer from: {value}")
        return None

# Main data processing function
def process_row(row, index):
    logging.info(f"Processing row {index + 1}")
    processed = {}

    # Assign ID from scratch, starting from 1
    processed['id'] = index + 1

    # Basic fields
    processed['address'] = row.get('Address', None)
    processed['property_type'] = row.get('Property Type', None)
    processed['energy_rating'] = row.get('Energy Rating', None)
    processed['eircode'] = row.get('Eircode', None)
    processed['url'] = row.get('URL', None) if pd.notna(row.get('URL', None)) else None
    processed['myhome_link'] = row.get('MyHome_Link', None)

    # Numeric fields with error handling
    processed['asking_price'] = clean_currency(row.get('Asking Price', None))
    processed['local_property_tax'] = clean_currency(row.get('Local Property Tax', None))

    # Handle 'Beds' and 'Baths' more carefully
    processed['beds'] = parse_integer(row.get('Beds', None))
    processed['baths'] = parse_integer(row.get('Baths', None))

    # Ensure numeric conversion and handle NaN
    processed['myhome_floor_area_value'] = pd.to_numeric(row.get('MyHome_Floor_Area_Value'), errors='coerce')
    if pd.isna(processed['myhome_floor_area_value']):
        processed['myhome_floor_area_value'] = None

    # Extract sale and list info
    sale_price, sale_date = extract_sale_info(row.get('Price Changes', None))
    first_list_price, first_list_date = extract_first_list_info(row.get('Price Changes', None))

    processed['sale_price'] = sale_price
    processed['sale_date'] = parse_date(sale_date).isoformat() if parse_date(sale_date) else None
    processed['first_list_price'] = first_list_price
    processed['first_list_date'] = parse_date(first_list_date).isoformat() if parse_date(first_list_date) else None

    # Compute price per square metre
    processed['price_per_square_meter'] = compute_price_per_square_metre(
        sale_price, 
        processed['myhome_floor_area_value'], 
        row.get('MyHome_Floor_Area_Unit', None)
    )

    # Latitude and Longitude with error handling
    processed['latitude'] = pd.to_numeric(row.get('MyHome_Latitude'), errors='coerce')
    processed['longitude'] = pd.to_numeric(row.get('MyHome_Longitude'), errors='coerce')

    # Validate JSON compatibility
    try:
        json.dumps(processed, default=str)
    except (TypeError, ValueError) as e:
        logging.error(f"Invalid JSON data for row {index + 1}: {e}")
        return None

    return processed

# Supabase setup
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clear_table():
    try:
        # Delete all rows from the table
        res = supabase.table('scraped_property_data_v2').delete().neq('id', 0).execute()
        logging.info("Cleared all data from scraped_property_data_v2 table.")
    except Exception as e:
        logging.error(f"Error clearing table: {e}")
        sys.exit(1)

# Main execution
if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Process and insert property data.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited rows')
    args = parser.parse_args()

    # Clear the table before inserting new data
    clear_table()

    # Read and process CSV
    csv_file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/scraped_property_results_Dublin_final_with_metadata_deduped.csv'
    df = pd.read_csv(csv_file_path)

    # Limit rows if in test mode
    if args.test:
        df = df.head(50)

    # Process and insert rows
    rows_processed = 0
    rows_failed = 0
    failed_rows = []  # List to store failed rows and their errors

    for index, row in df.iterrows():
        processed_data = process_row(row, index)
        
        logging.debug(f"Processed data for row {index + 1}:")
        logging.debug(json.dumps(processed_data, indent=2, default=str))
        
        try:
            res = supabase.table('scraped_property_data_v2').insert(processed_data).execute()
            
            # Check if the insert was successful
            if res.data:
                logging.info(f"Inserted row with ID: {processed_data['id']}, Address: {processed_data.get('address')}")
                rows_processed += 1
            else:
                logging.error(f"Failed to insert row with ID: {processed_data['id']}, Address: {processed_data.get('address')}")
                logging.error(f"Response: {res}")
                failed_rows.append((processed_data, "Failed to insert"))
                rows_failed += 1
        except Exception as e:
            logging.error(f"Error inserting data for row {index + 1}: {e}")
            logging.error(f"Data causing error: {json.dumps(processed_data, indent=2, default=str)}")
            failed_rows.append((processed_data, str(e)))
            rows_failed += 1

    logging.info(f"Data processing and insertion complete. Processed {rows_processed} rows successfully, {rows_failed} rows failed.")
    
    if rows_failed > 0:
        logging.warning(f"Some rows failed to insert. Please check the logs for details.")
        
        # Print two failed rows
        for i, (failed_row, error) in enumerate(failed_rows[:2]):
            logging.info(f"Failed Row {i+1}: {json.dumps(failed_row, indent=2, default=str)}")
            logging.info(f"Error: {error}")
