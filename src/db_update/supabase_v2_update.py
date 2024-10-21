import os
import pandas as pd
from supabase import create_client, Client
import logging
import json
import sys
import uuid

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Step 1: Read the CSV file and limit to first 3000 rows for testing
csv_file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/post_juypter_processing/scraped_property_results_metadata_Dublin_page_1_2024_10_28.csv'
df = pd.read_csv(csv_file_path).head(3000)

# Step 2: Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Step 3: Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_table_if_not_exists():
    """Checks if the scraped_property_data_v2 table exists."""
    try:
        # Check if the table exists
        response = supabase.table('scraped_property_data_v2').select('id').limit(1).execute()
        logging.info("Table 'scraped_property_data_v2' exists.")
    except Exception as e:
        if "relation" in str(e) and "does not exist" in str(e):
            logging.error("Table 'scraped_property_data_v2' does not exist. Please create it manually.")
            sys.exit(1)
        else:
            logging.error(f"Unexpected error checking table existence: {e}")
            raise

# Call the function to create the table if it doesn't exist
create_table_if_not_exists()

# Step 5: Clean data function to handle NaN values and generate ID if missing
def clean_data(row, index):
    """Convert NaN to None for JSON insertion, ensure valid types, and generate ID if missing."""
    cleaned_row = {}
    for key, value in row.items():
        # Replace NaN with None
        if pd.isna(value):
            cleaned_row[key] = None
        else:
            cleaned_row[key] = value
    
    # Generate a unique ID if it's missing
    if 'id' not in cleaned_row or pd.isna(cleaned_row['id']):
        cleaned_row['id'] = str(uuid.uuid4())  # Generate a UUID as a string
    
    return cleaned_row

# Step 6: Insert rows into the table using cleaned columns
for index, row in df.iterrows():
    data = {
        'address': row.get('address'),
        'asking_price': row.get('asking_price'),
        'beds': row.get('beds'),
        'baths': row.get('baths'),
        'property_type': row.get('property_type'),
        'energy_rating': row.get('energy_rating'),
        'eircode': row.get('eircode'),
        'local_property_tax': row.get('local_property_tax'),
        'url': row.get('url'),
        'myhome_floor_area_value': row.get('myhome_floor_area_value'),
        'myhome_link': row.get('myhome_link'),
        'price_per_square_meter': row.get('price_per_square_meter'),
        'sale_price': row.get('sale_price'),
        'sale_date': row.get('sale_date'),
        'first_list_price': row.get('first_list_price'),
        'first_list_date': row.get('first_list_date'),
        'latitude': row.get('latitude'),
        'longitude': row.get('longitude')
    }

    # Clean the data to handle NaN values and generate ID if missing
    cleaned_data = clean_data(data, index)

    # Log the data before sending to Supabase
    logging.debug(f"Attempting to insert data: {json.dumps(cleaned_data, indent=2, default=str)}")

    try:
        # Step 7: Insert the row into the Supabase table
        res = supabase.table('scraped_property_data_v2').insert(cleaned_data).execute()

        # Check if insertion was successful
        if res.status_code == 201:
            logging.info(f"Inserted row with Address: {cleaned_data.get('address')}")
        else:
            logging.error(f"Failed to insert row with Address: {cleaned_data.get('address')}")
            logging.error(f"Response: {res.json()}")
    except Exception as e:
        logging.error(f"Error inserting data for row {index}: {e}")
