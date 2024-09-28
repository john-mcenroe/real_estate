import os
import pandas as pd
from supabase import create_client, Client
import logging
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Step 1: Read the CSV file and limit to first 50 rows
csv_file_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/post_juypter_processing/scraped_property_results_metadata_Dublin_page_1_2024_10_28.csv'
df = pd.read_csv(csv_file_path).head(2)  ####### Limit to the first 2000 rows for testing

# Step 2: Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Step 3: Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Step 4: Clean data function to handle NaN values
def clean_data(row):
    """Convert NaN to None for JSON insertion and ensure valid types."""
    cleaned_row = {}
    for key, value in row.items():
        # Replace NaN with None
        if pd.isna(value):
            cleaned_row[key] = None
        else:
            cleaned_row[key] = value
    return cleaned_row

# Step 5: Insert rows into the table using cleaned columns
for _, row in df.iterrows():
    data = {
        'address': row['Address'],
        'asking_price': row['Cleaned Asking Price'],  # Use cleaned asking price
        'beds': row['Beds'],
        'baths': row['Baths'],
        'property_type': row['Property Type'],
        'energy_rating': row['Energy Rating'],
        'eircode': row['Eircode'],
        'local_property_tax': row['Cleaned Local Property Tax'],  # Use cleaned local property tax
        'url': row['URL'],
        'myhome_floor_area_value': row['MyHome_Floor_Area_Value'],
        'myhome_link': row['MyHome_Link'],
        'price_per_square_meter': row.get('price_per_square_meter', None),
        'sale_price': row['Sale Price'],
        'sale_date': row['Sale Date'],
        'first_list_price': row['First List Price'],
        'first_list_date': row['First List Date'],
        'latitude': row['Latitude'],  # Add latitude
        'longitude': row['Longitude']  # Add longitude
    }

    # Clean the data to handle NaN values
    cleaned_data = clean_data(data)

    # Log the data before sending to Supabase
    logging.debug(f"Attempting to insert data: {json.dumps(cleaned_data, indent=2)}")

    try:
        # Step 6: Insert the row into the Supabase table
        res = supabase.table('scraped_property_data_v1').insert(cleaned_data).execute()

        # Check if insertion was successful
        if res.status_code == 201:
            logging.info(f"Inserted row with Address: {row['Address']}")
        else:
            logging.error(f"Failed to insert row with Address: {row['Address']}")
            logging.error(f"Response: {res.json()}")
    except Exception as e:
        logging.error(f"Error inserting data for row {row['Address']}: {e}")
