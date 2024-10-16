# ======================================
# Required Libraries
# ======================================

import sys
import json
import logging
import os
import traceback
import math
from dotenv import load_dotenv
from supabase import create_client, Client
from statistics import median, mode
import pandas as pd
from datetime import datetime, timedelta
import re
import numpy as np

# ======================================
# Step 1: Configuration and Initialization
# ======================================

# Load environment variables from .env.local
load_dotenv(".env.local")

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Fetch Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("Supabase credentials are missing in the environment variables.")
    print(json.dumps({"error": "Supabase credentials are missing in the environment variables."}))
    sys.exit(1)

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("Successfully connected to Supabase.")
except Exception as e:
    logging.error(f"Failed to connect to Supabase: {e}")
    print(json.dumps({"error": "Failed to connect to Supabase."}))
    sys.exit(1)

# ======================================
# Step 2: Define Helper Functions
# ======================================

def safe_divide(numerator, denominator):
    """
    Safely divide two numbers, returning None if the division is not possible.
    """
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return numerator / denominator

def get_price_details(price_changes: str):
    """
    Extracts the asking price at sale, sold price, sold date, and first list date from the Price Changes column.
    """
    if not isinstance(price_changes, str):
        return (None, None, None, None)
    
    # Initialize variables
    sold_price = None
    sold_date = None
    asking_price_at_sale = None
    all_dates = []
    
    # Split the price changes by ';' to process each event separately
    entries = price_changes.split(';')
    
    for i, entry in enumerate(entries):
        entry = entry.strip()
        
        # Extract the date from the entry
        date_match = re.search(r"([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{4})", entry)
        if date_match:
            date_str = date_match.group(1)
            try:
                date_obj = datetime.strptime(date_str, "%a %b %d %Y")
                all_dates.append(date_obj)
            except ValueError:
                # Handle unexpected date formats
                pass
        
        # Match the 'Sold' event
        sold_match = re.match(
            r"Sold,\s*€([\d,]+),\s*([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{4})",
            entry,
            re.IGNORECASE
        )
        if sold_match:
            sold_price = float(sold_match.group(1).replace(',', ''))
            sold_date = sold_match.group(2)
            
            # Look for the next relevant event to find the asking price at sale
            for next_entry in entries[i+1:]:
                next_entry = next_entry.strip()
                asking_match = re.match(
                    r"(Sale Agreed|Price Drop|Created),\s*€([\d,]+),\s*[A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{4}",
                    next_entry,
                    re.IGNORECASE
                )
                if asking_match:
                    asking_price_at_sale = float(asking_match.group(2).replace(',', ''))
                    break
            # Assuming only one 'Sold' event exists
            break
    
    # Determine the first list date (earliest date)
    if all_dates:
        first_list_date_obj = min(all_dates)
        first_list_date = first_list_date_obj.strftime("%a %b %d %Y")
    else:
        first_list_date = None
    
    return (asking_price_at_sale, sold_price, sold_date, first_list_date)

def ber_to_numeric(ber):
    """
    Convert BER rating to a numeric value.
    A1 is the best (highest value), G is the worst (lowest value).
    """
    ber_order = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E2', 'F', 'G']
    if pd.isna(ber) or ber == '--' or ber not in ber_order:
        return np.nan
    return float(len(ber_order) - ber_order.index(ber))

def get_property_type_category(property_type):
    if not property_type:
        return 'Unknown'
    property_type = property_type.lower().strip()
    if property_type in ['apartment', 'flat', 'studio']:
        return 'Apartment'
    elif property_type in ['house', 'bungalow', 'cottage', 'villa', 'townhouse', 'end of terrace', 'terrace', 'semi-d', 'detached', 'duplex']:
        return 'House'
    else:
        return 'Other'

def extract_numeric(value):
    """
    Extract numeric value from a string.
    """
    if pd.isna(value):
        return np.nan
    match = re.search(r'\d+', str(value))
    return int(match.group()) if match else np.nan

def get_bed_category(beds):
    bed_count = extract_numeric(beds)
    if pd.isna(bed_count):
        return "Unknown"
    if bed_count <= 1:
        return "Studio/1 Bed"
    elif bed_count == 2:
        return "2 Bed"
    elif bed_count == 3:
        return "3 Bed"
    else:
        return "4+ Bed"

def get_bath_category(baths):
    bath_count = extract_numeric(baths)
    if pd.isna(bath_count):
        return "Unknown"
    if bath_count <= 1:
        return "1 Bath"
    elif bath_count == 2:
        return "2 Bath"
    else:
        return "3+ Bath"

def get_ber_category(ber_rating):
    if not ber_rating or pd.isna(ber_rating):
        return 'Unknown'
    ber_rating = str(ber_rating).upper().strip()
    if ber_rating in ['A1', 'A2', 'A3']:
        return 'A'
    elif ber_rating in ['B1', 'B2', 'B3']:
        return 'B'
    elif ber_rating in ['C1', 'C2', 'C3']:
        return 'C'
    elif ber_rating in ['D1', 'D2']:
        return 'D'
    elif ber_rating in ['E1', 'E2']:
        return 'E'
    elif ber_rating == 'F':
        return 'F'
    elif ber_rating == 'G':
        return 'G'
    else:
        return 'Unknown'

def get_size_category(size):
    try:
        size = float(size)
    except (ValueError, TypeError):
        logging.warning(f"Invalid size: {size}")
        return "Unknown"
    if size < 50:
        return 'Small'
    elif 50 <= size < 100:
        return 'Medium'
    elif 100 <= size < 150:
        return 'Large'
    else:
        return 'Very Large'

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in kilometers.
    """
    try:
        R = 6371.0  # Radius of the Earth in kilometers

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance
    except Exception as e:
        logging.error(f"Error calculating Haversine distance: {e}")
        return None

def fetch_nearby_properties(latitude, longitude, radius_km):
    try:
        logging.info(f"Fetching properties within {radius_km} KM of ({latitude}, {longitude})")
        
        # Calculate the approximate bounding box
        lat_range = radius_km / 111.32  # 1 degree of latitude is approximately 111.32 km
        lon_range = radius_km / (111.32 * math.cos(math.radians(latitude))) if latitude else 0
        
        min_lat = latitude - lat_range
        max_lat = latitude + lat_range
        min_lon = longitude - lon_range
        max_lon = longitude + lon_range
        
        # Query the database using the bounding box
        response = supabase.table("scraped_property_data_v1") \
            .select("*") \
            .gte("latitude", min_lat) \
            .lte("latitude", max_lat) \
            .gte("longitude", min_lon) \
            .lte("longitude", max_lon) \
            .execute()
        
        all_properties = response.data
        logging.info(f"Properties within bounding box: {len(all_properties)}")
        
        nearby_properties = []
        for prop in all_properties:
            prop_lat = prop.get('latitude')
            prop_lon = prop.get('longitude')
            if prop_lat is not None and prop_lon is not None:
                try:
                    prop_lat = float(prop_lat)
                    prop_lon = float(prop_lon)
                    distance = calculate_distance(latitude, longitude, prop_lat, prop_lon)
                    if distance is not None and distance <= radius_km:
                        # Preprocess the property before adding it to nearby_properties
                        preprocessed_prop = preprocess_property_data(prop)
                        nearby_properties.append(preprocessed_prop)
                except ValueError:
                    logging.warning(f"Invalid coordinates for property: {prop.get('id')}")
            else:
                logging.debug(f"Skipping property with missing coordinates: {prop.get('id')}")
        
        logging.info(f"Number of nearby properties found within {radius_km}km: {len(nearby_properties)}")
        return nearby_properties
    except Exception as e:
        logging.error(f"Error fetching nearby properties: {e}")
        logging.error(traceback.format_exc())
        return []

def calculate_nearby_metrics(nearby_properties, radius):
    metrics = {}
    
    # Calculate average and median price per square meter
    price_per_sqm = [
        prop['price_per_square_meter'] 
        for prop in nearby_properties 
        if 'price_per_square_meter' in prop and pd.notna(prop['price_per_square_meter'])
    ]
    
    logging.debug(f"Number of properties with valid price_per_square_meter within {radius}km: {len(price_per_sqm)}")
    logging.debug(f"Sample of price_per_sqm values: {price_per_sqm[:5]}")

    if price_per_sqm:
        metrics[f'avg_price_per_sqm_within_{radius}km'] = np.nanmean(price_per_sqm)
        metrics[f'median_price_per_sqm_within_{radius}km'] = np.nanmedian(price_per_sqm)
        logging.debug(f"Calculated avg_price_per_sqm_within_{radius}km: {metrics[f'avg_price_per_sqm_within_{radius}km']}")
    else:
        metrics[f'avg_price_per_sqm_within_{radius}km'] = None
        metrics[f'median_price_per_sqm_within_{radius}km'] = None
        logging.warning(f"No valid price_per_square_meter values found for {radius}km radius")

    # Calculate other metrics (e.g., avg_asking_price) for comparison
    asking_prices = [
        float(prop['asking_price']) 
        for prop in nearby_properties 
        if prop.get('asking_price') and pd.notna(prop['asking_price'])
    ]
    
    if asking_prices:
        metrics[f'avg_asking_price_within_{radius}km'] = np.nanmean(asking_prices)
        metrics[f'median_asking_price_within_{radius}km'] = np.nanmedian(asking_prices)
        logging.debug(f"Calculated avg_asking_price_within_{radius}km: {metrics[f'avg_asking_price_within_{radius}km']}")
        logging.debug(f"Number of properties used for asking_price calculation: {len(asking_prices)}")
    else:
        metrics[f'avg_asking_price_within_{radius}km'] = None
        metrics[f'median_asking_price_within_{radius}km'] = None
        logging.warning(f"No valid asking_price values found for {radius}km radius")
    
    # Calculate average and median sold price
    sold_prices = [float(prop['sale_price']) for prop in nearby_properties if prop.get('sale_price') and not pd.isna(prop['sale_price'])]
    if sold_prices:
        metrics[f'avg_sold_price_within_{radius}km'] = np.nanmean(sold_prices)
        metrics[f'median_sold_price_within_{radius}km'] = np.nanmedian(sold_prices)
    else:
        metrics[f'avg_sold_price_within_{radius}km'] = None
        metrics[f'median_sold_price_within_{radius}km'] = None
    
    # Calculate average and median delta between asking and sold prices
    deltas = [float(prop['sale_price']) - float(prop['asking_price']) 
              for prop in nearby_properties 
              if prop.get('sale_price') and prop.get('asking_price') and not pd.isna(prop['sale_price']) and not pd.isna(prop['asking_price'])]
    if deltas:
        metrics[f'avg_price_delta_within_{radius}km'] = np.nanmean(deltas)
        metrics[f'median_price_delta_within_{radius}km'] = np.nanmedian(deltas)
    else:
        metrics[f'avg_price_delta_within_{radius}km'] = None
        metrics[f'median_price_delta_within_{radius}km'] = None
    
    # Calculate most common BER rating
    ber_ratings = [prop['energy_rating'] for prop in nearby_properties if prop.get('energy_rating') and not pd.isna(prop['energy_rating'])]
    if ber_ratings:
        try:
            metrics[f'most_common_ber_rating_within_{radius}km'] = mode(ber_ratings)
        except:
            metrics[f'most_common_ber_rating_within_{radius}km'] = None
    else:
        metrics[f'most_common_ber_rating_within_{radius}km'] = None
    
    # Calculate property type distribution
    property_types = [prop['property_type'] for prop in nearby_properties if prop.get('property_type') and not pd.isna(prop['property_type'])]
    if property_types:
        type_distribution = {t: property_types.count(t) / len(property_types) * 100 for t in set(property_types)}
        metrics[f'property_type_distribution_within_{radius}km'] = type_distribution
    else:
        metrics[f'property_type_distribution_within_{radius}km'] = {}
    
    beds = [prop['beds'] for prop in nearby_properties if pd.notna(prop['beds'])]
    baths = [prop['baths'] for prop in nearby_properties if pd.notna(prop['baths'])]

    logging.info(f"Number of properties with valid 'beds' within {radius}km: {len(beds)}")
    logging.info(f"Number of properties with valid 'baths' within {radius}km: {len(baths)}")
    logging.debug(f"Sample of 'beds' values: {beds[:5]}")
    logging.debug(f"Sample of 'baths' values: {baths[:5]}")

    if beds:
        metrics[f'avg_bedrooms_within_{radius}km'] = np.mean(beds)
        logging.info(f"Calculated avg_bedrooms_within_{radius}km: {metrics[f'avg_bedrooms_within_{radius}km']}")
    else:
        metrics[f'avg_bedrooms_within_{radius}km'] = None
        logging.warning(f"No valid 'beds' values found for {radius}km radius")

    if baths:
        metrics[f'avg_bathrooms_within_{radius}km'] = np.mean(baths)
        logging.info(f"Calculated avg_bathrooms_within_{radius}km: {metrics[f'avg_bathrooms_within_{radius}km']}")
    else:
        metrics[f'avg_bathrooms_within_{radius}km'] = None
        logging.warning(f"No valid 'baths' values found for {radius}km radius")
    
    metrics[f'nearby_properties_count_within_{radius}km'] = len(nearby_properties)
    
    return metrics

def preprocess_property_data(prop):
    try:
        # Convert numeric fields
        numeric_fields = ['sale_price', 'myhome_floor_area_value', 'latitude', 'longitude', 'asking_price']
        for field in numeric_fields:
            if field in prop:
                prop[field] = pd.to_numeric(prop[field], errors='coerce')
        
        # Extract numeric values for beds and baths
        prop['beds'] = extract_numeric(prop.get('beds'))
        prop['baths'] = extract_numeric(prop.get('baths'))
        
        logging.debug(f"Property ID: {prop.get('id')}, Beds: {prop.get('beds')}, Baths: {prop.get('baths')}")

        # Calculate price per square meter using sale price
        if pd.notna(prop['sale_price']) and pd.notna(prop['myhome_floor_area_value']) and prop['myhome_floor_area_value'] > 0:
            prop['price_per_square_meter'] = prop['sale_price'] / prop['myhome_floor_area_value']
            logging.debug(f"Calculated price_per_square_meter: {prop['price_per_square_meter']}")
        else:
            prop['price_per_square_meter'] = None
            logging.debug(f"Unable to calculate price_per_square_meter. sale_price: {prop.get('sale_price')}, myhome_floor_area_value: {prop.get('myhome_floor_area_value')}")

        return prop
    except Exception as e:
        logging.error(f"Error in preprocess_property_data for property ID {prop.get('id')}: {e}")
        return prop

def generate_columns(data):
    try:
        logging.info("Starting generate_columns function.")
        
        # Preprocess the property data
        preprocessed_data = preprocess_property_data(data)

        # Generate derived categories
        result = {
            'bedCategory': get_bed_category(preprocessed_data.get('beds', '0')),
            'bathCategory': get_bath_category(preprocessed_data.get('baths', '0')),
            'propertyTypeCategory': get_property_type_category(preprocessed_data.get('property_type', '')),
            'berCategory': get_ber_category(preprocessed_data.get('energy_rating', '')),
            'sizeCategory': get_size_category(preprocessed_data.get('myhome_floor_area_value', 0)),
            'originalInputs': {k: v for k, v in preprocessed_data.items() if k.lower() not in [
                'asking_price', 'eircode', 'local_property_tax', 'url',
                'myhome_link', 'price_per_square_meter',
                'sale_price', 'sale_date', 'first_list_price', 'first_list_date'
            ]},
            'latitude': preprocessed_data.get('latitude'),
            'longitude': preprocessed_data.get('longitude'),
            'size': preprocessed_data.get('myhome_floor_area_value'),
        }

        # Fetch nearby properties and calculate metrics for 1km, 3km, and 5km
        if result['latitude'] is not None and result['longitude'] is not None:
            try:
                result['latitude'] = float(result['latitude'])
                result['longitude'] = float(result['longitude'])
                for radius in [1, 3, 5]:
                    nearby_props = fetch_nearby_properties(result['latitude'], result['longitude'], radius_km=radius)
                    result[f'nearby_properties_count_within_{radius}km'] = len(nearby_props)
                    if nearby_props:
                        nearby_metrics = calculate_nearby_metrics(nearby_props, radius)
                        for key, value in nearby_metrics.items():
                            if value is not None:
                                result[key] = value
                            else:
                                logging.warning(f"Metric {key} is None for radius {radius}km")
                                result[key] = None
                    else:
                        logging.warning(f"No nearby properties found within {radius}km to calculate metrics.")
                        result[f'avg_bedrooms_within_{radius}km'] = None
                        result[f'avg_bathrooms_within_{radius}km'] = None
                        # ... initialize other metrics ...
            except ValueError:
                logging.error(f"Invalid latitude or longitude: {result['latitude']}, {result['longitude']}")
                for radius in [1, 3, 5]:
                    result[f'nearby_properties_count_within_{radius}km'] = 0
                    result[f'avg_sold_price_within_{radius}km'] = None
                    result[f'median_sold_price_within_{radius}km'] = None
                    result[f'avg_asking_price_within_{radius}km'] = None
                    result[f'median_asking_price_within_{radius}km'] = None
                    result[f'avg_price_delta_within_{radius}km'] = None
                    result[f'median_price_delta_within_{radius}km'] = None
                    result[f'avg_price_per_sqm_within_{radius}km'] = None
                    result[f'median_price_per_sqm_within_{radius}km'] = None
                    result[f'most_common_ber_rating_within_{radius}km'] = None
                    result[f'property_type_distribution_within_{radius}km'] = {}
                    result[f'avg_bedrooms_within_{radius}km'] = None
                    result[f'avg_bathrooms_within_{radius}km'] = None
        else:
            logging.warning("Latitude or longitude is missing, skipping nearby properties calculation.")
            for radius in [1, 3, 5]:
                result[f'nearby_properties_count_within_{radius}km'] = 0
                result[f'avg_sold_price_within_{radius}km'] = None
                result[f'median_sold_price_within_{radius}km'] = None
                result[f'avg_asking_price_within_{radius}km'] = None
                result[f'median_asking_price_within_{radius}km'] = None
                result[f'avg_price_delta_within_{radius}km'] = None
                result[f'median_price_delta_within_{radius}km'] = None
                result[f'avg_price_per_sqm_within_{radius}km'] = None
                result[f'median_price_per_sqm_within_{radius}km'] = None
                result[f'most_common_ber_rating_within_{radius}km'] = None
                result[f'property_type_distribution_within_{radius}km'] = {}
                result[f'avg_bedrooms_within_{radius}km'] = None
                result[f'avg_bathrooms_within_{radius}km'] = None

        logging.info("Finished generate_columns function.")
        return result
    except Exception as e:
        logging.error(f"Error in generate_columns for property ID {data.get('id')}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# ======================================
# Step 3: Data Processing Pipeline
# ======================================

def process_all_properties(output_csv_path):
    """
    Fetch all properties from Supabase, generate metrics for each, and save to a CSV file.
    
    Args:
        output_csv_path (str): Path to save the processed CSV file.
    """
    try:
        logging.info("Starting data processing pipeline.")

        # Fetch all properties from Supabase
        logging.info("Fetching properties from Supabase.")
        response = supabase.table("scraped_property_data_v1").select("*").execute()
        all_properties = response.data
        logging.info(f"Total properties fetched: {len(all_properties)}")

        if not all_properties:
            logging.warning("No properties fetched from Supabase. Exiting pipeline.")
            return

        # Initialize a list to store processed property data
        processed_data = []

        # Iterate over each property and generate metrics
        for idx, prop in enumerate(all_properties, start=1):
            logging.info(f"Processing property {idx}/{len(all_properties)} with ID: {prop.get('id')}")
            try:
                # Generate columns/metrics for the property
                metrics = generate_columns(prop)
                # Combine original property data with generated metrics
                combined_data = {**prop, **metrics}
                processed_data.append(combined_data)
            except Exception as e:
                logging.error(f"Failed to process property ID {prop.get('id')}: {e}")
                logging.error(traceback.format_exc())
                continue

        # Create a DataFrame from the processed data
        logging.info("Creating DataFrame from processed data.")
        df_processed = pd.DataFrame(processed_data)
        logging.info(f"Processed DataFrame shape: {df_processed.shape}")

        # Save the processed DataFrame to a CSV file
        logging.info(f"Saving processed data to {output_csv_path}.")
        df_processed.to_csv(output_csv_path, index=False)
        logging.info("Data processing pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Error in process_all_properties: {e}")
        logging.error(traceback.format_exc())
        raise

# ======================================
# Step 4: Main Execution
# ======================================

def main():
    try:
        logging.debug("Python script started.")
        logging.debug(f"Current working directory: {os.getcwd()}")

        # Define output file path for the full production run
        output_csv = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/full_run_predictions_xgboost_v3.csv'

        # Start the data processing pipeline
        process_all_properties(output_csv)

        logging.info(f"Data processing completed. Output saved to {output_csv}")

        # Perform a preview of the processed data
        try:
            processed_df = pd.read_csv(output_csv)
            logging.info("\nDataFrame Preview:")
            print(processed_df.head())

            # Display column information
            logging.info("\nColumn Information:")
            column_info = pd.DataFrame({
                'Column Name': processed_df.columns,
                'Data Type': processed_df.dtypes
            })
            print(column_info)

            # Display statistical summary
            logging.info("\nStatistical Summary:")
            numeric_cols = processed_df.select_dtypes(include=['number']).columns
            stats = processed_df[numeric_cols].agg(['mean', 'median', 'max', 'min', 'std']).transpose()
            stats = stats.rename(columns={
                'mean': 'Mean',
                'median': 'Median',
                'max': 'Max',
                'min': 'Min',
                'std': 'Std Dev'
            })
            print(stats)

        except Exception as e:
            logging.error(f"Failed to read or analyze the processed CSV file: {e}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error(traceback.format_exc())
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()