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

OUTPUT_CSV = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/processed_data_v3_output.csv'

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
    ber_order = ['A', 'A1', 'A2', 'A3', 'B', 'B1', 'B2', 'B3', 'C', 'C1', 'C2', 'C3', 'D', 'D1', 'D2', 'E', 'E1', 'E2', 'F', 'G']
    if pd.isna(ber) or ber == '--' or ber not in ber_order:
        return np.nan
    return float(len(ber_order) - ber_order.index(ber))

def get_property_type_category(property_type):
    """
    Categorize property types into broader categories.
    """
    if not property_type:
        return 'Unknown'
    property_type = property_type.lower().strip()
    if property_type in ['apartment', 'flat', 'studio']:
        return 'Apartment'
    elif property_type in ['house', 'bungalow', 'cottage', 'villa', 'townhouse', 'end of terrace', 'terrace', 'semi-d', 'detached', 'duplex']:
        return 'House'
    else:
        return 'Other'

def get_bed_category(beds):
    """
    Categorize the number of bedrooms.
    """
    if pd.isna(beds):
        return 'Unknown'
    try:
        beds = int(beds)
        if beds <= 1:
            return '1 or less'
        elif beds == 2:
            return '2'
        elif beds == 3:
            return '3'
        else:
            return '4 or more'
    except ValueError:
        return 'Unknown'

def get_bath_category(baths):
    """
    Categorize the number of bathrooms.
    """
    if pd.isna(baths):
        return 'Unknown'
    try:
        baths = int(baths)
        if baths <= 1:
            return '1 or less'
        elif baths == 2:
            return '2'
        elif baths == 3:
            return '3'
        else:
            return '4 or more'
    except ValueError:
        return 'Unknown'

def get_ber_category(ber):
    """
    Categorize BER ratings.
    """
    if pd.isna(ber):
        return 'Unknown'
    ber_numeric = ber_to_numeric(ber)
    if ber_numeric >= 15:
        return 'A1'
    elif ber_numeric >= 14:
        return 'A2'
    elif ber_numeric >= 13:
        return 'A3'
    elif ber_numeric >= 12:
        return 'B1'
    elif ber_numeric >= 11:
        return 'B2'
    elif ber_numeric >= 10:
        return 'B3'
    elif ber_numeric >= 9:
        return 'C1'
    elif ber_numeric >= 8:
        return 'C2'
    elif ber_numeric >= 7:
        return 'C3'
    elif ber_numeric >= 6:
        return 'D1'
    elif ber_numeric >= 5:
        return 'D2'
    elif ber_numeric >= 4:
        return 'E1'
    elif ber_numeric >= 3:
        return 'E2'
    elif ber_numeric >= 2:
        return 'F'
    else:
        return 'G'

def get_size_category(floor_area):
    """
    Categorize property size based on floor area.
    """
    if pd.isna(floor_area):
        return 'Unknown'
    if floor_area < 50:
        return 'Small'
    elif 50 <= floor_area < 100:
        return 'Medium'
    else:
        return 'Large'

def extract_numeric(value):
    """
    Extract numeric value from a string.
    """
    if pd.isna(value):
        return np.nan
    match = re.search(r'\d+', str(value))
    return int(match.group()) if match else np.nan

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert coordinates to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def preprocess_property_data(prop):
    """
    Preprocess individual property data.
    """
    try:
        # Convert numeric fields
        numeric_fields = ['sale_price', 'myhome_floor_area_value', 'latitude', 'longitude', 'asking_price']
        for field in numeric_fields:
            if field in prop:
                prop[field] = pd.to_numeric(prop[field], errors='coerce')
        
        # Extract numeric values for beds and baths
        prop['beds'] = extract_numeric(prop.get('beds'))
        prop['baths'] = extract_numeric(prop.get('baths'))
        
        # Remove this debug logging
        # logging.debug(f"Property ID: {prop.get('id')}, Beds: {prop.get('beds')}, Baths: {prop.get('baths')}")

        # Calculate price per square meter using sale price
        if pd.notna(prop['sale_price']) and pd.notna(prop['myhome_floor_area_value']) and prop['myhome_floor_area_value'] > 0:
            prop['price_per_square_meter'] = prop['sale_price'] / prop['myhome_floor_area_value']
            # Remove this debug logging
            # logging.debug(f"Calculated price_per_square_meter: {prop['price_per_square_meter']}")
        else:
            prop['price_per_square_meter'] = None
            # Remove this debug logging
            # logging.debug(f"Unable to calculate price_per_square_meter. sale_price: {prop.get('sale_price')}, myhome_floor_area_value: {prop.get('myhome_floor_area_value')}")
    
        return prop
    except Exception as e:
        logging.error(f"Error in preprocess_property_data for property ID {prop.get('id')}: {e}")
        return prop

def fetch_nearby_properties(latitude, longitude, radius_km=3):
    """
    Fetch nearby properties within a specified radius from Supabase.
    """
    try:
        # Convert latitude and longitude to float
        lat, lon = float(latitude), float(longitude)
        
        # Calculate bounding box
        lat_range = radius_km / 111.32  # 1 degree of latitude is approximately 111.32 km
        lon_range = radius_km / (111.32 * math.cos(math.radians(lat)))
        
        min_lat, max_lat = lat - lat_range, lat + lat_range
        min_lon, max_lon = lon - lon_range, lon + lon_range
        
        query = (
            supabase.table("scraped_property_data_v1")
            .select("*")
            .gte("latitude", min_lat)
            .lte("latitude", max_lat)
            .gte("longitude", min_lon)
            .lte("longitude", max_lon)
            .execute()
        )
        properties = query.data
        
        nearby = [
            prop for prop in properties
            if haversine_distance(lat, lon, float(prop['latitude']), float(prop['longitude'])) <= radius_km
        ]
        
        logging.info(f"Found {len(nearby)} properties within {radius_km}km radius")
        return nearby
    except Exception as e:
        logging.error(f"Error fetching nearby properties: {e}")
        return []

def ber_to_category(ber):
    """
    Convert BER rating to category (A, B, C, D, E, F, G).
    """
    if pd.isna(ber):
        return 'Unknown'
    ber = ber.upper()
    if ber.startswith('A'):
        return 'A'
    elif ber.startswith('B'):
        return 'B'
    elif ber.startswith('C'):
        return 'C'
    elif ber.startswith('D'):
        return 'D'
    elif ber.startswith('E'):
        return 'E'
    elif ber.startswith('F'):
        return 'F'
    elif ber.startswith('G'):
        return 'G'
    else:
        return 'Unknown'

def calculate_time_based_metrics(df, days, radius):
    """
    Calculate time-based metrics for a given number of days and radius.
    """
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent_df = df[df['sale_date'] >= cutoff_date]
    
    metrics = {
        f'{days}d_{radius}km_median_sold_price': recent_df['sale_price'].median(),
        f'{days}d_{radius}km_avg_asking_price': recent_df['asking_price'].mean(),
        f'{days}d_{radius}km_num_properties_sold': recent_df['sale_price'].notna().sum(),
        f'{days}d_{radius}km_avg_days_on_market': (recent_df['sale_date'] - recent_df['first_list_date']).dt.days.mean(),
        f'{days}d_{radius}km_median_price_per_sqm': (recent_df['sale_price'] / recent_df['myhome_floor_area_value']).median(),
    }
    return metrics

def calculate_nearby_metrics(nearby_props, radius):
    """
    Calculate metrics for nearby properties within a specified radius.
    """
    metrics = {}
    try:
        df = pd.DataFrame(nearby_props)
        
        # Convert 'beds' and 'baths' to numeric using extract_numeric function
        for col in ['beds', 'baths']:
            df[col] = df[col].apply(extract_numeric)
        
        # Convert date columns to datetime
        date_columns = ['sale_date', 'first_list_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate metrics for different time periods
        for days in [30, 90, 180]:
            time_metrics = calculate_time_based_metrics(df, days, radius)
            metrics.update(time_metrics)
        
        # BER distribution
        df['ber_category'] = df['energy_rating'].apply(ber_to_category)
        ber_dist = df['ber_category'].value_counts(normalize=True) * 100
        for ber, percent in ber_dist.items():
            metrics[f'{radius}km_ber_dist_{ber}'] = percent
        
        # Property type distribution
        prop_type_dist = df['property_type'].value_counts(normalize=True) * 100
        for prop_type, percent in prop_type_dist.items():
            metrics[f'{radius}km_property_type_dist_{prop_type}'] = percent
        
        # Other general metrics
        metrics.update({
            f'{radius}km_avg_property_size': df['myhome_floor_area_value'].mean(),
            f'{radius}km_median_beds': df['beds'].median(),
            f'{radius}km_median_baths': df['baths'].median(),
            f'{radius}km_price_to_income_ratio': df['sale_price'].median() / 50000,  # Assuming median income of 50,000
            f'{radius}km_price_growth_rate': ((df['sale_price'].mean() / df['first_list_price'].mean()) - 1) * 100,
        })
        
        return metrics
    except Exception as e:
        logging.error(f"Error calculating nearby metrics for radius {radius}: {str(e)}")
        return metrics

def calculate_market_trends(df):
    """
    Calculate market trends such as percent change over the last 30 days.
    """
    try:
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_sales = df[df['sale_date'] >= thirty_days_ago]
        older_sales = df[df['sale_date'] < thirty_days_ago]
        recent_avg = recent_sales['sale_price'].mean()
        older_avg = older_sales['sale_price'].mean()
        if older_avg and older_avg != 0:
            percent_change = ((recent_avg - older_avg) / older_avg) * 100
            return percent_change
        else:
            return None
    except Exception as e:
        logging.error(f"Error calculating market trends: {e}")
        return None

def calculate_price_benchmarks(df, lower_bound, upper_bound):
    """
    Calculate the ratio of average prices between two areas.
    """
    try:
        if upper_bound == 'overall':
            overall_avg = df['sale_price'].mean()
            lower_avg = df[df['sale_price'] >= lower_bound]['sale_price'].mean()
            return safe_divide(lower_avg, overall_avg)
        else:
            upper_avg = df[df['sale_price'] <= upper_bound]['sale_price'].mean()
            lower_avg = df[df['sale_price'] >= lower_bound]['sale_price'].mean()
            return safe_divide(lower_avg, upper_avg)
    except Exception as e:
        logging.error(f"Error calculating price benchmarks between {lower_bound}km and {upper_bound}km: {e}")
        return None

def calculate_price_trend(df, days):
    """
    Calculate the price trend over a specified number of days.
    """
    try:
        target_date = datetime.now() - timedelta(days=days)
        target_sales = df[df['sale_date'] >= target_date]
        trend = target_sales['sale_price'].mean()
        return trend
    except Exception as e:
        logging.error(f"Error calculating price trend over {days} days: {e}")
        return None

def generate_columns(data):
    """
    Generate all required columns/metrics for a property.
    """
    try:
        logging.info("Starting generate_columns function.")
        
        # Preprocess the property data
        preprocessed_data = preprocess_property_data(data)

        result = {
            'bedCategory': get_bed_category(preprocessed_data.get('beds', '0')),
            'bathCategory': get_bath_category(preprocessed_data.get('baths', '0')),
            'propertyTypeCategory': get_property_type_category(preprocessed_data.get('property_type', '')),
            'berCategory': ber_to_category(preprocessed_data.get('energy_rating', '')),
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
            for radius in [1, 3, 5]:
                nearby_props = fetch_nearby_properties(result['latitude'], result['longitude'], radius_km=radius)
                result[f'nearby_properties_count_within_{radius}km'] = len(nearby_props)
                if nearby_props:
                    nearby_metrics = calculate_nearby_metrics(nearby_props, radius)
                    result.update(nearby_metrics)
                else:
                    logging.warning(f"No nearby properties found within {radius}km to calculate metrics.")
        else:
            logging.warning("Latitude or longitude is missing, skipping nearby properties calculation.")

        # Remove None values
        result = {k: v for k, v in result.items() if v is not None}

        logging.info("Finished generate_columns function.")
        return result
    except Exception as e:
        logging.error(f"Error in generate_columns: {str(e)}")
        return {}

# ======================================
# Step 3: Data Processing Pipeline
# ======================================

def process_properties(output_csv_path):
    try:
        logging.info("Starting data processing pipeline.")

        # Fetch all properties at once
        response = supabase.table("scraped_property_data_v1").select("*").execute()
        all_properties = response.data

        total_properties = len(all_properties)
        logging.info(f"Total properties fetched: {total_properties}")

        if not all_properties:
            logging.warning("No properties fetched from Supabase. Exiting pipeline.")
            return

        # Convert to DataFrame for faster processing
        df_all = pd.DataFrame(all_properties)

        # Process properties
        processed_data = []
        skipped_properties = 0
        for idx, prop in df_all.iterrows():
            try:
                metrics = generate_columns(prop.to_dict())
                combined_data = {**prop.to_dict(), **metrics}
                processed_data.append(combined_data)
                if (idx + 1) % 100 == 0 or idx == len(df_all) - 1:
                    print(f"Processed {idx+1}/{len(df_all)} properties")
            except Exception as e:
                skipped_properties += 1
                logging.error(f"Failed to process property ID {prop.get('id')}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        # Create DataFrame and save to CSV
        df_processed = pd.DataFrame(processed_data)
        logging.info(f"Processed DataFrame shape: {df_processed.shape}")
        logging.info(f"Skipped properties: {skipped_properties}")
        
        # Only remove columns if they are completely empty (all NaN)
        columns_before = df_processed.shape[1]
        df_processed = df_processed.dropna(axis=1, how='all')
        columns_after = df_processed.shape[1]
        logging.info(f"Columns removed due to being completely empty: {columns_before - columns_after}")
        
        # Log information about columns with NaN values
        nan_counts = df_processed.isna().sum()
        columns_with_nans = nan_counts[nan_counts > 0]
        logging.info(f"Columns with NaN values:\n{columns_with_nans}")
        
        # Don't remove rows with NaN values
        logging.info(f"Final processed DataFrame shape: {df_processed.shape}")
        
        df_processed.to_csv(output_csv_path, index=False)
        logging.info("Data processing pipeline completed successfully.")

        return df_processed

    except Exception as e:
        logging.error(f"Error in process_properties: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# ======================================
# Step 4: Main Execution
# ======================================

def main():
    try:
        logging.info("Python script started.")

        # Start the data processing pipeline
        df_processed = process_properties(OUTPUT_CSV)

        logging.info(f"Data processing completed. Output saved to {OUTPUT_CSV}")

        # Preview processed data
        if df_processed is not None:
            logging.info("\nDataFrame Preview:")
            print(df_processed.head())

            logging.info("\nColumn Information:")
            column_info = pd.DataFrame({
                'Column Name': df_processed.columns,
                'Data Type': df_processed.dtypes
            })
            print(column_info)

            logging.info("\nStatistical Summary:")
            numeric_cols = df_processed.select_dtypes(include=['number']).columns
            stats = df_processed[numeric_cols].agg(['mean', 'median', 'max', 'min', 'std']).transpose()
            stats = stats.rename(columns={
                'mean': 'Mean', 'median': 'Median', 'max': 'Max', 'min': 'Min', 'std': 'Std Dev'
            })
            print(stats)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
