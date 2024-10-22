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
# Step 1.1: Test Mode Configuration
# ======================================

# Test Mode Settings
TEST_ROWS = 5
OUTPUT_CSV_TEST = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/processed_data_v3_test_output.csv'

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

def get_ber_category(ber_rating):
    if not ber_rating or pd.isna(ber_rating):
        return 'Unknown'
    
    ber_rating = str(ber_rating).upper()
    
    if ber_rating in ['A', 'B', 'C', 'D', 'E', 'F']:
        return ber_rating
    elif ber_rating == 'G':
        return 'F'  # Grouping G with F as the lowest category
    else:
        return 'Unknown'

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

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized version of haversine distance calculation.
    """
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def get_nearby_properties(df, lat, lon, radius_km=3):
    """
    Get nearby properties within a specified radius using vectorized operations.
    """
    df['distance'] = haversine_distance_vectorized(lat, lon, df['latitude'], df['longitude'])
    return df[df['distance'] <= radius_km]

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
        
        # Transaction Volume Metrics
        metrics[f'transaction_volume_num_sold_within_{radius}km'] = df['sale_price'].notna().sum()
        metrics[f'transaction_volume_avg_monthly_transactions_within_{radius}km'] = (
            df['sale_date'].notna().sum() / 6  # Assuming last 6 months
        )
        
        # Only calculate if sale_date is present
        if 'sale_date' in df.columns:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=180)
            metrics[f'transaction_volume_total_listings_last_180days_within_{radius}km'] = df[df['sale_date'] >= cutoff_date].shape[0]
        
        # Transaction Value Metrics
        metrics[f'transaction_value_median_sold_price_within_{radius}km'] = df['sale_price'].median()
        metrics[f'transaction_value_p75_sold_price_within_{radius}km'] = df['sale_price'].quantile(0.75)
        metrics[f'transaction_value_avg_price_per_sqm_within_{radius}km'] = df['price_per_square_meter'].mean()
        
        # Only calculate if sale_date is present
        if 'sale_date' in df.columns:
            recent_sales = df[df['sale_date'] >= (pd.Timestamp.now() - pd.Timedelta(days=180))]
            metrics[f'transaction_value_avg_sold_price_within_{radius}km_180days'] = recent_sales['sale_price'].mean()
        
        if 'sale_date' in df.columns:
            very_recent_sales = df[df['sale_date'] >= (pd.Timestamp.now() - pd.Timedelta(days=90))]
            metrics[f'transaction_value_median_asking_price_within_{radius}km_90days'] = very_recent_sales['asking_price'].median()
        
        # Price Dynamics Metrics
        df['price_diff'] = df['sale_price'] - df['asking_price']
        metrics[f'price_dynamics_avg_price_diff_within_{radius}km'] = df['price_diff'].mean()
        df['price_change_pct'] = (df['price_diff'] / df['asking_price']) * 100
        metrics[f'price_dynamics_median_price_change_pct_within_{radius}km'] = df['price_change_pct'].median()
        metrics[f'price_dynamics_percent_sold_above_asking_within_{radius}km'] = (df['sale_price'] > df['asking_price']).mean() * 100
        
        # Time-Based Metrics
        if 'first_list_date' in df.columns and 'sale_date' in df.columns:
            df['days_on_market'] = (df['sale_date'] - df['first_list_date']).dt.days
            metrics[f'time_based_avg_days_on_market_within_{radius}km'] = df['days_on_market'].mean()
            metrics[f'time_based_median_days_on_market_within_{radius}km'] = df['days_on_market'].median()
        
        # Property Condition Metrics
        metrics[f'property_condition_avg_ber_within_{radius}km'] = df['energy_rating'].apply(ber_to_numeric).mean()
        metrics[f'property_condition_percent_energy_efficient_within_{radius}km'] = ((df['energy_rating'].apply(ber_to_numeric) >= 5) & (df['energy_rating'].apply(ber_to_numeric) <=7 )).mean() * 100
        metrics[f'property_condition_median_ber_within_{radius}km'] = df['energy_rating'].apply(ber_to_numeric).median()
        ber_counts = df['energy_rating'].value_counts(normalize=True) * 100
        for ber, percent in ber_counts.items():
            metrics[f'property_condition_ber_dist_{ber}_within_{radius}km'] = percent
        
        # House Size Benchmark Metrics
        metrics[f'house_size_benchmark_avg_floor_area_within_{radius}km'] = df['myhome_floor_area_value'].mean()
        metrics[f'house_size_benchmark_median_beds_within_{radius}km'] = df['beds'].median()
        metrics[f'house_size_benchmark_median_baths_within_{radius}km'] = df['baths'].median()
        metrics[f'house_size_benchmark_std_floor_area_within_{radius}km'] = df['myhome_floor_area_value'].std()
        avg_floor_area = df['myhome_floor_area_value'].mean()
        metrics[f'house_size_benchmark_percent_larger_than_avg_within_{radius}km'] = (df['myhome_floor_area_value'] > avg_floor_area).mean() * 100
        
        # Property Type Distribution Metrics
        property_types = df['property_type'].value_counts(normalize=True) * 100
        for prop_type, percent in property_types.items():
            metrics[f'property_type_dist_{prop_type}_percent_within_{radius}km'] = percent
        metrics[f'property_type_diversity_within_{radius}km'] = df['property_type'].nunique()
        metrics[f'property_type_unique_count_within_{radius}km'] = df['property_type'].nunique()
        
        # Listing Activity Metrics
        active_listings = df[df['sale_price'].isna()]
        metrics[f'listing_activity_num_active_within_{radius}km'] = active_listings.shape[0]
        metrics[f'listing_activity_median_asking_price_active_within_{radius}km'] = active_listings['asking_price'].median()
        
        # Sales Velocity Metrics
        if 'sale_date' in df.columns:
            sales_velocity_avg = df['sale_date'].notna().sum() / 6  # Assuming last 6 months
            sales_velocity_median = df.groupby(df['sale_date'].dt.to_period('M')).size().median()
            metrics[f'sales_velocity_avg_properties_sold_per_month_within_{radius}km'] = sales_velocity_avg
            metrics[f'sales_velocity_median_properties_sold_per_month_within_{radius}km'] = sales_velocity_median
        
        # Pricing Consistency Metrics
        metrics[f'pricing_consistency_avg_asking_sold_ratio_within_{radius}km'] = safe_divide(df['sale_price'].sum(), df['asking_price'].sum())
        
        return metrics
    except Exception as e:
        # Suppress the specific error message about converting to numeric
        if "Cannot convert" not in str(e):
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

def generate_columns(data, df_all):
    try:
        logging.info("Starting generate_columns function.")
        
        # Preprocess the property data
        preprocessed_data = preprocess_property_data(data)

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

        # Calculate nearby properties metrics for 1km, 3km, and 5km
        if result['latitude'] is not None and result['longitude'] is not None:
            for radius in [1, 3, 5]:
                nearby_props = get_nearby_properties(df_all, result['latitude'], result['longitude'], radius_km=radius)
                result[f'nearby_properties_count_within_{radius}km'] = len(nearby_props)
                if not nearby_props.empty:
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
        logging.error(f"Error in generate_columns for property ID {data.get('id')}: {str(e)}")
        logging.error(traceback.format_exc())
        return {}  # Return an empty dict instead of raising an exception

# ======================================
# Step 3: Data Processing Pipeline (Test Version)
# ======================================

def process_test_properties(output_csv_path, test_rows=5):
    try:
        logging.info("Starting test data processing pipeline.")

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
        for idx, prop in df_all.iloc[:test_rows].iterrows():
            try:
                metrics = generate_columns(prop.to_dict(), df_all)
                combined_data = {**prop.to_dict(), **metrics}
                processed_data.append(combined_data)
                logging.info(f"Processed test property {idx+1}/{test_rows}")
            except Exception as e:
                skipped_properties += 1
                logging.error(f"Failed to process test property ID {prop.get('id')}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        # Create DataFrame and save to CSV
        df_processed = pd.DataFrame(processed_data)
        logging.info(f"Processed test DataFrame shape: {df_processed.shape}")
        logging.info(f"Skipped test properties: {skipped_properties}")
        
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
        logging.info(f"Final processed test DataFrame shape: {df_processed.shape}")
        
        df_processed.to_csv(output_csv_path, index=False)
        logging.info("Test data processing pipeline completed successfully.")

        return df_processed

    except Exception as e:
        logging.error(f"Error in process_test_properties: {e}")
        logging.error(traceback.format_exc())
        raise

# ======================================
# Step 4: Main Execution (Test Version)
# ======================================

def main_test():
    try:
        logging.info("Python test script started.")
        
        # Process test properties
        df_processed = process_test_properties(OUTPUT_CSV_TEST, test_rows=TEST_ROWS)

        logging.info(f"Test data processing completed. Output saved to {OUTPUT_CSV_TEST}")

        # Preview processed data
        if df_processed is not None:
            logging.info("\nDataFrame Preview:")
            print(df_processed)

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
        logging.error(f"Error in main test execution: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main_test()
