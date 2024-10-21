# ======================================
# Required Libraries
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import widgets, interact
import re
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import logging

# ======================================
# Step 1: Define Helper Functions
# ======================================

def safe_divide(row):
    """
    Safely divide 'Asking Price' by 'MyHome_Floor_Area_Value'.
    Returns None if either value is NaN or if division by zero occurs.
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        float or None: The price per square meter or None.
    """
    if pd.isna(row['Asking Price']) or pd.isna(row['MyHome_Floor_Area_Value']) or row['MyHome_Floor_Area_Value'] == 0:
        return None
    else:
        return row['Asking Price'] / row['MyHome_Floor_Area_Value']

def get_price_details(price_changes: str):
    """
    Extracts the asking price at sale, sold price, sold date, and first list date from the Price Changes column.

    Args:
        price_changes (str): The value from the Price Changes column.

    Returns:
        tuple: A tuple containing:
            - Asking Price at Sale (float or None)
            - Sold Price (float or None)
            - Sold Date (str or None)
            - First List Date (str or None)
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

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert coordinates to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def ber_to_numeric(ber):
    """
    Convert BER rating to a numeric value.
    A1 is the best (highest value), G is the worst (lowest value).
    """
    ber_order = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E2', 'F', 'G']
    if pd.isna(ber) or ber == '--' or ber not in ber_order:
        return np.nan
    return float(len(ber_order) - ber_order.index(ber))

def preprocess_dataframe(df):
    """
    Preprocess the dataframe by converting columns to appropriate data types and handling missing values.
    Also parses the 'Price Changes' column to extract relevant details.
    """
    try:
        # Define required columns with their expected data types
        required_columns = {
            'Sold Date': 'datetime',
            # Removed 'Price Changes' as 'numeric' since we will parse it
            'Local Property Tax': 'numeric',
            'MyHome_Asking_Price': 'numeric',
            'MyHome_Beds': 'numeric',
            'MyHome_Baths': 'numeric',
            'First List Date': 'datetime',
            'Energy Rating': 'string',  # Changed to string
            'MyHome_Latitude': 'numeric',
            'MyHome_Longitude': 'numeric'
        }

        # Ensure column names are stripped of leading/trailing spaces
        df.columns = df.columns.str.strip()

        # Parse 'Price Changes' to extract details and create new columns
        price_details = df['Price Changes'].apply(get_price_details).apply(pd.Series)
        price_details.columns = ['Sold Asking Price', 'Sold Price', 'Sold Date', 'First List Date']
        df = pd.concat([df, price_details], axis=1)

        # Convert 'Sold Asking Price' and 'Sold Price' to numeric types
        df['Sold Asking Price'] = pd.to_numeric(df['Sold Asking Price'], errors='coerce')
        df['Sold Price'] = pd.to_numeric(df['Sold Price'], errors='coerce')

        # Convert 'Sold Date' to datetime
        df['Sold Date'] = pd.to_datetime(df['Sold Date'], format="%a %b %d %Y", errors='coerce')

        # Convert 'First List Date' to datetime
        df['First List Date'] = pd.to_datetime(df['First List Date'], format="%a %b %d %Y", errors='coerce')

        # Remove 'Price Changes' column if it's no longer needed
        if 'Price Changes' in df.columns:
            df.drop(columns=['Price Changes'], inplace=True)
            logging.info("Removed 'Price Changes' column after parsing.")

        # Remove 'MyHome_BER_Rating' if it exists
        if 'MyHome_BER_Rating' in df.columns:
            df.drop(columns=['MyHome_BER_Rating'], inplace=True)
            logging.info("Removed 'MyHome_BER_Rating' column as it's no longer needed.")

        # Convert and handle each required column
        for col, dtype in required_columns.items():
            if col in df.columns:
                if dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'numeric':
                    # Remove non-numeric characters before conversion
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce'
                    )
                elif dtype == 'string':
                    df[col] = df[col].astype(str)
            else:
                logging.warning(f"'{col}' column is missing. Filling with NaN or default values.")
                if dtype == 'datetime':
                    df[col] = pd.NaT
                elif dtype == 'numeric':
                    df[col] = np.nan
                elif dtype == 'string':
                    df[col] = ''

        # Handle 'First List Date' missing values
        if 'First List Date' in df.columns:
            missing_first_list_date = df['First List Date'].isna().sum()
            if missing_first_list_date > 0:
                if 'Sold Date' in df.columns:
                    df['First List Date'] = df['First List Date'].fillna(df['Sold Date'] - timedelta(days=30))
                    logging.info("Filled missing 'First List Date' with 'Sold Date' minus 30 days.")
                else:
                    df['First List Date'] = df['First List Date'].fillna(datetime.now() - timedelta(days=30))
                    logging.info("Filled missing 'First List Date' with current date minus 30 days.")

        # Handle other numeric columns
        numeric_columns = [
            'Asking Price', 'Beds', 'Baths', 'MyHome_Asking_Price',
            'MyHome_Beds', 'MyHome_Baths', 'MyHome_Floor_Area_Value',
            'Local Property Tax', 'Sold Asking Price',
            'Sold Price', 'price_per_square_meter'
        ]

        # Clean and convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                # Remove non-numeric characters except decimal points
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce'
                )
                # Fill missing values with median or a default value if median is NaN
                median_value = df[col].median()
                if pd.isna(median_value):
                    median_value = 0  # Default value if median is NaN
                df[col] = df[col].fillna(median_value)
                logging.info(f"Filled missing values in '{col}' with median: {median_value}")
            else:
                df[col] = np.nan
                logging.warning(f"'{col}' column is missing. Filling with NaN.")

        # Convert BER ratings to numeric values
        if 'Energy Rating' in df.columns:
            df['Energy Rating'] = df['Energy Rating'].apply(ber_to_numeric)
            logging.info("Converted BER ratings to numeric values.")

        # Ensure 'Geo Distance (km)' is removed as it will be calculated per row
        if 'Geo Distance (km)' in df.columns:
            df.drop(columns=['Geo Distance (km)'], inplace=True)
            logging.info("Removed precomputed 'Geo Distance (km)' to calculate it per row.")

        # Initialize 'Dynamic Geo Distance (km)' as 0.0 (placeholder)
        df['Dynamic Geo Distance (km)'] = 0.0

        # Handle 'Property Type' consistency
        if 'Property Type' in df.columns:
            df['Property Type'] = df['Property Type'].astype(str).str.title().str.strip()
        else:
            df['Property Type'] = 'Unknown'
            logging.warning("'Property Type' column is missing. Filled with 'Unknown'.")

        return df

    except Exception as e:
        logging.error(f"Error in preprocess_dataframe: {e}")
        raise

def calculate_metrics_for_row(df, row):
    """
    Calculate all metrics for a single row and return a dictionary of new columns.
    """
    new_columns = {}

    # Extract the current property's latitude and longitude
    current_lat = row.get('MyHome_Latitude', np.nan)
    current_lon = row.get('MyHome_Longitude', np.nan)

    if pd.isna(current_lat) or pd.isna(current_lon):
        logging.warning(f"Missing latitude or longitude for row index {row.name}. Skipping metrics calculation.")
        return new_columns  # Return empty metrics if location data is missing

    # Calculate distances from the current property to all other properties
    df = df.copy()  # To avoid SettingWithCopyWarning
    df['Dynamic Geo Distance (km)'] = df.apply(
        lambda r: haversine_distance(current_lat, current_lon, r['MyHome_Latitude'], r['MyHome_Longitude'])
        if pd.notna(r['MyHome_Latitude']) and pd.notna(r['MyHome_Longitude']) else np.nan,
        axis=1
    )

    # Define metric categories with distances and time frames
    transaction_volume_categories = {
        '3km_90days': {'distance': 3, 'days': 90},
        '5km_180days': {'distance': 5, 'days': 180}
    }

    transaction_value_categories = {
        '3km_90days': {'distance': 3, 'days': 90},
        '5km_180days': {'distance': 5, 'days': 180}
    }

    price_dynamics_categories = {
        '3km_90days': {'distance': 3, 'days': 90},
        '5km_180days': {'distance': 5, 'days': 180}
    }

    time_based_categories = {
        '3km': {'distance': 3},
        '5km': {'distance': 5}
    }

    property_condition_categories = {
        '3km': {'distance': 3},
        '5km': {'distance': 5}
    }

    house_size_benchmark_categories = {
        '3km_90days': {'distance': 3, 'days': 90},
        '5km_180days': {'distance': 5, 'days': 180}
    }

    property_type_distribution_categories = {
        '3km': {'distance': 3},
        '5km': {'distance': 5}
    }

    listing_activity_categories = {
        '3km': {'distance': 3},
        '5km': {'distance': 5}
    }

    sales_velocity_categories = {
        '3km': {'distance': 3, 'months': 6},
        '5km': {'distance': 5, 'months': 6}
    }

    pricing_consistency_categories = {
        '3km': {'distance': 3},
        '5km': {'distance': 5, 'days': 180}
    }

    # Define a list of property types considered as houses
    house_types = [
        'End of Terrace', 'Terrace', 'Semi-D', 'Detached',
        'Duplex', 'Bungalow', 'Townhouse', 'Houses'
    ]

    # Transaction Volume Metrics
    for key, params in transaction_volume_categories.items():
        distance = params['distance']
        days = params['days']
        mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days)))
        filtered = df[mask]
        num_sold = filtered.shape[0]
        new_columns[f'transaction_volume_num_sold_within_{distance}km_{days}days'] = num_sold

        if days >= 30:
            months = days / 30
            avg_monthly = num_sold / months if months > 0 else np.nan
            new_columns[f'transaction_volume_avg_monthly_transactions_within_{distance}km'] = avg_monthly
        else:
            new_columns[f'transaction_volume_avg_monthly_transactions_within_{distance}km'] = np.nan

        if key == '3km_90days':
            total_listings = df[
                (df['Dynamic Geo Distance (km)'] <= distance) &
                (df['First List Date'] >= (row['Sold Date'] - timedelta(days=days)))
            ].shape[0]
            new_columns[f'transaction_volume_total_listings_last_{days}days_within_{distance}km'] = total_listings

    # Transaction Value Metrics
    for key, params in transaction_value_categories.items():
        distance = params['distance']
        days = params['days']
        mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days)))
        filtered = df[mask]

        if 'Sold Price' in df.columns and not filtered.empty:
            median_sold_price = filtered['Sold Price'].median()
            new_columns[f'transaction_value_median_sold_price_within_{distance}km_{days}days'] = median_sold_price

            percentile_75_sold_price = filtered['Sold Price'].quantile(0.75)
            new_columns[f'transaction_value_p75_sold_price_within_{distance}km_{days}days'] = percentile_75_sold_price
        else:
            new_columns[f'transaction_value_median_sold_price_within_{distance}km_{days}days'] = np.nan
            new_columns[f'transaction_value_p75_sold_price_within_{distance}km_{days}days'] = np.nan

        if 'price_per_square_meter' in df.columns and not filtered.empty:
            avg_price_sqm = filtered['price_per_square_meter'].mean()
            new_columns[f'transaction_value_avg_price_per_sqm_within_{distance}km'] = avg_price_sqm
        else:
            new_columns[f'transaction_value_avg_price_per_sqm_within_{distance}km'] = np.nan

        if key == '5km_180days' and 'Sold Price' in df.columns and not filtered.empty:
            avg_sold_price = filtered['Sold Price'].mean()
            new_columns[f'transaction_value_avg_sold_price_within_{distance}km_{days}days'] = avg_sold_price

        if key == '3km_90days' and 'Asking Price' in df.columns and not filtered.empty:
            median_asking_price = filtered['Asking Price'].median()
            new_columns[f'transaction_value_median_asking_price_within_{distance}km_{days}days'] = median_asking_price

    # Price Dynamics Metrics
    for key, params in price_dynamics_categories.items():
        distance = params['distance']
        days = params['days']
        mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days)))
        filtered = df[mask].copy()

        if 'Sold Price' in filtered.columns and 'Asking Price' in filtered.columns and not filtered.empty:
            filtered['Price_Difference'] = filtered['Sold Price'] - filtered['Asking Price']
            avg_price_diff = filtered['Price_Difference'].mean()
            new_columns[f'price_dynamics_avg_price_diff_within_{distance}km_{days}days'] = avg_price_diff

            filtered_non_zero = filtered[filtered['Asking Price'] != 0]
            if not filtered_non_zero.empty:
                filtered_non_zero['Price_Change_Pct'] = (filtered_non_zero['Price_Difference'] / filtered_non_zero['Asking Price']) * 100
                median_price_change_pct = filtered_non_zero['Price_Change_Pct'].median()
                new_columns[f'price_dynamics_median_price_change_pct_within_{distance}km_{days}days'] = median_price_change_pct

                percent_above = (filtered_non_zero['Sold Price'] > filtered_non_zero['Asking Price']).mean() * 100
                new_columns[f'price_dynamics_percent_sold_above_asking_within_{distance}km_{days}days'] = percent_above
            else:
                new_columns[f'price_dynamics_median_price_change_pct_within_{distance}km_{days}days'] = np.nan
                new_columns[f'price_dynamics_percent_sold_above_asking_within_{distance}km_{days}days'] = np.nan
        else:
            new_columns[f'price_dynamics_avg_price_diff_within_{distance}km_{days}days'] = np.nan
            new_columns[f'price_dynamics_median_price_change_pct_within_{distance}km_{days}days'] = np.nan
            new_columns[f'price_dynamics_percent_sold_above_asking_within_{distance}km_{days}days'] = np.nan

    # Time-Based Metrics
    for key, params in time_based_categories.items():
        distance = params['distance']
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask].copy()

        if 'Sold Date' in filtered.columns and 'First List Date' in filtered.columns:
            # Calculate Days_on_Market safely
            filtered['Days_on_Market'] = (filtered['Sold Date'] - filtered['First List Date']).dt.days
            avg_days = filtered['Days_on_Market'].mean() if not filtered['Days_on_Market'].empty else np.nan
            median_days = filtered['Days_on_Market'].median() if not filtered['Days_on_Market'].empty else np.nan
            new_columns[f'time_based_avg_days_on_market_within_{distance}km'] = avg_days
            new_columns[f'time_based_median_days_on_market_within_{distance}km'] = median_days
        else:
            new_columns[f'time_based_avg_days_on_market_within_{distance}km'] = np.nan
            new_columns[f'time_based_median_days_on_market_within_{distance}km'] = np.nan

        if key == '5km':
            if 'Days_on_Market' in filtered.columns and not filtered['Days_on_Market'].empty:
                avg_time_to_sell = filtered['Days_on_Market'].mean()
                new_columns['time_based_avg_time_to_sell_within_5km'] = avg_time_to_sell

                mask_days = (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=180)))
                filtered_days = df[mask & mask_days]
                if 'Days_on_Market' in filtered_days.columns and not filtered_days['Days_on_Market'].empty:
                    median_time_on_market = filtered_days['Days_on_Market'].median()
                    new_columns['time_based_median_time_on_market_last_180days_within_5km'] = median_time_on_market
                else:
                    new_columns['time_based_median_time_on_market_last_180days_within_5km'] = np.nan
            else:
                new_columns['time_based_avg_time_to_sell_within_5km'] = np.nan
                new_columns['time_based_median_time_on_market_last_180days_within_5km'] = np.nan

        if key == '3km':
            mask_listings = (df['Dynamic Geo Distance (km)'] <= distance) & \
                            (df['First List Date'] >= (row['Sold Date'] - timedelta(days=90)))
            total_listings = df[mask_listings].shape[0]
            new_columns['time_based_total_listings_last_90days_within_3km'] = total_listings

    # Property Condition Metrics
    # Using 'Energy Rating' instead of 'MyHome_BER_Rating'
    # Energy Rating Distribution within 3 km
    distance = 3
    if 'Dynamic Geo Distance (km)' in df.columns and 'Energy Rating' in df.columns:
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask]
        if not filtered.empty:
            ber_distribution = filtered['Energy Rating'].value_counts(normalize=True).sort_index() * 100
            for ber, percent in ber_distribution.items():
                if pd.notna(ber) and ber != '--':
                    new_columns[f'property_condition_ber_dist_{ber}_within_{distance}km'] = percent
        else:
            logging.warning("No data available for BER distribution within 3 km.")
    else:
        logging.warning("Required columns for BER distribution are missing.")

    # Percentage of BER Rating A within 5 km
    distance = 5
    if 'Dynamic Geo Distance (km)' in df.columns and 'Energy Rating' in df.columns:
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask]
        if not filtered.empty:
            # Assuming BER Rating A corresponds to the highest rating, e.g., 15 (from ber_to_numeric)
            ber_a_value = df['Energy Rating'].max()  # Adjust based on 'ber_to_numeric' mapping
            ber_a_percent = (filtered['Energy Rating'] == ber_a_value).mean() * 100
            new_columns[f'property_condition_percent_ber_A_within_{distance}km'] = ber_a_percent
        else:
            new_columns[f'property_condition_percent_ber_A_within_{distance}km'] = np.nan
    else:
        logging.warning("Required columns for BER rating percentage are missing.")

    # Average BER Rating within 3 km
    distance = 3
    if 'Dynamic Geo Distance (km)' in df.columns and 'Energy Rating' in df.columns:
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask]
        if not filtered.empty:
            avg_ber = filtered['Energy Rating'].mean()
            new_columns[f'property_condition_avg_ber_within_{distance}km'] = avg_ber
        else:
            new_columns[f'property_condition_avg_ber_within_{distance}km'] = np.nan
    else:
        logging.warning("Required columns for average BER rating are missing.")

    # Percentage of Energy Efficient Homes within 3 km (BER Rating >=5 assuming 5-7 are efficient)
    distance = 3
    if 'Dynamic Geo Distance (km)' in df.columns and 'Energy Rating' in df.columns:
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask]
        if not filtered.empty:
            energy_efficient_percent = ((filtered['Energy Rating'] >= 5) & (filtered['Energy Rating'] <= 7)).mean() * 100
            new_columns[f'property_condition_percent_energy_efficient_within_{distance}km'] = energy_efficient_percent
        else:
            new_columns[f'property_condition_percent_energy_efficient_within_{distance}km'] = np.nan
    else:
        logging.warning("Required columns for energy efficiency percentage are missing.")

    # Median BER Rating within 5 km in Last 180 Days
    distance = 5
    days = 180
    if 'Dynamic Geo Distance (km)' in df.columns and 'Sold Date' in df.columns and 'Energy Rating' in df.columns:
        mask = (df['Dynamic Geo Distance (km)'] <= distance) & \
               (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days)))
        filtered = df[mask]
        if not filtered.empty:
            median_ber = filtered['Energy Rating'].median()
            new_columns[f'property_condition_median_ber_within_{distance}km_{days}days'] = median_ber
        else:
            new_columns[f'property_condition_median_ber_within_{distance}km_{days}days'] = np.nan
    else:
        logging.warning("Required columns for median BER rating are missing.")

    # House Size Benchmark Metrics
    for key, params in house_size_benchmark_categories.items():
        distance = params['distance']
        days = params['days']
        mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days)))
        filtered = df[mask]

        # Average Floor Area
        if 'MyHome_Floor_Area_Value' in filtered.columns and not filtered['MyHome_Floor_Area_Value'].empty:
            avg_floor_area = filtered['MyHome_Floor_Area_Value'].mean()
            new_columns[f'house_size_benchmark_avg_floor_area_within_{distance}km_{days}days'] = avg_floor_area
        else:
            avg_floor_area = np.nan
            new_columns[f'house_size_benchmark_avg_floor_area_within_{distance}km_{days}days'] = np.nan

        # Median Beds
        if 'MyHome_Beds' in filtered.columns and not filtered['MyHome_Beds'].empty:
            median_beds = filtered['MyHome_Beds'].median()
            new_columns[f'house_size_benchmark_median_beds_within_{distance}km_{days}days'] = median_beds
        else:
            median_beds = np.nan
            new_columns[f'house_size_benchmark_median_beds_within_{distance}km_{days}days'] = np.nan

        # Median Baths
        if 'MyHome_Baths' in filtered.columns and not filtered['MyHome_Baths'].empty:
            median_baths = filtered['MyHome_Baths'].median()
            new_columns[f'house_size_benchmark_median_baths_within_{distance}km_{days}days'] = median_baths
        else:
            median_baths = np.nan
            new_columns[f'house_size_benchmark_median_baths_within_{distance}km_{days}days'] = np.nan

        # Std Floor Area
        if 'MyHome_Floor_Area_Value' in filtered.columns and not filtered['MyHome_Floor_Area_Value'].empty:
            std_floor_area = filtered['MyHome_Floor_Area_Value'].std()
            new_columns[f'house_size_benchmark_std_floor_area_within_{distance}km'] = std_floor_area
        else:
            new_columns[f'house_size_benchmark_std_floor_area_within_{distance}km'] = np.nan

        # Percent Larger Than Average Floor Area
        if 'MyHome_Floor_Area_Value' in df.columns and not pd.isna(avg_floor_area):
            percent_larger = (row['MyHome_Floor_Area_Value'] > avg_floor_area) * 100
            new_columns[f'house_size_benchmark_percent_larger_than_avg_within_{distance}km_{days}days'] = percent_larger
        else:
            new_columns[f'house_size_benchmark_percent_larger_than_avg_within_{distance}km_{days}days'] = np.nan

        # Comparison Indicators
        if 'MyHome_Floor_Area_Value' in df.columns and not pd.isna(avg_floor_area):
            comparison = float(row['MyHome_Floor_Area_Value'] > avg_floor_area)
            new_columns[f'house_size_comparison_larger_than_avg_within_{distance}km_{days}days'] = comparison
        else:
            new_columns[f'house_size_comparison_larger_than_avg_within_{distance}km_{days}days'] = np.nan

        if 'MyHome_Beds' in df.columns and not pd.isna(median_beds):
            comparison = float(row['MyHome_Beds'] > median_beds)
            new_columns[f'house_size_comparison_beds_above_median_within_{distance}km_{days}days'] = comparison
        else:
            new_columns[f'house_size_comparison_beds_above_median_within_{distance}km_{days}days'] = np.nan

        if 'MyHome_Baths' in df.columns and not pd.isna(median_baths):
            comparison = float(row['MyHome_Baths'] > median_baths)
            new_columns[f'house_size_comparison_baths_above_median_within_{distance}km_{days}days'] = comparison
        else:
            new_columns[f'house_size_comparison_baths_above_median_within_{distance}km_{days}days'] = np.nan

    # Property Type Distribution Metrics
    for key, params in property_type_distribution_categories.items():
        distance = params['distance']
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask]

        if 'Property Type' in filtered.columns and not filtered['Property Type'].empty:
            # Distribution of Property Types
            property_counts = filtered['Property Type'].value_counts(normalize=True) * 100
            for prop_type, percent in property_counts.items():
                new_columns[f'property_type_dist_{prop_type}_percent_within_{distance}km'] = percent

            # Percentage of Each Property Type
            for prop_type in df['Property Type'].unique():
                percent = (filtered['Property Type'] == prop_type).mean() * 100 if not filtered.empty else np.nan
                new_columns[f'property_type_percent_{prop_type}_within_{distance}km'] = percent

            # Median Property Type Count within 3 km in Last 90 Days
            if key == '3km' and 'First List Date' in df.columns:
                days = 90
                mask_days = (df['First List Date'] >= (row['Sold Date'] - timedelta(days=days)))
                filtered_days = df[mask & mask_days]
                if 'Property Type' in filtered_days.columns and not filtered_days['Property Type'].empty:
                    median_prop_count = filtered_days['Property Type'].value_counts().median()
                    new_columns[f'property_type_median_count_within_{distance}km_{days}days'] = median_prop_count
                else:
                    new_columns[f'property_type_median_count_within_{distance}km_{days}days'] = np.nan

            # Diversity Index of Property Types within 5 km
            if key == '5km':
                diversity = filtered['Property Type'].nunique()
                new_columns[f'property_type_diversity_within_{distance}km'] = diversity

            # Number of Unique Property Types within 3 km
            if key == '3km':
                unique_prop_types = filtered['Property Type'].nunique()
                new_columns[f'property_type_unique_count_within_{distance}km'] = unique_prop_types
        else:
            logging.warning("Required columns for property type distribution are missing or empty.")

    # Listing Activity Metrics
    for key, params in listing_activity_categories.items():
        distance = params['distance']

        if key == '3km':
            # Active Listings: Listings without a Sold Date
            mask_active = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'].isna())
            num_active = df[mask_active].shape[0]
            new_columns[f'listing_activity_num_active_within_{distance}km'] = num_active

            # Median Asking Price of Active Listings within 3 km
            if 'Asking Price' in df.columns and not df.loc[mask_active, 'Asking Price'].empty:
                median_asking = df.loc[mask_active, 'Asking Price'].median()
                new_columns[f'listing_activity_median_asking_price_active_within_{distance}km'] = median_asking
            else:
                new_columns[f'listing_activity_median_asking_price_active_within_{distance}km'] = np.nan

            # Average Number of Price Changes within 3 km
            if 'Price Changes' in df.columns and not df.loc[mask_active, 'Price Changes'].empty:
                avg_price_changes = df.loc[mask_active, 'Price Changes'].fillna(0).mean()
                new_columns[f'listing_activity_avg_price_changes_within_{distance}km'] = avg_price_changes
            else:
                new_columns[f'listing_activity_avg_price_changes_within_{distance}km'] = np.nan

        if key == '5km':
            # Active Listings: Listings without a Sold Date
            mask_active = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'].isna())
            active_listings = df[mask_active].copy()
            if 'First List Date' in active_listings.columns and not active_listings['First List Date'].isna().all():
                active_listings['Days_on_Market'] = (row['Sold Date'] - active_listings['First List Date']).dt.days
                avg_time_on_market = active_listings['Days_on_Market'].mean() if not active_listings['Days_on_Market'].empty else np.nan
                new_columns[f'listing_activity_avg_days_on_market_active_within_{distance}km'] = avg_time_on_market
            else:
                new_columns[f'listing_activity_avg_days_on_market_active_within_{distance}km'] = np.nan

            # Percentage of Listings with Price Changes within 5 km
            if 'Price Changes' in df.columns and not df[df['Dynamic Geo Distance (km)'] <= distance]['Price Changes'].empty:
                percent_price_changes = ((df['Dynamic Geo Distance (km)'] <= distance) & df['Price Changes'].notna()).mean() * 100
                new_columns[f'listing_activity_percent_price_changes_within_{distance}km'] = percent_price_changes
            else:
                new_columns[f'listing_activity_percent_price_changes_within_{distance}km'] = np.nan

    # Sales Velocity Metrics
    for key, params in sales_velocity_categories.items():
        distance = params['distance']
        months = params['months']
        start_date = row['Sold Date'] - pd.DateOffset(months=months)

        mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] >= start_date)
        filtered = df[mask]

        # Average Sales Velocity (Properties Sold per Month)
        sales_velocity_avg = filtered.shape[0] / months if months > 0 else np.nan
        new_columns[f'sales_velocity_avg_properties_sold_per_month_within_{distance}km'] = sales_velocity_avg

        # Median Sales Velocity
        if not filtered.empty:
            monthly_sales = filtered.groupby(filtered['Sold Date'].dt.to_period('M')).size()
            sales_velocity_median = monthly_sales.median() if not monthly_sales.empty else np.nan
            new_columns[f'sales_velocity_median_properties_sold_per_month_within_{distance}km'] = sales_velocity_median
        else:
            new_columns[f'sales_velocity_median_properties_sold_per_month_within_{distance}km'] = np.nan

        # Sales Velocity Trend over Last 6 Months within 3 km
        if key == '3km' and not filtered.empty:
            monthly_sales = filtered.groupby(filtered['Sold Date'].dt.to_period('M')).size()
            for period, count in monthly_sales.items():
                new_columns[f'sales_velocity_count_{period}_within_{distance}km'] = count

        # Seasonal Sales Velocity Patterns within 5 km
        if key == '5km' and not filtered.empty:
            # Ensure that 'Month' is correctly set using .loc to avoid SettingWithCopyWarning
            filtered = filtered.copy()
            filtered['Month'] = filtered['Sold Date'].dt.month
            seasonal_sales = filtered.groupby('Month').size()
            for month, count in seasonal_sales.items():
                new_columns[f'sales_velocity_seasonal_month_{month}_within_{distance}km'] = count

        # Comparative Sales Velocity between 3 km and 5 km Radii
        if key == '5km':
            # Calculate sales velocity for 3 km if not already present
            mask_3km = (df['Dynamic Geo Distance (km)'] <= 3) & (df['Sold Date'] >= (row['Sold Date'] - pd.DateOffset(months=6)))
            filtered_3km = df[mask_3km]
            sales_velocity_avg_3km = filtered_3km.shape[0] / 6 if 6 > 0 else np.nan
            new_columns['sales_velocity_comparative_3km_vs_5km'] = sales_velocity_avg - sales_velocity_avg_3km

    # Pricing Consistency Metrics
    for key, params in pricing_consistency_categories.items():
        distance = params['distance']
        days = params.get('days', None)

        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask].copy()

        # Average Asking Price vs. Sold Price Ratio within distance
        if 'Sold Price' in filtered.columns and 'Asking Price' in filtered.columns and not filtered.empty:
            filtered_non_zero = filtered[filtered['Asking Price'] != 0]
            if not filtered_non_zero.empty:
                asking_sold_ratio = filtered_non_zero['Sold Price'] / filtered_non_zero['Asking Price']
                avg_ratio = asking_sold_ratio.mean()
                new_columns[f'pricing_consistency_avg_asking_sold_ratio_within_{distance}km'] = avg_ratio
            else:
                new_columns[f'pricing_consistency_avg_asking_sold_ratio_within_{distance}km'] = np.nan
        else:
            new_columns[f'pricing_consistency_avg_asking_sold_ratio_within_{distance}km'] = np.nan

        # Median Price Adjustment Ratio within 5 km in Last 180 Days
        if key == '5km' and days:
            mask_days = (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days)))
            filtered_days = df[mask & mask_days]
            if 'Sold Price' in filtered_days.columns and 'Asking Price' in filtered_days.columns and not filtered_days.empty:
                filtered_days_non_zero = filtered_days[filtered_days['Asking Price'] != 0]
                if not filtered_days_non_zero.empty:
                    median_ratio = (filtered_days_non_zero['Sold Price'] / filtered_days_non_zero['Asking Price']).median()
                    new_columns[f'pricing_consistency_median_asking_sold_ratio_within_{distance}km_{days}days'] = median_ratio
                else:
                    new_columns[f'pricing_consistency_median_asking_sold_ratio_within_{distance}km_{days}days'] = np.nan
            else:
                new_columns[f'pricing_consistency_median_asking_sold_ratio_within_{distance}km_{days}days'] = np.nan

        # Percentage of Listings with Consistent Pricing (Minimal Changes) within 3 km
        if key == '3km':
            consistent_threshold = 5  # Define what constitutes minimal change, e.g., <=5%
            if 'Price Changes' in filtered.columns and 'Asking Price' in filtered.columns and not filtered.empty:
                filtered_non_zero = filtered[filtered['Asking Price'] != 0].copy()
                if not filtered_non_zero.empty:
                    filtered_non_zero['Price_Change_Pct'] = (filtered_non_zero['Price Changes'] / filtered_non_zero['Asking Price']) * 100
                    percent_consistent = (filtered_non_zero['Price_Change_Pct'].abs() <= consistent_threshold).mean() * 100
                    new_columns[f'pricing_consistency_percent_minimal_changes_within_{distance}km'] = percent_consistent
                else:
                    new_columns[f'pricing_consistency_percent_minimal_changes_within_{distance}km'] = np.nan
            else:
                new_columns[f'pricing_consistency_percent_minimal_changes_within_{distance}km'] = np.nan

        # Average Time Between Price Changes within 5 km
        if key == '5km':
            if 'Price Changes' in filtered.columns and not filtered['Price Changes'].empty:
                avg_time_between_changes = filtered['Price Changes'].fillna(0).mean()
                new_columns[f'pricing_consistency_avg_time_between_changes_within_{distance}km'] = avg_time_between_changes
            else:
                new_columns[f'pricing_consistency_avg_time_between_changes_within_{distance}km'] = np.nan

        # Price Change Frequency within 3 km
        if key == '3km':
            if 'Price Changes' in filtered.columns and not filtered['Price Changes'].empty:
                price_change_freq = (filtered['Price Changes'].fillna(0) > 0).mean() * 100
                new_columns[f'pricing_consistency_price_change_freq_within_{distance}km'] = price_change_freq
            else:
                new_columns[f'pricing_consistency_price_change_freq_within_{distance}km'] = np.nan
    
    # 1. Market Trends
    days_back = 180
    days_period = 30

    # Calculate the average sold price for the last 30 days
    mask_last_30_days = (df['Dynamic Geo Distance (km)'] <= 5) & \
                        (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days_period)))
    filtered_last_30_days = df[mask_last_30_days]
    avg_price_last_30_days = filtered_last_30_days['Sold Price'].mean() if not filtered_last_30_days.empty else np.nan

    # Calculate the average sold price for the period 180-210 days ago
    mask_180_210_days = (df['Dynamic Geo Distance (km)'] <= 5) & \
                        (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=days_back + days_period))) & \
                        (df['Sold Date'] < (row['Sold Date'] - timedelta(days=days_back)))
    filtered_180_210_days = df[mask_180_210_days]
    avg_price_180_210_days = filtered_180_210_days['Sold Price'].mean() if not filtered_180_210_days.empty else np.nan

    # Calculate the percentage change
    if pd.notna(avg_price_last_30_days) and pd.notna(avg_price_180_210_days) and avg_price_180_210_days != 0:
        percent_change = ((avg_price_last_30_days - avg_price_180_210_days) / avg_price_180_210_days) * 100
    else:
        percent_change = np.nan

    new_columns['market_trends_percent_change_30day_avg'] = percent_change

    # 3. Price vs. Dataset Benchmarks
    if 'Sold Price' in df.columns:
        # Calculate median sold price within 3KM
        mask_3km = (df['Dynamic Geo Distance (km)'] <= 3)
        median_3km = df.loc[mask_3km, 'Sold Price'].median()

        # Calculate median sold price within 10KM
        mask_10km = (df['Dynamic Geo Distance (km)'] <= 10)
        median_10km = df.loc[mask_10km, 'Sold Price'].median()

        # Calculate overall median sold price
        overall_median = df['Sold Price'].median()

        # New column: Ratio of 3KM median to 10KM median
        new_columns['price_benchmarks_ratio_3km_to_10km'] = median_3km / median_10km if median_10km else np.nan

        # New column: Ratio of 10KM median to overall median
        new_columns['price_benchmarks_ratio_10km_to_overall'] = median_10km / overall_median if overall_median else np.nan

    # 4. Value to Asking Ratio in Area
    # Value to Asking Ratio in Area for 1km, 3km, and 5km
    for distance in [1, 3, 5]:
        mask = (df['Dynamic Geo Distance (km)'] <= distance)
        filtered = df[mask]
        if not filtered.empty and 'Sold Price' in filtered.columns and 'Asking Price' in filtered.columns:
            value_asking_ratio = (filtered['Sold Price'] / filtered['Asking Price']).mean()
            new_columns[f'price_dynamics_area_value_asking_ratio_{distance}km'] = value_asking_ratio
        else:
            new_columns[f'price_dynamics_area_value_asking_ratio_{distance}km'] = np.nan

    # Price Trend Metrics
    # Calculate price trend over 30 and 90 days within 3km and 5km
    for period, distance in [(30, 3), (30, 5), (90, 3), (90, 5)]:
        if 'Dynamic Geo Distance (km)' in df.columns and 'Sold Date' in df.columns and 'Sold Price' in df.columns:
            mask_recent = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=period)))
            mask_previous = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'] < (row['Sold Date'] - timedelta(days=period))) & (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=period * 2)))
            avg_price_recent = df[mask_recent]['Sold Price'].mean()
            avg_price_previous = df[mask_previous]['Sold Price'].mean()

            if pd.notna(avg_price_recent) and pd.notna(avg_price_previous) and avg_price_previous != 0:
                price_trend_change = (avg_price_recent - avg_price_previous) / avg_price_previous * 100
            else:
                price_trend_change = np.nan

            new_columns[f'price_trend_{period}_days_{distance}km'] = price_trend_change
        else:
            logging.warning(f"Missing required columns for price trend calculation ({period} days, {distance}km)")
            new_columns[f'price_trend_{period}_days_{distance}km'] = np.nan

    # Sold-to-Asking Price Ratio
    for distance in [1, 3, 5]:
        if 'Dynamic Geo Distance (km)' in df.columns and 'Sold Price' in df.columns and 'Asking Price' in df.columns:
            mask = (df['Dynamic Geo Distance (km)'] <= distance)
            filtered = df[mask]
            if not filtered.empty:
                sold_to_asking_ratio = (filtered['Sold Price'] / filtered['Asking Price']).mean()
                new_columns[f'sold_to_asking_ratio_{distance}km'] = sold_to_asking_ratio
            else:
                new_columns[f'sold_to_asking_ratio_{distance}km'] = np.nan
        else:
            logging.warning(f"Missing required columns for sold-to-asking price ratio calculation ({distance}km)")
            new_columns[f'sold_to_asking_ratio_{distance}km'] = np.nan

    # Time to Sale
    for distance in [3, 5]:
        if 'Dynamic Geo Distance (km)' in df.columns and 'Sold Date' in df.columns and 'First List Date' in df.columns:
            mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Sold Date'].notna()) & (df['First List Date'].notna())
            filtered = df[mask]
            if not filtered.empty:
                time_to_sale = (filtered['Sold Date'] - filtered['First List Date']).dt.days.mean()
                new_columns[f'avg_time_to_sale_{distance}km'] = time_to_sale
            else:
                new_columns[f'avg_time_to_sale_{distance}km'] = np.nan
        else:
            logging.warning(f"Missing required columns for time to sale calculation ({distance}km)")
            new_columns[f'avg_time_to_sale_{distance}km'] = np.nan

    # Price Fluctuation
    for distance in [3, 5]:
        if 'Dynamic Geo Distance (km)' in df.columns and 'Price Changes' in df.columns and 'Asking Price' in df.columns:
            mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Price Changes'].notna())
            filtered = df[mask]
            if not filtered.empty:
                price_fluctuation = ((filtered['Price Changes'].astype(float)) / filtered['Asking Price']).mean()
                new_columns[f'avg_asking_price_fluctuation_{distance}km'] = price_fluctuation
            else:
                new_columns[f'avg_asking_price_fluctuation_{distance}km'] = np.nan
        else:
            logging.warning(f"Missing required columns for price fluctuation calculation ({distance}km)")
            new_columns[f'avg_asking_price_fluctuation_{distance}km'] = np.nan

    # Percentage of listings with price drop
    for distance in [3, 5]:
        if 'Dynamic Geo Distance (km)' in df.columns and 'Price Changes' in df.columns:
            mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['Price Changes'].notna())
            filtered = df[mask]
            if not filtered.empty:
                percent_price_drop = (filtered['Price Changes'].astype(float) < 0).mean() * 100
                new_columns[f'percent_listings_with_price_drop_{distance}km'] = percent_price_drop
            else:
                new_columns[f'percent_listings_with_price_drop_{distance}km'] = np.nan
        else:
            logging.warning(f"Missing required columns for price drop percentage calculation ({distance}km)")
            new_columns[f'percent_listings_with_price_drop_{distance}km'] = np.nan

    # Market Saturation Metrics
    for distance in [3, 5]:
        if 'Dynamic Geo Distance (km)' in df.columns and 'First List Date' in df.columns and 'Sold Date' in df.columns:
            new_listings_mask = (df['Dynamic Geo Distance (km)'] <= distance) & (df['First List Date'] >= (row['Sold Date'] - timedelta(days=30)))
            new_listings = df[new_listings_mask].shape[0]
            total_listings = df[(df['Dynamic Geo Distance (km)'] <= distance)].shape[0]

            if total_listings > 0:
                new_columns[f'new_listings_last_30_days_{distance}km'] = new_listings
                new_columns[f'percent_active_listings_vs_sold_{distance}km'] = (new_listings / total_listings) * 100
            else:
                new_columns[f'new_listings_last_30_days_{distance}km'] = np.nan
                new_columns[f'percent_active_listings_vs_sold_{distance}km'] = np.nan
        else:
            logging.warning(f"Missing required columns for market saturation metrics calculation ({distance}km)")
            new_columns[f'new_listings_last_30_days_{distance}km'] = np.nan
            new_columns[f'percent_active_listings_vs_sold_{distance}km'] = np.nan
            
    # Add this at the beginning of the function to log available columns
    logging.info(f"Available columns: {df.columns.tolist()}")            

    return new_columns

def add_all_metrics(df):
    """
    Add all calculated metrics to the dataframe.
    """
    try:
        # Preprocess the dataframe
        df = preprocess_dataframe(df)

        # Initialize a list to store metrics dictionaries
        metrics_list = []

        # Iterate over each row to calculate metrics
        for idx, row in df.iterrows():
            logging.info(f"Calculating metrics for row {idx + 1}/{len(df)}")
            metrics = calculate_metrics_for_row(df, row)
            metrics_list.append(metrics)

        # Create a DataFrame from the list of metrics
        metrics_df = pd.DataFrame(metrics_list)

        # Concatenate metrics to the original DataFrame
        df = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

        # Drop the 'Dynamic Geo Distance (km)' column as it's no longer needed
        if 'Dynamic Geo Distance (km)' in df.columns:
            df.drop(columns=['Dynamic Geo Distance (km)'], inplace=True)

        logging.info("All metrics have been successfully added to the dataframe.")
        return df

    except Exception as e:
        logging.error(f"Error in add_all_metrics: {e}")
        raise

def get_column_info(df):
    """
    Return a DataFrame with column names and their data types.
    """
    return pd.DataFrame({'Column Name': df.columns, 'Data Type': df.dtypes})

def get_column_info_with_stats(df):
    """
    Return a DataFrame with column statistics including mean, median, 75th percentile, and non-null counts.
    """
    # Filter numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Create the base DataFrame with column names and data types
    stats = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes
    })

    # Calculate statistics only for numeric columns
    stats['Mean'] = df[numeric_columns].mean()
    stats['Median'] = df[numeric_columns].median()
    stats['75th Percentile'] = df[numeric_columns].quantile(0.75)

    # Calculate the count of non-null values for all columns
    stats['Non-null Count'] = df.notnull().sum()

    return stats

# ======================================
# Step 2: Data Processing Pipeline
# ======================================

def process_data(input_csv_path, output_csv_path):
    """
    Process the input CSV file and save the processed data to the output CSV file.

    Steps:
    1. Load the data.
    2. Clean and preprocess the data.
    3. Add new signals and metrics.
    4. Save the processed data to a new CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the processed CSV file.
    """
    try:
        # 2.1 Load the Data
        print("Loading data...")
        df = pd.read_csv(input_csv_path)
        print(f"Data loaded. Shape: {df.shape}")

        # 2.2 Add All Metrics
        print("Adding metrics...")
        df_with_metrics = add_all_metrics(df)
        print("Metrics added.")

        # 2.3 Save the Processed Data
        print(f"Saving processed data to {output_csv_path}...")
        df_with_metrics.to_csv(output_csv_path, index=False)
        print("Data saved successfully.")

    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        raise

# ======================================
# Step 3: Main Execution
# ======================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Example usage:
    # Define input and output file paths
    input_csv = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/scraped_property_results_Dublin_final_with_metadata_deduped.csv'
    output_csv = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/preprocess_output_test.csv'
    
    # Process the data
    process_data(input_csv, output_csv)
    
    # Optionally, perform a preview of the processed data
    try:
        processed_df = pd.read_csv(output_csv)
        print("\nFinal DataFrame Preview:")
        print(processed_df.head())
    except Exception as e:
        print(f"Failed to read the processed CSV file: {e}")
    
    # Display column information
    try:
        print("\nColumn Information:")
        print(get_column_info(processed_df))
    except Exception as e:
        print(f"Failed to retrieve column information: {e}")
    
    # Display statistical summary
    try:
        print("\nStatistical Summary:")
        print(get_column_info_with_stats(processed_df).head(30))
    except Exception as e:
        print(f"Failed to retrieve statistical summary: {e}")
