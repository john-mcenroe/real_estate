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
    # Define required columns with their expected data types
    required_columns = {
        'Sold Date': 'datetime',
        'Local Property Tax': 'numeric',
        'MyHome_Asking_Price': 'numeric',
        'MyHome_Beds': 'numeric',
        'MyHome_Baths': 'numeric',
        'First List Date': 'datetime',
        'Energy Rating': 'string',
        'MyHome_Latitude': 'numeric',
        'MyHome_Longitude': 'numeric'
    }

    # Ensure column names are stripped of leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Drop less important columns to streamline the dataset
    columns_to_drop = [
        # Transaction Value Metrics
        'transaction_value_avg_price_per_sqm_within_3km',
        'transaction_value_avg_price_per_sqm_within_5km',
        'transaction_value_median_sold_price_within_3km_90days',
        'transaction_value_p75_sold_price_within_3km_90days',
        # Time-Based Metrics
        'time_based_median_time_on_market_last_180days_within_5km',
        # Listing Activity Metrics (Removed as per user instruction)
        'listing_activity_median_asking_price_active_within_3km',
        'listing_activity_avg_price_changes_within_3km',
        'listing_activity_avg_days_on_market_active_within_5km',
        'listing_activity_percent_price_changes_within_5km',
        # Sales Velocity Metrics
        'sales_velocity_count_2024-05_within_3km',
        'sales_velocity_count_2024-03_within_3km',
        # Pricing Consistency Metrics
        'pricing_consistency_percent_minimal_changes_within_3km',
        'pricing_consistency_price_change_freq_within_3km',
        'pricing_consistency_avg_time_between_changes_within_5km',
        # Property Type Distribution Metrics
        'property_type_dist_Townhouse_percent_within_3km',
        'property_type_dist_Studio_percent_within_3km',
        'property_type_dist_Site_percent_within_3km',
        'property_type_dist_Studio_percent_within_5km',
        'property_type_dist_Houses_percent_within_5km',
        'property_type_dist_Site_percent_within_5km',
        'property_type_dist_Houses_percent_within_3km',
        # Property Condition Metrics
        'property_condition_ber_dist_15.0_within_3km'
    ]

    # Drop the columns if they exist in the dataframe
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    logging.info(f"Dropped columns: {', '.join([col for col in columns_to_drop if col in df.columns])}")

    # Parse 'Price Changes' to extract details and create new columns
    if 'Price Changes' in df.columns:
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

    # Calculate 'Days_on_Market' if both 'Sold Date' and 'First List Date' are available
    if 'Sold Date' in df.columns and 'First List Date' in df.columns:
        df['Days_on_Market'] = (df['Sold Date'] - df['First List Date']).dt.days
    else:
        df['Days_on_Market'] = np.nan
        logging.warning("Either 'Sold Date' or 'First List Date' is missing. 'Days_on_Market' set to NaN.")

    # Calculate price per square meter if not already present
    if 'price_per_square_meter' not in df.columns:
        df['price_per_square_meter'] = df.apply(safe_divide, axis=1)
        logging.info("Calculated 'price_per_square_meter'.")

    logging.info("Preprocessing dataframe completed successfully.")
    return df

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

    # Calculate distance to City Center (Assuming City Center coordinates)
    CITY_CENTER_LAT = 53.3498053  # Example: Dublin City Center latitude
    CITY_CENTER_LON = -6.2603097  # Example: Dublin City Center longitude
    distance_to_city_center = haversine_distance(current_lat, current_lon, CITY_CENTER_LAT, CITY_CENTER_LON)
    new_columns['geo_distance_city_center'] = distance_to_city_center

    # ======================================
    # Existing Metrics Calculations
    # ======================================

    # (Assuming existing metrics calculations are already included elsewhere in the code)
    # For demonstration purposes, only new metrics are included here.

    # ======================================
    # New Metrics Calculations
    # ======================================

    # 1. Price Adjustment Metrics
    # Number of Price Changes
    if 'Sold Asking Price' in row and pd.notna(row['Sold Asking Price']):
        if 'Asking Price' in row and pd.notna(row['Asking Price']):
            if 'Price Changes' in row and pd.notna(row['Price Changes']):
                num_price_changes = row['Price Changes'].count(';') + 1
            else:
                num_price_changes = 0
        else:
            num_price_changes = 0
    else:
        num_price_changes = 0
    new_columns['price_adjustment_num_changes'] = num_price_changes

    # Total Percentage Change
    if pd.notna(row['Asking Price']) and pd.notna(row['Sold Asking Price']) and row['Asking Price'] != 0:
        total_pct_change = ((row['Sold Asking Price'] - row['Asking Price']) / row['Asking Price']) * 100
    else:
        total_pct_change = np.nan
    new_columns['price_adjustment_total_pct_change'] = total_pct_change

    # Average Percentage Change per Adjustment
    if 'Price Changes' in df.columns and num_price_changes > 0 and pd.notna(row['Asking Price']):
        changes = row['Price Changes'].split(';')
        pct_changes = []
        for change in changes:
            match = re.search(r'€([\d,]+)', change)
            if match:
                price = float(match.group(1).replace(',', ''))
                pct_change = ((price - row['Asking Price']) / row['Asking Price']) * 100
                pct_changes.append(pct_change)
        if pct_changes:
            avg_pct_change = np.mean(pct_changes)
        else:
            avg_pct_change = np.nan
    else:
        avg_pct_change = np.nan
    new_columns['price_adjustment_avg_pct_change'] = avg_pct_change

    # 2. Geographic Proximity Metrics
    # Number of Sold Properties Within 1 km
    mask_within_1km = df['Dynamic Geo Distance (km)'] <= 1
    num_sold_within_1km = df[mask_within_1km].shape[0]
    new_columns['geo_num_sold_within_1km'] = num_sold_within_1km

    # Average Sold Price Within 1 km
    if 'Sold Price' in df.columns and not df[mask_within_1km]['Sold Price'].empty:
        avg_sold_price_within_1km = df[mask_within_1km]['Sold Price'].mean()
    else:
        avg_sold_price_within_1km = np.nan
    new_columns['geo_avg_sold_price_within_1km'] = avg_sold_price_within_1km

    # 3. Agency Performance Metrics
    if 'Agency Name' in df.columns and 'Agency Name' in row:
        agency = row['Agency Name']
        agency_mask = df['Agency Name'] == agency
        num_listings = df[agency_mask].shape[0]
        num_sales = df[agency_mask]['Sold Price'].notna().sum()
        success_rate = (num_sales / num_listings) * 100 if num_listings > 0 else np.nan
        if 'Sold Date' in df.columns and 'First List Date' in df.columns:
            selling_times = (df[agency_mask]['Sold Date'] - df[agency_mask]['First List Date']).dt.days
            avg_selling_time = selling_times.mean() if not selling_times.empty else np.nan
        else:
            avg_selling_time = np.nan
        new_columns['agency_num_listings'] = num_listings
        new_columns['agency_avg_selling_time'] = avg_selling_time
        new_columns['agency_success_rate'] = success_rate
    else:
        new_columns['agency_num_listings'] = np.nan
        new_columns['agency_avg_selling_time'] = np.nan
        new_columns['agency_success_rate'] = np.nan

    # 4. Price per Unit Metrics
    if 'MyHome_Beds' in row and row['MyHome_Beds'] > 0:
        new_columns['price_per_bedroom'] = row['MyHome_Asking_Price'] / row['MyHome_Beds']
    else:
        new_columns['price_per_bedroom'] = np.nan

    if 'MyHome_Baths' in row and row['MyHome_Baths'] > 0:
        new_columns['price_per_bathroom'] = row['MyHome_Asking_Price'] / row['MyHome_Baths']
    else:
        new_columns['price_per_bathroom'] = np.nan

    if 'MyHome_Floor_Area_Value' in row and row['MyHome_Floor_Area_Value'] > 0:
        new_columns['price_per_sqm'] = row['MyHome_Asking_Price'] / row['MyHome_Floor_Area_Value']
    else:
        new_columns['price_per_sqm'] = np.nan

    # 5. Floor Area Metrics
    if 'MyHome_Beds' in row and row['MyHome_Beds'] > 0:
        new_columns['floor_area_per_bedroom'] = row['MyHome_Floor_Area_Value'] / row['MyHome_Beds']
    else:
        new_columns['floor_area_per_bedroom'] = np.nan

    if 'MyHome_Baths' in row and row['MyHome_Baths'] > 0:
        new_columns['floor_area_per_bathroom'] = row['MyHome_Floor_Area_Value'] / row['MyHome_Baths']
    else:
        new_columns['floor_area_per_bathroom'] = np.nan

    # Floor Area Variance in Area (within 3 km)
    mask_within_3km = df['Dynamic Geo Distance (km)'] <= 3
    if 'MyHome_Floor_Area_Value' in df.columns and not df[mask_within_3km]['MyHome_Floor_Area_Value'].empty:
        floor_area_variance = df[mask_within_3km]['MyHome_Floor_Area_Value'].var()
    else:
        floor_area_variance = np.nan
    new_columns['floor_area_variance_within_3km'] = floor_area_variance

    # 6. Market Activity Metrics
    mask_active_within_5km = (df['Dynamic Geo Distance (km)'] <= 5) & (df['Sold Price'].isna())
    num_active_within_5km = df[mask_active_within_5km].shape[0]
    new_columns['market_activity_active_listings_within_5km'] = num_active_within_5km

    if pd.notna(row['Sold Date']):
        sold_last_month = row['Sold Date'] - timedelta(days=30)
        mask_sold_last_month = (df['Dynamic Geo Distance (km)'] <= 5) & (df['Sold Date'] >= sold_last_month)
        num_sold_last_month = df[mask_sold_last_month].shape[0]
    else:
        num_sold_last_month = np.nan
    new_columns['market_activity_sold_last_month'] = num_sold_last_month

    # Average Time on Market for Area (within 5 km)
    if 'Days_on_Market' in df.columns:
        avg_time_on_market = df.loc[mask_active_within_5km, 'Days_on_Market'].mean()
    else:
        avg_time_on_market = np.nan
    new_columns['market_activity_avg_time_on_market'] = avg_time_on_market

    # 7. Transaction Timing Metrics
    if pd.notna(row['Sold Date']):
        sale_month = row['Sold Date'].strftime("%B")
        new_columns['transaction_timing_sale_month'] = sale_month

        median_sell_time = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                             (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=180)))]['Days_on_Market'].median()
        new_columns['transaction_timing_median_sell_time_last_6_months'] = median_sell_time

        # Sales Velocity Trend (simplified as sales in recent months)
        sales_trend = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                        (df['Sold Date'] >= (row['Sold Date'] - timedelta(days=180)))]
        monthly_sales = sales_trend['Sold Date'].dt.to_period('M').value_counts().sort_index()
        for period, count in monthly_sales.items():
            new_columns[f'transaction_timing_sales_trend_{period}'] = count
    else:
        new_columns['transaction_timing_sale_month'] = np.nan
        new_columns['transaction_timing_median_sell_time_last_6_months'] = np.nan

    # 8. Comparative Market Metrics
    area_median_sold_price = df[df['Dynamic Geo Distance (km)'] <= 5]['Sold Price'].median()
    if area_median_sold_price and not np.isnan(area_median_sold_price):
        comparative_sold_price = row['Sold Price'] / area_median_sold_price
    else:
        comparative_sold_price = np.nan
    new_columns['comparative_market_comparative_sold_price'] = comparative_sold_price

    area_median_asking_price = df[df['Dynamic Geo Distance (km)'] <= 5]['Asking Price'].median()
    if area_median_asking_price and not np.isnan(area_median_asking_price):
        comparative_asking_price = row['Asking Price'] / area_median_asking_price
    else:
        comparative_asking_price = np.nan
    new_columns['comparative_market_comparative_asking_price'] = comparative_asking_price

    if area_median_asking_price and not np.isnan(area_median_asking_price):
        price_competitiveness = row['Asking Price'] / area_median_asking_price
    else:
        price_competitiveness = np.nan
    new_columns['comparative_market_price_competitiveness'] = price_competitiveness

    # 9. Price Trend Metrics
    if 'Sold Date' in df.columns and 'Sold Price' in df.columns and pd.notna(row['Sold Date']):
        one_year_ago = row['Sold Date'] - timedelta(days=365)
        median_price_trend_5km = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                                    (df['Sold Date'] >= one_year_ago)]['Sold Price'].median()
        new_columns['price_trend_median_price_trend_5km'] = median_price_trend_5km

        median_price_trend_10km = df[(df['Dynamic Geo Distance (km)'] <= 10) & 
                                     (df['Sold Date'] >= one_year_ago)]['Sold Price'].median()
        new_columns['price_trend_median_price_trend_10km'] = median_price_trend_10km

        if not pd.isna(median_price_trend_5km):
            adjusted_market_rate_price = row['Sold Price'] - median_price_trend_5km
        else:
            adjusted_market_rate_price = np.nan
        new_columns['price_trend_adjusted_market_rate_price'] = adjusted_market_rate_price
    else:
        new_columns['price_trend_median_price_trend_5km'] = np.nan
        new_columns['price_trend_median_price_trend_10km'] = np.nan
        new_columns['price_trend_adjusted_market_rate_price'] = np.nan

    # 10. Price Volume Metrics
    if pd.notna(row['Sold Date']):
        one_year_ago = row['Sold Date'] - timedelta(days=365)
        sales_last_year = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                             (df['Sold Date'] >= one_year_ago)]['Sold Price'].count()
        sales_previous_year = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                                 (df['Sold Date'] >= (one_year_ago - timedelta(days=365))) & 
                                 (df['Sold Date'] < one_year_ago)]['Sold Price'].count()
        if sales_previous_year > 0:
            sales_volume_growth = ((sales_last_year - sales_previous_year) / sales_previous_year) * 100
        else:
            sales_volume_growth = np.nan
    else:
        sales_volume_growth = np.nan
    new_columns['price_volume_sales_volume_growth'] = sales_volume_growth

    if pd.notna(row['Sold Date']):
        monthly_sales_volume_series = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                                         (df['Sold Date'] >= one_year_ago)]['Sold Price'].resample('M', on='Sold Date').count()
        monthly_sales_volume = monthly_sales_volume_series.to_dict()
        for month, count in monthly_sales_volume.items():
            month_str = month.strftime("%Y_%m")
            new_columns[f'price_volume_monthly_sales_volume_{month_str}'] = count
    else:
        new_columns['price_volume_monthly_sales_volume'] = np.nan

    if 'Sold Price' in df.columns and pd.notna(row['Sold Price']):
        sales_price_growth = df[(df['Dynamic Geo Distance (km)'] <= 5) & 
                                (df['Sold Date'] >= one_year_ago)]['Sold Price'].pct_change().mean() * 100
    else:
        sales_price_growth = np.nan
    new_columns['price_volume_avg_sales_price_growth'] = sales_price_growth

    # 11. Unique Location Insights Metrics
    # Proximity to Amenities (Assuming amenities data is not available, hence setting as NaN)
    new_columns['unique_location_proximity_to_amenities'] = np.nan

    # Neighborhood Popularity Index (Simplified as number of sales and average price)
    neighborhood_sales = df[(df['Dynamic Geo Distance (km)'] <= 5)]['Sold Price'].count()
    neighborhood_avg_price = df[(df['Dynamic Geo Distance (km)'] <= 5)]['Sold Price'].mean()
    if neighborhood_sales and neighborhood_avg_price:
        neighborhood_popularity_index = neighborhood_sales * neighborhood_avg_price
    else:
        neighborhood_popularity_index = np.nan
    new_columns['unique_location_neighborhood_popularity_index'] = neighborhood_popularity_index

    # Accessibility Score (Assuming data is not available, hence setting as NaN)
    new_columns['unique_location_accessibility_score'] = np.nan

    return new_columns

def add_all_metrics(df):
    """
    Add all calculated metrics to the dataframe.
    """
    # Preprocess the dataframe
    df = preprocess_dataframe(df)
    logging.info("Preprocessing dataframe completed.")

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
    logging.info("Metrics concatenated to the original DataFrame.")

    # Drop the 'Dynamic Geo Distance (km)' column as it's no longer needed
    if 'Dynamic Geo Distance (km)' in df.columns:
        df.drop(columns=['Dynamic Geo Distance (km)'], inplace=True)
        logging.info("Removed 'Dynamic Geo Distance (km)' column.")

    logging.info("All metrics have been successfully added to the dataframe.")
    return df

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

# ======================================
# Step 3: Main Execution
# ======================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Logging configured.")

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
