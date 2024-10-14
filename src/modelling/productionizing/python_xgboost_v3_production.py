import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load the data
input_path = '/Users/johnmcenroe/Documents/programming_misc/real_estate/data/processed/scraped_dublin/added_metadata/full_run_predictions_xgboost_v3.csv'

# Check if file exists and its size
print(f"File exists: {os.path.exists(input_path)}")
print(f"File size: {os.path.getsize(input_path)} bytes")

# Load the data
df = pd.read_csv(input_path)

# After loading the data
print(f"Initial number of rows: {len(df)}")
print(f"Initial number of columns: {len(df.columns)}")
print("\nFirst few rows of the dataframe:")
print(df.head())
print("\nDataframe info:")
print(df.info())

# Define target and features
target = 'sale_price'
features = [
    'beds', 'baths', 'myhome_floor_area_value', 'latitude', 'longitude',
    'energy_rating_numeric', 'bedCategory', 'bathCategory', 'propertyTypeCategory', 'berCategory', 'sizeCategory',
    'nearby_properties_count_within_1km',
    'avg_sold_price_within_1km', 'median_sold_price_within_1km',
    'avg_asking_price_within_1km', 'median_asking_price_within_1km',
    'avg_price_delta_within_1km', 'median_price_delta_within_1km',
    'avg_price_per_sqm_within_1km', 'median_price_per_sqm_within_1km',
    'avg_bedrooms_within_1km', 'avg_bathrooms_within_1km',
    'nearby_properties_count_within_3km',
    'avg_sold_price_within_3km', 'median_sold_price_within_3km',
    'avg_asking_price_within_3km', 'median_asking_price_within_3km',
    'avg_price_delta_within_3km', 'median_price_delta_within_3km',
    'avg_price_per_sqm_within_3km', 'median_price_per_sqm_within_3km',
    'avg_bedrooms_within_3km', 'avg_bathrooms_within_3km',
    'nearby_properties_count_within_5km',
    'avg_sold_price_within_5km', 'median_sold_price_within_5km',
    'avg_asking_price_within_5km', 'median_asking_price_within_5km',
    'avg_price_delta_within_5km', 'median_price_delta_within_5km',
    'avg_price_per_sqm_within_5km', 'median_price_per_sqm_within_5km',
    'avg_bedrooms_within_5km', 'avg_bathrooms_within_5km',
    'property_type', 'energy_rating'
]

# Ensure all required features are present
available_features = [f for f in features if f in df.columns]
if len(available_features) != len(features):
    missing_features = set(features) - set(available_features)
    print(f"\nWarning: The following features are missing from the dataset: {missing_features}")
    features = available_features

print(f"\nNumber of features: {len(features)}")
print(f"Target column: {target}")

# Handle missing values and infinities
df = df.replace([np.inf, -np.inf], np.nan)
print(f"\nNumber of rows after replacing inf: {len(df)}")

# Print missing value information
print("\nMissing values in each column:")
print(df[features + [target]].isnull().sum())

# Separate features and target
X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create imputers
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values
X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])

X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])
X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

# Create and train the model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Make predictions and calculate R^2 score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Add these lines for debugging
print(f"y_test shape: {y_test.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"y_test sample: {y_test.head()}")
print(f"y_pred sample: {y_pred[:5]}")

# Check for NaN or infinite values
print(f"NaN in y_test: {np.isnan(y_test).any()}")
print(f"NaN in y_pred: {np.isnan(y_pred).any()}")
print(f"Inf in y_test: {np.isinf(y_test).any()}")
print(f"Inf in y_pred: {np.isinf(y_pred).any()}")

# Save the model
model_path = os.path.join(os.path.dirname(input_path), 'xgboost_model.joblib')
joblib.dump(model, model_path)
print(f"Saved trained model to {model_path}")

# Function to make predictions
def predict_price(**kwargs):
    input_data = pd.DataFrame([kwargs])
    prediction = model.predict(input_data)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    print("\nRunning example prediction:")
    example_price = predict_price(
        beds=3,
        baths=2,
        myhome_floor_area_value=120,
        latitude=53.3498,
        longitude=-6.2603,
        energy_rating_numeric=9,
        bedCategory="3 Bed",
        bathCategory="2 Bath",
        propertyTypeCategory="House",
        berCategory="B",
        sizeCategory="Large",
        nearby_properties_count_within_1km=50,
        avg_sold_price_within_1km=400000,
        median_sold_price_within_1km=380000,
        avg_asking_price_within_1km=420000,
        median_asking_price_within_1km=400000,
        avg_price_delta_within_1km=20000,
        median_price_delta_within_1km=20000,
        avg_price_per_sqm_within_1km=4000,
        median_price_per_sqm_within_1km=3800,
        avg_bedrooms_within_1km=3,
        avg_bathrooms_within_1km=2,
        nearby_properties_count_within_3km=150,
        avg_sold_price_within_3km=380000,
        median_sold_price_within_3km=360000,
        avg_asking_price_within_3km=400000,
        median_asking_price_within_3km=380000,
        avg_price_delta_within_3km=20000,
        median_price_delta_within_3km=20000,
        avg_price_per_sqm_within_3km=3800,
        median_price_per_sqm_within_3km=3600,
        avg_bedrooms_within_3km=3,
        avg_bathrooms_within_3km=2,
        nearby_properties_count_within_5km=300,
        avg_sold_price_within_5km=360000,
        median_sold_price_within_5km=340000,
        avg_asking_price_within_5km=380000,
        median_asking_price_within_5km=360000,
        avg_price_delta_within_5km=20000,
        median_price_delta_within_5km=20000,
        avg_price_per_sqm_within_5km=3600,
        median_price_per_sqm_within_5km=3400,
        avg_bedrooms_within_5km=3,
        avg_bathrooms_within_5km=2,
        property_type="House",
        energy_rating="B1"
    )
    print(f"Predicted price: €{example_price:,.2f}")